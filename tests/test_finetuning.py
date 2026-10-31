"""Tests for gotrain/finetuning.py's freezing logic, using a minimal
duck-typed fake model (a named_parameters()/parameters() pair, like
torch.nn.Module exposes) so the actual freezing/classification/counting
logic is exercised without needing torch installed.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import finetuning


class FakeParam:
    def __init__(self, n, requires_grad=True):
        self._n = n
        self.requires_grad = requires_grad

    def numel(self):
        return self._n


class FakeModel:
    """Duck-types torch.nn.Module's named_parameters()/parameters()."""

    def __init__(self, named_params):
        self._named_params = named_params

    def named_parameters(self):
        return iter(self._named_params)

    def parameters(self):
        return (p for _, p in self._named_params)


def _timm_like_model():
    # Mirrors a resnet-family timm model: "fc.weight"/"fc.bias" is the
    # head, everything else ("layer1.0.conv1.weight", ...) is backbone.
    return FakeModel([
        ("conv1.weight", FakeParam(100)),
        ("layer1.0.conv1.weight", FakeParam(200)),
        ("layer2.0.conv1.weight", FakeParam(300)),
        ("layer3.0.conv1.weight", FakeParam(400)),
        ("layer4.0.conv1.weight", FakeParam(500)),
        ("fc.weight", FakeParam(10)),
        ("fc.bias", FakeParam(1)),
    ])


def _smp_like_model():
    return FakeModel([
        ("encoder.conv1.weight", FakeParam(100)),
        ("encoder.layer1.weight", FakeParam(200)),
        ("decoder.block1.weight", FakeParam(150)),
        ("segmentation_head.0.weight", FakeParam(20)),
        ("segmentation_head.0.bias", FakeParam(2)),
    ])


# ---------------------------------------------------------------------------
# classify_parameters
# ---------------------------------------------------------------------------


def test_classify_parameters_splits_timm_head_from_backbone():
    groups = finetuning.classify_parameters(_timm_like_model(), "timm")
    head_names = [n for n, _ in groups["head"]]
    backbone_names = [n for n, _ in groups["backbone"]]
    assert head_names == ["fc.weight", "fc.bias"]
    assert "conv1.weight" in backbone_names
    assert "fc.weight" not in backbone_names


def test_classify_parameters_splits_smp_head_from_backbone():
    groups = finetuning.classify_parameters(_smp_like_model(), "smp")
    head_names = [n for n, _ in groups["head"]]
    assert head_names == ["segmentation_head.0.weight", "segmentation_head.0.bias"]
    assert len(groups["backbone"]) == 3  # encoder + decoder params, none of which are the head


def test_classify_parameters_rejects_unknown_family():
    with pytest.raises(ValueError):
        finetuning.classify_parameters(_timm_like_model(), "not_a_real_family")


# ---------------------------------------------------------------------------
# apply_fine_tuning_strategy
# ---------------------------------------------------------------------------


def test_full_strategy_makes_everything_trainable():
    model = _timm_like_model()
    summary = finetuning.apply_fine_tuning_strategy(model, "timm", "full")
    assert all(p.requires_grad for _, p in model.named_parameters())
    assert summary["trainable_parameters"] == summary["total_parameters"]
    assert summary["frozen_parameters"] == 0


def test_from_scratch_makes_everything_trainable():
    model = _timm_like_model()
    summary = finetuning.apply_fine_tuning_strategy(model, "timm", "from_scratch")
    assert all(p.requires_grad for _, p in model.named_parameters())
    assert summary["frozen_parameters"] == 0


def test_linear_probe_freezes_backbone_only():
    model = _timm_like_model()
    summary = finetuning.apply_fine_tuning_strategy(model, "timm", "linear_probe")

    groups = finetuning.classify_parameters(model, "timm")
    assert all(not p.requires_grad for _, p in groups["backbone"])
    assert all(p.requires_grad for _, p in groups["head"])

    expected_trainable = sum(p.numel() for _, p in groups["head"])
    assert summary["trainable_parameters"] == expected_trainable
    assert summary["frozen_parameters"] == summary["total_parameters"] - expected_trainable


def test_partial_unfreezes_last_fraction_of_backbone_plus_head():
    model = _timm_like_model()
    summary = finetuning.apply_fine_tuning_strategy(model, "timm", "partial", partial_unfreeze_fraction=0.4)

    groups = finetuning.classify_parameters(model, "timm")
    backbone = groups["backbone"]  # 5 backbone params -> unfreeze last max(1, int(5*0.4))=2
    frozen_names = [n for n, p in backbone if not p.requires_grad]
    unfrozen_names = [n for n, p in backbone if p.requires_grad]

    assert len(unfrozen_names) == 2
    assert unfrozen_names == [backbone[-2][0], backbone[-1][0]]
    assert len(frozen_names) == 3
    assert all(p.requires_grad for _, p in groups["head"])  # head always trainable
    assert 0 < summary["trainable_fraction"] < 1


def test_partial_always_unfreezes_at_least_one_backbone_param():
    # fraction so small it would round to 0 -- must still unfreeze >=1.
    model = _timm_like_model()
    finetuning.apply_fine_tuning_strategy(model, "timm", "partial", partial_unfreeze_fraction=0.01)
    groups = finetuning.classify_parameters(model, "timm")
    assert sum(1 for _, p in groups["backbone"] if p.requires_grad) >= 1


def test_apply_fine_tuning_strategy_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        finetuning.apply_fine_tuning_strategy(_timm_like_model(), "timm", "not_a_real_strategy")


def test_smp_linear_probe_freezes_encoder_and_decoder_not_head():
    model = _smp_like_model()
    finetuning.apply_fine_tuning_strategy(model, "smp", "linear_probe")
    groups = finetuning.classify_parameters(model, "smp")
    assert all(not p.requires_grad for _, p in groups["backbone"])
    assert all(p.requires_grad for _, p in groups["head"])


# ---------------------------------------------------------------------------
# trainable_parameters
# ---------------------------------------------------------------------------


def test_trainable_parameters_excludes_frozen_params():
    model = _timm_like_model()
    finetuning.apply_fine_tuning_strategy(model, "timm", "linear_probe")
    trainable = finetuning.trainable_parameters(model)
    assert all(p.requires_grad for p in trainable)
    assert len(trainable) == 2  # fc.weight, fc.bias
