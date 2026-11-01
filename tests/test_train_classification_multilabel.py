"""Tests for train_classification.py's multi-label classification support
(added for m-bigearthnet, label_type=geobench.MultiLabelClassification) and
its interaction with the pre-existing single-label path (m-eurosat,
m-so2sat), to confirm the new is_multi_label branches don't change
single-label behavior.

torch/numpy are exercised directly (both are installed in this
environment, confirmed via `import torch` succeeding) -- segmentation_models_pytorch's
absence (noted in test_train_segmentation.py) does not apply to this
template, which only needs timm.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain.templates.train_classification import (
    ClassificationArrayDataset,
    build_loss,
    compute_accuracy_metrics,
    compute_class_frequencies,
    compute_multilabel_metrics,
)

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# compute_multilabel_metrics -- pure numpy, the multi-label equivalent of
# compute_accuracy_metrics
# ---------------------------------------------------------------------------


def test_compute_multilabel_metrics_perfect_predictions():
    target = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=np.int64)
    pred = target.copy()
    result = compute_multilabel_metrics(pred, target, n_classes=3)

    assert result["exact_match"] == 1.0
    assert result["micro_f1"] == 1.0
    assert result["macro_f1"] == 1.0
    assert all(f == 1.0 for f in result["per_class_f1"])


def test_compute_multilabel_metrics_all_wrong():
    target = np.array([[1, 0], [0, 1]], dtype=np.int64)
    pred = np.array([[0, 1], [1, 0]], dtype=np.int64)
    result = compute_multilabel_metrics(pred, target, n_classes=2)

    assert result["exact_match"] == 0.0
    assert result["micro_f1"] == 0.0


def test_compute_multilabel_metrics_exact_match_stricter_than_micro_f1():
    # 3 samples, 4 classes each. Every sample gets exactly one of its
    # active classes wrong -- exact_match is 0 (no sample's full label
    # SET matches) but most individual (sample, class) decisions are still
    # right, so micro_f1 should be well above 0. This is the concrete
    # scenario compute_multilabel_metrics's docstring says micro_f1 is for:
    # exact_match alone would read as "the model learned nothing" when it
    # mostly didn't.
    target = np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
    ], dtype=np.int64)
    pred = np.array([
        [1, 0, 0, 0],  # missed class 1
        [0, 1, 1, 0],  # perfect
        [1, 0, 0, 0],  # missed class 3
    ], dtype=np.int64)
    result = compute_multilabel_metrics(pred, target, n_classes=4)

    assert result["exact_match"] == pytest.approx(1 / 3)
    assert result["micro_f1"] > 0.5


def test_compute_multilabel_metrics_per_class_f1_none_for_absent_class():
    # Class index 2 never appears in target at all across any sample --
    # there's no denominator for its recall, so it must report None, not
    # a fabricated 0.0 or 1.0 (mirrors compute_accuracy_metrics's identical
    # "absent class -> None" contract for single-label).
    target = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int64)
    pred = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int64)
    result = compute_multilabel_metrics(pred, target, n_classes=3)

    assert result["per_class_f1"][2] is None
    assert result["per_class_f1"][0] == 1.0
    assert result["per_class_f1"][1] == 1.0


def test_compute_multilabel_metrics_handles_43_class_bigearthnet_shape():
    # m-bigearthnet's real shape: 43 classes. Nothing here should be
    # hardcoded to a small class count.
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, size=(50, 43)).astype(np.int64)
    pred = rng.integers(0, 2, size=(50, 43)).astype(np.int64)
    result = compute_multilabel_metrics(pred, target, n_classes=43)

    assert len(result["per_class_f1"]) == 43
    assert 0.0 <= result["micro_f1"] <= 1.0
    assert 0.0 <= result["exact_match"] <= 1.0


# ---------------------------------------------------------------------------
# compute_accuracy_metrics -- confirm the pre-existing single-label path is
# unchanged by the multi-label additions (no shared mutable state, no
# accidental branch leakage)
# ---------------------------------------------------------------------------


def test_compute_accuracy_metrics_still_works_for_single_label():
    pred = np.array([0, 1, 2, 1])
    target = np.array([0, 1, 1, 1])
    accuracy, per_class = compute_accuracy_metrics(pred, target, n_classes=3)

    assert accuracy == 0.75
    assert per_class[2] is None  # class 2 never appears in target


# ---------------------------------------------------------------------------
# compute_class_frequencies -- is_multi_label branch
# ---------------------------------------------------------------------------


def test_compute_class_frequencies_single_label_unchanged():
    import logging

    logger = logging.getLogger("test")
    labels = np.array([0, 0, 1, 2, 2, 2])
    freqs = compute_class_frequencies(labels, n_classes=3, logger=logger)

    assert freqs.tolist() == pytest.approx([2 / 6, 1 / 6, 3 / 6])


def test_compute_class_frequencies_multi_label_is_per_class_active_fraction():
    import logging

    logger = logging.getLogger("test")
    # 4 samples, 3 classes: class 0 active in all 4, class 1 in none,
    # class 2 in half -- these do NOT sum to 1 (classes aren't mutually
    # exclusive), unlike the single-label case above.
    labels = np.array([
        [1, 0, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 0, 0],
    ], dtype=np.float32)
    freqs = compute_class_frequencies(labels, n_classes=3, logger=logger, is_multi_label=True)

    assert freqs.tolist() == pytest.approx([1.0, 0.0, 0.5])


# ---------------------------------------------------------------------------
# build_loss -- CrossEntropyLoss (single-label) vs BCEWithLogitsLoss
# (multi-label)
# ---------------------------------------------------------------------------


def test_build_loss_single_label_is_cross_entropy():
    loss_fn = build_loss(n_classes=5, class_weights=None, label_smoothing=0.0, device=torch.device("cpu"))
    assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)


def test_build_loss_multi_label_is_bce_with_logits():
    loss_fn = build_loss(
        n_classes=43, class_weights=None, label_smoothing=0.0, device=torch.device("cpu"), is_multi_label=True
    )
    assert isinstance(loss_fn, torch.nn.BCEWithLogitsLoss)


def test_build_loss_multi_label_accepts_multihot_targets_without_raising():
    # The actual shape BCEWithLogitsLoss needs: (N, n_classes) float logits
    # and (N, n_classes) float {0,1} targets -- confirms the loss and the
    # label dtype train_classification.py now produces for multi-label
    # (see ClassificationArrayDataset below) are compatible end to end.
    loss_fn = build_loss(n_classes=4, class_weights=None, label_smoothing=0.0, device=torch.device("cpu"), is_multi_label=True)
    logits = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8, 4)).float()
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0.0


def test_build_loss_multi_label_pos_weight_shape_matches_n_classes():
    class_weights = [1.0, 2.0, 0.5]
    loss_fn = build_loss(
        n_classes=3, class_weights=class_weights, label_smoothing=0.0, device=torch.device("cpu"), is_multi_label=True
    )
    assert loss_fn.pos_weight.tolist() == pytest.approx(class_weights)


# ---------------------------------------------------------------------------
# ClassificationArrayDataset -- label dtype differs by is_multi_label
# ---------------------------------------------------------------------------


def test_dataset_single_label_yields_long_scalar_label():
    images = np.random.rand(3, 8, 8, 3).astype(np.float32)
    labels = np.array([0, 1, 2], dtype=np.int64)
    mean, std = np.zeros(3), np.ones(3)
    ds = ClassificationArrayDataset(images, labels, mean, std, augment="none", seed=0)

    _, label_tensor = ds[0]
    assert label_tensor.dtype == torch.long
    assert label_tensor.ndim == 0


def test_dataset_multi_label_yields_float_vector_label():
    images = np.random.rand(3, 8, 8, 3).astype(np.float32)
    labels = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=np.float32)
    mean, std = np.zeros(3), np.ones(3)
    ds = ClassificationArrayDataset(images, labels, mean, std, augment="none", seed=0, is_multi_label=True)

    _, label_tensor = ds[0]
    assert label_tensor.dtype == torch.float32
    assert label_tensor.shape == (3,)
    assert label_tensor.tolist() == [1.0, 0.0, 1.0]
