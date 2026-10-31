"""Tests for the m-pv4ger-seg crash investigation fix: class_weights
length was never validated anywhere (a mismatch reached
torch.nn.CrossEntropyLoss uncaught, as a bare RuntimeError deep inside a
training subprocess that had already spent run budget getting there),
and DataLoader had no drop_last guard (an unguarded last batch of exactly
1 sample crashes BatchNorm in training mode).

validate_class_weights and safe_to_drop_last are pure functions -- no
torch needed. The pre-flight integration through
gotrain.tools.training.launch_training is also torch-free: the rejection
happens before any subprocess is spawned.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import ledger, logging_utils
from gotrain.config_schema import ConfigValidationError, validate_class_weights
from gotrain.templates.train_classification import safe_to_drop_last as classification_safe_to_drop_last
from gotrain.templates.train_segmentation import safe_to_drop_last as segmentation_safe_to_drop_last
from gotrain.tools import final_eval, training


# ---------------------------------------------------------------------------
# validate_class_weights (pure)
# ---------------------------------------------------------------------------


def test_validate_class_weights_passes_on_matching_length():
    validate_class_weights("segmentation", [0.3, 1.7], n_classes=2)  # must not raise


def test_validate_class_weights_passes_when_none():
    validate_class_weights("segmentation", None, n_classes=2)  # must not raise


def test_validate_class_weights_skips_when_n_classes_unresolved():
    # n_classes=None means "couldn't resolve it cheaply" -- must not
    # block, per the design (the template's own defense-in-depth check
    # still catches a real mismatch).
    validate_class_weights("segmentation", [1.0, 2.0, 3.0], n_classes=None)  # must not raise


def test_validate_class_weights_rejects_too_few_entries():
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_class_weights("segmentation", [1.0], n_classes=2)
    assert "2 entries" in str(exc_info.value)
    assert "got 1" in str(exc_info.value)


def test_validate_class_weights_rejects_too_many_entries():
    # The exact failure mode under investigation: a 17-class weight list
    # (e.g. copied from m-so2sat) applied to a 2-class dataset.
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_class_weights("classification", [1.0] * 17, n_classes=2)
    assert "2 entries" in str(exc_info.value)
    assert "got 17" in str(exc_info.value)


# ---------------------------------------------------------------------------
# safe_to_drop_last (pure, both templates)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("safe_to_drop_last", [classification_safe_to_drop_last, segmentation_safe_to_drop_last])
class TestSafeToDropLast:
    def test_true_when_dataset_larger_than_batch(self, safe_to_drop_last):
        assert safe_to_drop_last(3000, 8) is True

    def test_true_when_exactly_one_batch(self, safe_to_drop_last):
        assert safe_to_drop_last(8, 8) is True

    def test_false_when_dataset_smaller_than_batch(self, safe_to_drop_last):
        # The silent-zero-training failure mode this guard exists to prevent.
        assert safe_to_drop_last(5, 8) is False

    def test_false_for_nz_cattle_scale_with_large_batch(self, safe_to_drop_last):
        # m-nz-cattle has only 524 train samples -- a batch_size the agent
        # is free to set larger than that must not silently skip training.
        assert safe_to_drop_last(524, 600) is False


# ---------------------------------------------------------------------------
# Pre-flight integration through launch_training (no subprocess spawned)
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setattr(logging_utils, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", logs_dir / "events.jsonl")
    monkeypatch.setattr(logging_utils, "STATUS_PATH", logs_dir / "status.json")
    monkeypatch.setattr(ledger, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(ledger, "LEDGER_PATH", logs_dir / "ledger.jsonl")
    monkeypatch.setattr(final_eval, "LOGS_DIR", logs_dir)
    return logs_dir


def test_launch_training_rejects_mismatched_class_weights_preflight(isolated_logs, monkeypatch):
    # Simulate a resolvable n_classes=2 (m-pv4ger-seg is binary) without
    # needing real downloaded GEO-Bench data.
    monkeypatch.setattr(
        training.datasets_tools, "get_dataset_info",
        lambda dataset: {"label_info": {"n_classes": 2}},
    )

    with pytest.raises(ConfigValidationError) as exc_info:
        training.launch_training("m-pv4ger-seg", "segmentation", {"class_weights": [1.0] * 17})
    assert "2 entries" in str(exc_info.value)

    # Rejected before any subprocess ran -- no ledger record.
    assert training.query_history("m-pv4ger-seg") == []


def test_launch_training_allows_matching_class_weights_past_preflight(isolated_logs, monkeypatch):
    monkeypatch.setattr(
        training.datasets_tools, "get_dataset_info",
        lambda dataset: {"label_info": {"n_classes": 2}},
    )
    # task_type="fake" so it actually completes without torch/real data --
    # proves the check passes through cleanly for a correct config.
    record = training.launch_training("m-pv4ger-seg", "fake", {"class_weights": [0.4, 1.6]})
    assert record["val_metric"] is not None


def test_launch_training_skips_check_gracefully_if_n_classes_unresolvable(isolated_logs, monkeypatch):
    def raise_not_found(dataset):
        raise ValueError(f"Dataset {dataset!r} not found")

    monkeypatch.setattr(training.datasets_tools, "get_dataset_info", raise_not_found)

    # Must not raise an unrelated ValueError from the lookup itself --
    # degrades to "can't check pre-flight" rather than a confusing crash.
    record = training.launch_training("m-pv4ger-seg", "fake", {"class_weights": [1.0] * 17})
    assert record["val_metric"] is not None


def test_launch_training_does_not_check_class_weights_for_fake_without_mock(isolated_logs):
    # task_type="fake" never calls get_dataset_info for this check at all
    # (no real dataset lookup needed) when class_weights matches nothing
    # in particular -- confirms the check is skipped cleanly, not just
    # gracefully degraded, for the smoke-test task type.
    record = training.launch_training("m-eurosat", "fake", {"class_weights": [1.0] * 99})
    assert record["val_metric"] is not None
