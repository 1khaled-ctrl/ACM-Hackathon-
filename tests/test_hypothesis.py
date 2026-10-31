"""Tests for structured experiment hypotheses (see prompts/system_prompt.md
step 3/4 and gotrain/config_schema.py::validate_hypothesis +
gotrain/ledger.py::_evaluate_hypothesis).

Covers: hypothesis validation rules (required fields, allowed enums, the
has_parent requirement), the deterministic hypothesis_outcome grading, and
the end-to-end path through gotrain.tools.training.launch_training using
fake_train.py so no GPU/dataset is required.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import ledger, logging_utils
from gotrain.config_schema import HypothesisValidationError, PreflightValidationError, validate_hypothesis
from gotrain.tools import final_eval, training


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


# ---------------------------------------------------------------------------
# config_schema.validate_hypothesis
# ---------------------------------------------------------------------------


def test_validate_hypothesis_none_allowed_without_parent():
    validate_hypothesis(None, has_parent=False)  # must not raise


def test_validate_hypothesis_none_rejected_with_parent():
    with pytest.raises(HypothesisValidationError):
        validate_hypothesis(None, has_parent=True)


def test_validate_hypothesis_is_a_preflight_error():
    assert issubclass(HypothesisValidationError, PreflightValidationError)


def test_validate_hypothesis_requires_change_and_why():
    with pytest.raises(HypothesisValidationError):
        validate_hypothesis({"why": "some reason"}, has_parent=True)
    with pytest.raises(HypothesisValidationError):
        validate_hypothesis({"change": "switched optimizer"}, has_parent=True)
    with pytest.raises(HypothesisValidationError):
        validate_hypothesis({"change": "", "why": "  "}, has_parent=True)


def test_validate_hypothesis_accepts_minimal_valid_hypothesis():
    validate_hypothesis({"change": "switched to adamw", "why": "better generalization"}, has_parent=True)


def test_validate_hypothesis_rejects_bad_expected_direction():
    with pytest.raises(HypothesisValidationError):
        validate_hypothesis(
            {"change": "x", "why": "y", "expected_direction": "sideways"}, has_parent=True
        )


def test_validate_hypothesis_rejects_bad_expected_magnitude():
    with pytest.raises(HypothesisValidationError):
        validate_hypothesis(
            {"change": "x", "why": "y", "expected_magnitude": "huge"}, has_parent=True
        )


def test_validate_hypothesis_rejects_non_dict():
    with pytest.raises(HypothesisValidationError):
        validate_hypothesis("not a dict", has_parent=True)


# ---------------------------------------------------------------------------
# ledger._evaluate_hypothesis, via add_run's hypothesis_outcome
# ---------------------------------------------------------------------------


def test_hypothesis_outcome_increase_supported(isolated_logs):
    r1 = ledger.add_run("m-eurosat", {"lr": 0.01}, val_metric=0.70)
    r2 = ledger.add_run(
        "m-eurosat", {"lr": 0.001}, val_metric=0.80, parent_id=r1["run_id"],
        hypothesis={"change": "lowered lr", "why": "less overshoot", "expected_direction": "increase"},
    )
    assert r2["hypothesis_outcome"] == "supported"


def test_hypothesis_outcome_increase_not_supported(isolated_logs):
    r1 = ledger.add_run("m-eurosat", {"lr": 0.01}, val_metric=0.80)
    r2 = ledger.add_run(
        "m-eurosat", {"lr": 0.001}, val_metric=0.70, parent_id=r1["run_id"],
        hypothesis={"change": "lowered lr", "why": "less overshoot", "expected_direction": "increase"},
    )
    assert r2["hypothesis_outcome"] == "not_supported"


def test_hypothesis_outcome_inconclusive_on_tiny_delta(isolated_logs):
    r1 = ledger.add_run("m-eurosat", {"lr": 0.01}, val_metric=0.8000)
    r2 = ledger.add_run(
        "m-eurosat", {"lr": 0.001}, val_metric=0.8002, parent_id=r1["run_id"],
        hypothesis={"change": "lowered lr", "why": "less overshoot", "expected_direction": "increase"},
    )
    assert r2["hypothesis_outcome"] == "inconclusive"


def test_hypothesis_outcome_no_change_supported_on_small_delta(isolated_logs):
    r1 = ledger.add_run("m-eurosat", {"seed": 0}, val_metric=0.8000)
    r2 = ledger.add_run(
        "m-eurosat", {"seed": 1}, val_metric=0.8002, parent_id=r1["run_id"],
        hypothesis={"change": "changed seed only", "why": "sanity check", "expected_direction": "no_change"},
    )
    assert r2["hypothesis_outcome"] == "supported"


def test_hypothesis_outcome_no_change_not_supported_on_large_delta(isolated_logs):
    r1 = ledger.add_run("m-eurosat", {"seed": 0}, val_metric=0.80)
    r2 = ledger.add_run(
        "m-eurosat", {"seed": 1}, val_metric=0.60, parent_id=r1["run_id"],
        hypothesis={"change": "changed seed only", "why": "sanity check", "expected_direction": "no_change"},
    )
    assert r2["hypothesis_outcome"] == "not_supported"


def test_hypothesis_outcome_none_without_expected_direction(isolated_logs):
    r1 = ledger.add_run("m-eurosat", {"lr": 0.01}, val_metric=0.70)
    r2 = ledger.add_run(
        "m-eurosat", {"lr": 0.001}, val_metric=0.80, parent_id=r1["run_id"],
        hypothesis={"change": "lowered lr", "why": "less overshoot"},
    )
    assert r2["hypothesis_outcome"] is None


def test_hypothesis_outcome_none_without_hypothesis(isolated_logs):
    r1 = ledger.add_run("m-eurosat", {"lr": 0.01}, val_metric=0.70)
    r2 = ledger.add_run("m-eurosat", {"lr": 0.001}, val_metric=0.80, parent_id=r1["run_id"])
    assert r2["hypothesis_outcome"] is None
    assert r2["hypothesis"] is None


# ---------------------------------------------------------------------------
# End-to-end through gotrain.tools.training.launch_training (fake template)
# ---------------------------------------------------------------------------


def test_launch_training_requires_hypothesis_when_parent_given(isolated_logs):
    first = training.launch_training("m-eurosat", "fake", {"lr": 0.01})
    with pytest.raises(HypothesisValidationError):
        training.launch_training("m-eurosat", "fake", {"lr": 0.001}, parent_id=first["run_id"])


def test_launch_training_rejects_malformed_hypothesis_before_subprocess_runs(isolated_logs):
    first = training.launch_training("m-eurosat", "fake", {"lr": 0.01})
    with pytest.raises(HypothesisValidationError):
        training.launch_training(
            "m-eurosat", "fake", {"lr": 0.001}, parent_id=first["run_id"],
            hypothesis={"change": "", "why": ""},
        )
    # the rejected call must not have produced a second ledger record
    assert len(training.query_history("m-eurosat")) == 1


def test_launch_training_stores_valid_hypothesis_and_grades_it(isolated_logs):
    first = training.launch_training("m-eurosat", "fake", {"lr": 0.01})
    second = training.launch_training(
        "m-eurosat", "fake", {"lr": 0.001}, parent_id=first["run_id"],
        hypothesis={
            "change": "lowered lr from 0.01 to 0.001",
            "why": "0.01 may be overshooting the minimum",
            "expected_direction": "increase",
        },
    )
    assert second["hypothesis"]["change"] == "lowered lr from 0.01 to 0.001"
    assert second["hypothesis_outcome"] in ("supported", "not_supported", "inconclusive")


def test_launch_training_first_baseline_run_needs_no_hypothesis(isolated_logs):
    record = training.launch_training("m-eurosat", "fake", {"lr": 0.01})
    assert record["hypothesis"] is None
    assert record["hypothesis_outcome"] is None
