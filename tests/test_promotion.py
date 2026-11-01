"""Tests for gotrain/promotion.py. Pure Python/ledger/events I/O with a
stubbed inference check (run_inference_check=False equivalent is not
directly reachable through promote_submitted_model, so these tests patch
rollback.verify_inference_sanity instead of needing torch).
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import control, ledger, logging_utils, promotion, registry, rollback


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_utils, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(logging_utils, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(control, "CONTROL_PATH", tmp_path / "control.json")
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    return tmp_path


def _make_registered_version(tmp_path, dataset, version="v1"):
    version_dir = tmp_path / "models" / dataset / version
    version_dir.mkdir(parents=True)
    (version_dir / "config.json").write_text("{}", encoding="utf-8")
    (version_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (version_dir / "model.pt").write_bytes(b"fake-checkpoint-bytes")


def test_promote_raises_when_nothing_submitted(isolated_env):
    with pytest.raises(promotion.NoSubmittedModelError):
        promotion.promote_submitted_model("m-eurosat", "classification")


def test_promote_raises_when_submitted_run_has_no_version(isolated_env):
    ledger.add_run(dataset="m-eurosat", config={}, val_metric=0.9, run_id="run-1", task_type="classification")
    ledger.update_run("run-1", submitted_final=True)

    with pytest.raises(promotion.ModelNotTestEvaluatedError):
        promotion.promote_submitted_model("m-eurosat", "classification")


def test_promote_raises_when_not_test_evaluated_yet(isolated_env):
    _make_registered_version(isolated_env, "m-eurosat")
    ledger.add_run(
        dataset="m-eurosat", config={}, val_metric=0.9, run_id="run-1",
        task_type="classification", version_id="v1",
    )
    ledger.update_run("run-1", submitted_final=True)

    with pytest.raises(promotion.ModelNotTestEvaluatedError):
        promotion.promote_submitted_model("m-eurosat", "classification")


def test_promote_succeeds_after_test_eval_and_sets_active_version(isolated_env, monkeypatch):
    _make_registered_version(isolated_env, "m-eurosat")
    ledger.add_run(
        dataset="m-eurosat", config={}, val_metric=0.9, run_id="run-1",
        task_type="classification", version_id="v1",
    )
    ledger.update_run("run-1", submitted_final=True)
    logging_utils.log_evaluation(task="m-eurosat", split="test", metrics={"accuracy": 0.91}, model="run-1")

    # Stub the torch-dependent inference check -- its own behavior is
    # covered by test_rollback.py; here we're only testing promotion's
    # own gating logic (submitted + test-evaluated) wraps rollback correctly.
    monkeypatch.setattr(rollback, "verify_inference_sanity", lambda dataset, version, task_type: {"passed": True, "problems": []})

    result = promotion.promote_submitted_model("m-eurosat", "classification")
    assert result["version"] == "v1"
    assert rollback.get_active_version("m-eurosat") == "v1"

    promoted_events = logging_utils.read_events(event="model_promoted", dataset="m-eurosat")
    assert len(promoted_events) == 1
    assert promoted_events[0]["run_id"] == "run-1"
    assert promoted_events[0]["test_metrics"] == {"accuracy": 0.91}


def test_promote_does_not_activate_when_inference_check_fails(isolated_env, monkeypatch):
    _make_registered_version(isolated_env, "m-eurosat")
    ledger.add_run(
        dataset="m-eurosat", config={}, val_metric=0.9, run_id="run-1",
        task_type="classification", version_id="v1",
    )
    ledger.update_run("run-1", submitted_final=True)
    logging_utils.log_evaluation(task="m-eurosat", split="test", metrics={"accuracy": 0.91}, model="run-1")

    monkeypatch.setattr(
        rollback, "verify_inference_sanity",
        lambda dataset, version, task_type: {"passed": False, "problems": ["shape mismatch"]},
    )

    with pytest.raises(rollback.RollbackVerificationError):
        promotion.promote_submitted_model("m-eurosat", "classification")
    assert rollback.get_active_version("m-eurosat") is None
