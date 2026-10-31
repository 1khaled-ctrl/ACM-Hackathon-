"""Tests for verified rollback (gotrain/rollback.py).

verify_checkpoint_integrity and the orchestration logic in
rollback_to_version/execute_rollback are pure file-based checks and are
exercised directly. verify_inference_sanity itself needs torch -- not
installed in this environment -- so rollback_to_version's tests inject a
stub via its `run_inference_check` parameter instead of calling the real
torch-dependent function; the real one still needs a run on the SageMaker
lab to confirm.
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import control, logging_utils, registry, rollback, status


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    models_dir = tmp_path / "models"

    monkeypatch.setattr(logging_utils, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", logs_dir / "events.jsonl")
    monkeypatch.setattr(logging_utils, "STATUS_PATH", logs_dir / "status.json")
    monkeypatch.setattr(control, "CONTROL_PATH", logs_dir / "control.json")
    monkeypatch.setattr(registry, "MODELS_DIR", models_dir)
    return {"logs_dir": logs_dir, "models_dir": models_dir}


def _make_version(models_dir: Path, dataset: str, version: str, n_bands=3, n_classes=10, checkpoint_bytes=b"fake") -> Path:
    version_dir = models_dir / dataset / version
    version_dir.mkdir(parents=True)
    (version_dir / "config.json").write_text(json.dumps({"backbone": "resnet18"}), encoding="utf-8")
    (version_dir / "metrics.json").write_text(json.dumps({"n_bands": n_bands, "n_classes": n_classes}), encoding="utf-8")
    if checkpoint_bytes is not None:
        (version_dir / "model.pt").write_bytes(checkpoint_bytes)
    return version_dir


# ---------------------------------------------------------------------------
# verify_checkpoint_integrity
# ---------------------------------------------------------------------------


def test_integrity_passes_on_complete_version(isolated_env):
    _make_version(isolated_env["models_dir"], "m-eurosat", "v1")
    result = rollback.verify_checkpoint_integrity("m-eurosat", "v1")
    assert result["passed"] is True
    assert result["problems"] == []


def test_integrity_fails_on_missing_version_dir(isolated_env):
    result = rollback.verify_checkpoint_integrity("m-eurosat", "v99")
    assert result["passed"] is False


def test_integrity_fails_on_missing_checkpoint(isolated_env):
    _make_version(isolated_env["models_dir"], "m-eurosat", "v1", checkpoint_bytes=None)
    result = rollback.verify_checkpoint_integrity("m-eurosat", "v1")
    assert result["passed"] is False
    assert any("model.pt missing" in p for p in result["problems"])


def test_integrity_fails_on_empty_checkpoint(isolated_env):
    _make_version(isolated_env["models_dir"], "m-eurosat", "v1", checkpoint_bytes=b"")
    result = rollback.verify_checkpoint_integrity("m-eurosat", "v1")
    assert result["passed"] is False
    assert any("empty" in p for p in result["problems"])


# ---------------------------------------------------------------------------
# rollback_to_version (orchestration; inference check stubbed)
# ---------------------------------------------------------------------------


def test_rollback_to_version_succeeds_and_sets_active_pointer(isolated_env):
    _make_version(isolated_env["models_dir"], "m-eurosat", "v1")
    stub_inference = lambda version, task_type: {"passed": True, "problems": []}

    assert rollback.get_active_version("m-eurosat") is None
    result = rollback.rollback_to_version("m-eurosat", "v1", "classification", run_inference_check=stub_inference)

    assert result["version"] == "v1"
    assert rollback.get_active_version("m-eurosat") == "v1"


def test_rollback_to_version_fails_integrity_never_touches_active_pointer(isolated_env):
    _make_version(isolated_env["models_dir"], "m-eurosat", "v1")
    _make_version(isolated_env["models_dir"], "m-eurosat", "v2", checkpoint_bytes=None)  # broken
    stub_inference = lambda version, task_type: {"passed": True, "problems": []}

    rollback.rollback_to_version("m-eurosat", "v1", "classification", run_inference_check=stub_inference)
    assert rollback.get_active_version("m-eurosat") == "v1"

    with pytest.raises(rollback.RollbackVerificationError):
        rollback.rollback_to_version("m-eurosat", "v2", "classification", run_inference_check=stub_inference)

    # v1 must still be active -- v2's failure never touched the pointer.
    assert rollback.get_active_version("m-eurosat") == "v1"


def test_rollback_to_version_fails_inference_never_touches_active_pointer(isolated_env):
    _make_version(isolated_env["models_dir"], "m-eurosat", "v1")
    failing_inference = lambda version, task_type: {"passed": False, "problems": ["shape mismatch"]}

    with pytest.raises(rollback.RollbackVerificationError):
        rollback.rollback_to_version("m-eurosat", "v1", "classification", run_inference_check=failing_inference)

    assert rollback.get_active_version("m-eurosat") is None


def test_rollback_to_version_can_skip_inference_check(isolated_env):
    _make_version(isolated_env["models_dir"], "m-eurosat", "v1")
    result = rollback.rollback_to_version("m-eurosat", "v1", "classification", run_inference_check=False)
    assert result["version"] == "v1"
    assert result["verification"]["inference"]["skipped"] is True


# ---------------------------------------------------------------------------
# execute_rollback: control-plane transitions + conflict detection
# ---------------------------------------------------------------------------


def test_execute_rollback_refuses_while_training_in_progress(isolated_env):
    _make_version(isolated_env["models_dir"], "m-eurosat", "v1")
    status.set_current_run({"run_id": "abc123", "dataset": "m-eurosat"})

    with pytest.raises(rollback.RollbackConflictError):
        rollback.execute_rollback("m-eurosat", "v1", "classification")

    # Must not have touched control state at all.
    assert control.get_state() == control.RUNNING_AUTONOMOUSLY


def test_execute_rollback_success_leaves_system_paused(isolated_env, monkeypatch):
    _make_version(isolated_env["models_dir"], "m-eurosat", "v1")
    monkeypatch.setattr(
        rollback, "verify_inference_sanity", lambda dataset, version, task_type: {"passed": True, "problems": []}
    )

    result = rollback.execute_rollback("m-eurosat", "v1", "classification")

    assert result["version"] == "v1"
    assert control.get_state() == control.PAUSED_FOR_ENGINEER
    assert rollback.get_active_version("m-eurosat") == "v1"


def test_execute_rollback_failure_leaves_system_failed_safe(isolated_env, monkeypatch):
    _make_version(isolated_env["models_dir"], "m-eurosat", "v1", checkpoint_bytes=None)

    with pytest.raises(rollback.RollbackVerificationError):
        rollback.execute_rollback("m-eurosat", "v1", "classification")

    assert control.get_state() == control.FAILED_SAFE
    assert rollback.get_active_version("m-eurosat") is None


def test_execute_rollback_never_auto_resumes(isolated_env, monkeypatch):
    """A successful rollback must leave the system paused -- an engineer
    has to explicitly resume, it must never silently go back to
    RUNNING_AUTONOMOUSLY on its own.
    """
    _make_version(isolated_env["models_dir"], "m-eurosat", "v1")
    monkeypatch.setattr(
        rollback, "verify_inference_sanity", lambda dataset, version, task_type: {"passed": True, "problems": []}
    )
    rollback.execute_rollback("m-eurosat", "v1", "classification")
    assert control.get_state() != control.RUNNING_AUTONOMOUSLY
    assert control.is_blocking() is True
