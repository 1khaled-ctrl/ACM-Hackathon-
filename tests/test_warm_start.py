"""Tests for warm_start_from_parent -- the recursive self-improvement
primitive in gotrain/tools/training.py. All of launch_training's
pre-flight checks (including this one) run before any subprocess is
spawned, so task_type="fake" is used throughout -- no torch needed.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import approval, control, ledger, logging_utils, registry
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
    monkeypatch.setattr(control, "CONTROL_PATH", logs_dir / "control.json")
    monkeypatch.setattr(approval, "APPROVALS_STATE_PATH", logs_dir / "approvals_state.json")
    monkeypatch.setattr(approval, "APPROVALS_LOG_PATH", logs_dir / "approvals.jsonl")
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    return logs_dir


def _make_registered_version(tmp_path, dataset, version="v1"):
    version_dir = tmp_path / "models" / dataset / version
    version_dir.mkdir(parents=True)
    (version_dir / "model.pt").write_bytes(b"fake-checkpoint-bytes")


# ---------------------------------------------------------------------------
# _resolve_warm_start_checkpoint, in isolation
# ---------------------------------------------------------------------------


def test_resolve_raises_without_parent_id(isolated_logs):
    with pytest.raises(training.WarmStartError, match="parent_id"):
        training._resolve_warm_start_checkpoint("m-eurosat", None)


def test_resolve_raises_when_parent_not_in_ledger(isolated_logs):
    with pytest.raises(training.WarmStartError, match="not found"):
        training._resolve_warm_start_checkpoint("m-eurosat", "nonexistent-run")


def test_resolve_raises_on_cross_dataset_parent(isolated_logs):
    ledger.add_run(dataset="m-so2sat", config={}, val_metric=0.5, run_id="parent-1", version_id="v1")
    with pytest.raises(training.WarmStartError, match="m-so2sat"):
        training._resolve_warm_start_checkpoint("m-eurosat", "parent-1")


def test_resolve_raises_when_parent_has_no_version(isolated_logs):
    ledger.add_run(dataset="m-eurosat", config={}, val_metric=0.5, run_id="parent-1")
    with pytest.raises(training.WarmStartError, match="no registered checkpoint"):
        training._resolve_warm_start_checkpoint("m-eurosat", "parent-1")


def test_resolve_raises_when_checkpoint_file_missing(isolated_logs):
    ledger.add_run(dataset="m-eurosat", config={}, val_metric=0.5, run_id="parent-1", version_id="v1")
    with pytest.raises(training.WarmStartError, match="not found"):
        training._resolve_warm_start_checkpoint("m-eurosat", "parent-1")


def test_resolve_succeeds_and_returns_injectable_config_keys(isolated_logs):
    _make_registered_version(isolated_logs.parent, "m-eurosat")
    ledger.add_run(dataset="m-eurosat", config={}, val_metric=0.5, run_id="parent-1", version_id="v1")

    result = training._resolve_warm_start_checkpoint("m-eurosat", "parent-1")
    assert result["_warm_start_parent_run_id"] == "parent-1"
    assert result["_warm_start_parent_version"] == "v1"
    assert result["_warm_start_checkpoint_path"].endswith("model.pt")


# ---------------------------------------------------------------------------
# launch_training integration -- rejected before any subprocess spawns
# ---------------------------------------------------------------------------


def test_launch_training_rejects_warm_start_without_parent_id(isolated_logs):
    with pytest.raises(training.WarmStartError):
        training.launch_training("m-eurosat", "fake", {"warm_start_from_parent": True})


def test_launch_training_ignores_flag_when_false(isolated_logs):
    # Must not even attempt resolution (and therefore not raise) when the
    # flag is absent/false -- existing behavior for every run that doesn't
    # ask for this.
    result = training.launch_training("m-eurosat", "fake", {"warm_start_from_parent": False})
    assert result["val_metric"] is not None
