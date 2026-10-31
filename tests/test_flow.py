"""End-to-end tests for the ledger/status/tools flow, using fake_train.py so
no real model training or GPU is required. Each test gets an isolated
logs/ directory (via monkeypatching the module-level paths) so tests don't
clobber real run history.
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import ledger, logging_utils, status
from gotrain.tools import final_eval, training


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    """Redirect every module's LOGS_DIR/*_PATH constants to a temp directory."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    monkeypatch.setattr(logging_utils, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", logs_dir / "events.jsonl")
    monkeypatch.setattr(logging_utils, "STATUS_PATH", logs_dir / "status.json")
    monkeypatch.setattr(ledger, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(ledger, "LEDGER_PATH", logs_dir / "ledger.jsonl")
    monkeypatch.setattr(final_eval, "LOGS_DIR", logs_dir)

    return logs_dir


def test_add_run_and_get_best(isolated_logs):
    r1 = ledger.add_run("m-eurosat", {"lr": 0.01}, val_metric=0.70, decision="kept")
    r2 = ledger.add_run(
        "m-eurosat", {"lr": 0.001}, val_metric=0.80, parent_id=r1["run_id"], decision="kept"
    )

    assert r2["config_diff"] == {"lr": {"from": 0.01, "to": 0.001}}
    assert r2["delta_vs_parent"] == pytest.approx(0.10)

    best = ledger.get_best("m-eurosat")
    assert best["run_id"] == r2["run_id"]


def test_add_run_rejects_non_validation_split(isolated_logs):
    with pytest.raises(ValueError):
        ledger.add_run("m-eurosat", {}, val_metric=0.5, split_used="test")


def test_get_lineage_follows_parent_chain(isolated_logs):
    r1 = ledger.add_run("m-eurosat", {"lr": 0.01}, val_metric=0.70)
    r2 = ledger.add_run("m-eurosat", {"lr": 0.005}, val_metric=0.75, parent_id=r1["run_id"])
    r3 = ledger.add_run("m-eurosat", {"lr": 0.001}, val_metric=0.80, parent_id=r2["run_id"])

    lineage = ledger.get_lineage(r3["run_id"])
    assert [r["run_id"] for r in lineage] == [r1["run_id"], r2["run_id"], r3["run_id"]]


def test_status_merges_and_is_atomic(isolated_logs):
    status.set_stage("exploring")
    status.set_action("training", where="m-eurosat", iteration=1)

    current = status.get_status()
    assert current["stage"] == "exploring"
    assert current["action"] == "training"
    assert current["where"] == "m-eurosat"
    assert current["iteration"] == 1

    # status.json must never be left as a .tmp file after a write
    assert not list(isolated_logs.glob(".status_*.tmp"))
    assert (isolated_logs / "status.json").exists()


def test_launch_training_with_fake_template_records_ledger(isolated_logs):
    record = training.launch_training("m-eurosat", "fake", {"lr": 0.01, "epochs": 1})
    assert record["split_used"] == "validation"
    assert isinstance(record["val_metric"], float)

    history = training.query_history("m-eurosat")
    assert len(history) == 1
    assert history[0]["run_id"] == record["run_id"]


def test_launch_training_rejects_unknown_task_type(isolated_logs):
    with pytest.raises(ValueError):
        training.launch_training("m-eurosat", "not-a-real-type", {})


def test_final_test_eval_runs_once_then_refuses(isolated_logs):
    record = final_eval.final_test_eval("m-eurosat", "run-1", lambda: {"accuracy": 0.9})
    assert record["split"] == "test"
    assert record["metrics"]["accuracy"] == 0.9

    with pytest.raises(final_eval.TestAlreadyEvaluatedError):
        final_eval.final_test_eval("m-eurosat", "run-2", lambda: {"accuracy": 0.1})


def test_final_test_eval_is_scoped_per_dataset(isolated_logs):
    final_eval.final_test_eval("m-eurosat", "run-1", lambda: {"accuracy": 0.9})
    # a different dataset's test split hasn't been touched yet, so this must succeed
    record = final_eval.final_test_eval("m-nz-cattle", "run-2", lambda: {"iou": 0.5})
    assert record["task"] == "m-nz-cattle"
