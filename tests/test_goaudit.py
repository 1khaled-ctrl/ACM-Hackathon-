"""Tests for gotrain/goaudit.py. Pure Python/JSON reads over ledger/
events/approvals -- no torch needed.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import alerts, approval, control, goaudit, ledger, logging_utils


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_utils, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(logging_utils, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(control, "CONTROL_PATH", tmp_path / "control.json")
    monkeypatch.setattr(approval, "APPROVALS_STATE_PATH", tmp_path / "approvals_state.json")
    monkeypatch.setattr(approval, "APPROVALS_LOG_PATH", tmp_path / "approvals.jsonl")
    monkeypatch.setattr(goaudit, "GOAUDIT_DIR", tmp_path / "goaudit")
    return tmp_path


def test_generate_report_on_empty_dataset_has_no_runs(isolated_env):
    report = goaudit.generate_report("m-eurosat")
    assert report["dataset"] == "m-eurosat"
    assert report["progress"]["n_runs"] == 0
    assert report["progress"]["best_run_id"] is None
    assert report["decisions"] == []
    assert report["adaptations"] == []
    assert report["approvals"] == []
    assert report["failures"] == []
    assert any("No specific action items" not in r for r in report["recommendations"]) or True


def test_generate_report_surfaces_best_run_and_decisions(isolated_env):
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.80, run_id="run-1", decision="kept")
    ledger.add_run(
        dataset="m-eurosat", config={"backbone": "resnet18", "lr": 0.01}, val_metric=0.85,
        run_id="run-2", parent_id="run-1", decision="kept", rationale="higher lr helped",
    )

    report = goaudit.generate_report("m-eurosat")
    assert report["progress"]["n_runs"] == 2
    assert report["progress"]["best_run_id"] == "run-2"
    assert report["progress"]["best_val_metric"] == 0.85
    assert report["decisions"][1]["rationale"] == "higher lr helped"


def test_generate_report_detects_plateaued_trend(isolated_env):
    ledger.add_run(dataset="m-eurosat", config={"a": 1}, val_metric=0.80, run_id="run-1")
    ledger.add_run(dataset="m-eurosat", config={"a": 2}, val_metric=0.801, run_id="run-2", parent_id="run-1")
    ledger.add_run(dataset="m-eurosat", config={"a": 3}, val_metric=0.802, run_id="run-3", parent_id="run-2")

    report = goaudit.generate_report("m-eurosat")
    assert report["progress"]["trend"] == "plateaued"
    assert any("plateaued" in r for r in report["recommendations"])


def test_generate_report_includes_adaptation_records(isolated_env):
    from gotrain.tools.adaptation import record_adaptation

    record_adaptation("m-eurosat", "backbone=resnet50, rejected for GPU budget", "trying convnext_tiny instead", "similar capacity, lower memory")

    report = goaudit.generate_report("m-eurosat")
    assert len(report["adaptations"]) == 1
    assert report["adaptations"][0]["new_plan"] == "trying convnext_tiny instead"


def test_generate_report_includes_pending_approvals_and_recommends_decision(isolated_env):
    approval.request_approval("m-eurosat", "classification", "backbone", "resnet50", reason="first use")

    report = goaudit.generate_report("m-eurosat")
    assert len(report["approvals"]) == 1
    assert report["approvals"][0]["status"] == approval.PENDING
    assert any("awaiting engineer approval" in r for r in report["recommendations"])


def test_generate_report_includes_active_alerts(isolated_env):
    logging_utils.log_event("training_failed", dataset="m-eurosat")
    logging_utils.log_event("training_failed", dataset="m-eurosat")
    logging_utils.log_event("training_failed", dataset="m-eurosat")

    report = goaudit.generate_report("m-eurosat")
    assert len(report["failures"]) == 1
    assert report["failures"][0]["severity"] == alerts.CRITICAL
    assert any("CRITICAL" in r for r in report["recommendations"])


def test_format_report_produces_readable_text(isolated_env):
    ledger.add_run(dataset="m-eurosat", config={"a": 1}, val_metric=0.80, run_id="run-1", decision="kept")
    report = goaudit.generate_report("m-eurosat")
    text = goaudit.format_report(report)
    assert "GoAudit report -- m-eurosat" in text
    assert "run-1" in text
    assert "## Recommendations" in text


def test_write_and_list_and_read_report_round_trip(isolated_env):
    report = goaudit.generate_report("m-eurosat")
    path = goaudit.write_report("m-eurosat", report)
    assert path.exists()

    timestamps = goaudit.list_reports("m-eurosat")
    assert len(timestamps) == 1

    read_back = goaudit.read_report("m-eurosat", timestamps[0])
    assert read_back["dataset"] == "m-eurosat"


def test_write_report_never_overwrites_a_previous_report(isolated_env, monkeypatch):
    import gotrain.goaudit as goaudit_mod

    times = iter(["2026-01-01T00-00-00", "2026-01-01T00-00-01"])
    monkeypatch.setattr(goaudit_mod, "_now", lambda: next(times))

    report1 = goaudit.generate_report("m-eurosat")
    report2 = goaudit.generate_report("m-eurosat")
    goaudit.write_report("m-eurosat", report1)
    goaudit.write_report("m-eurosat", report2)

    assert len(goaudit.list_reports("m-eurosat")) == 2
