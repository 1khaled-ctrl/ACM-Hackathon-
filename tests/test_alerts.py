"""Tests for gotrain/alerts.py. Pure Python/JSON reads, no torch needed."""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import alerts, control, logging_utils


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_utils, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(control, "CONTROL_PATH", tmp_path / "control.json")
    return tmp_path


def test_check_alerts_returns_empty_list_when_nothing_is_wrong(isolated_logs):
    assert alerts.check_alerts("m-eurosat") == []


def test_failed_safe_control_state_produces_critical_alert(isolated_logs):
    control.transition(control.PAUSE_REQUESTED)
    control.transition(control.PAUSED_FOR_ENGINEER)
    control.transition(control.ROLLBACK_REQUESTED)
    control.transition(control.ROLLBACK_VERIFYING)
    control.transition(control.FAILED_SAFE, reason="integrity check failed")

    result = alerts.check_alerts("m-eurosat")
    failed_safe = [a for a in result if a["type"] == "failed_safe"]
    assert len(failed_safe) == 1
    assert failed_safe[0]["severity"] == alerts.CRITICAL
    assert "integrity check failed" in failed_safe[0]["message"]


def test_single_training_failure_is_only_a_warning(isolated_logs):
    logging_utils.log_event("training_failed", dataset="m-eurosat")

    result = alerts.check_alerts("m-eurosat")
    failure_alerts = [a for a in result if a["type"] == "training_failures"]
    assert len(failure_alerts) == 1
    assert failure_alerts[0]["severity"] == alerts.WARNING


def test_repeated_training_failures_escalate_to_critical(isolated_logs):
    for _ in range(3):
        logging_utils.log_event("training_failed", dataset="m-eurosat")

    result = alerts.check_alerts("m-eurosat")
    failure_alerts = [a for a in result if a["type"] == "training_failures"]
    assert len(failure_alerts) == 1
    assert failure_alerts[0]["severity"] == alerts.CRITICAL


def test_training_failures_for_other_dataset_are_not_mixed_in(isolated_logs):
    logging_utils.log_event("training_failed", dataset="m-so2sat")
    assert alerts.check_alerts("m-eurosat") == []


def test_severe_miscalibration_produces_warning_alert(isolated_logs):
    logging_utils.log_event(
        "dataiq_loss_scan_finished", dataset="m-eurosat", run_id="run-1",
        calibration={"verdict": "severely overconfident -- confidence should not be trusted at face value", "ece": 0.42},
    )

    result = alerts.check_alerts("m-eurosat")
    calib_alerts = [a for a in result if a["type"] == "severe_miscalibration"]
    assert len(calib_alerts) == 1
    assert calib_alerts[0]["run_id"] == "run-1"
    assert "0.420" in calib_alerts[0]["message"]


def test_moderate_miscalibration_does_not_alert(isolated_logs):
    logging_utils.log_event(
        "dataiq_loss_scan_finished", dataset="m-eurosat", run_id="run-1",
        calibration={"verdict": "moderately overconfident -- confidence should not be trusted at face value", "ece": 0.10},
    )
    assert alerts.check_alerts("m-eurosat") == []


def test_rollback_failure_produces_critical_alert(isolated_logs):
    logging_utils.log_event(
        "rollback_failed", dataset="m-eurosat", version="v2", stage="integrity", problems=["model.pt missing"],
    )

    result = alerts.check_alerts("m-eurosat")
    rollback_alerts = [a for a in result if a["type"] == "rollback_verification_failed"]
    assert len(rollback_alerts) == 1
    assert rollback_alerts[0]["severity"] == alerts.CRITICAL


def test_critical_alerts_sort_before_warnings(isolated_logs):
    logging_utils.log_event("training_failed", dataset="m-eurosat")  # warning
    logging_utils.log_event("rollback_failed", dataset="m-eurosat", version="v2", stage="integrity", problems=[])  # critical

    result = alerts.check_alerts("m-eurosat")
    assert result[0]["severity"] == alerts.CRITICAL
