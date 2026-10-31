"""Tests for gotrain/dataiq/loss_scan.py::_record_and_flag -- the shared
tail end of both scan_classification_run/scan_segmentation_run. Needs no
torch: it only writes files and calls dataiq.decision/log_event, same as
every other DataIQ module. Also covers the exact bug this session found
and fixed: log_event was called with dataset=/run_id=/task_type= kwargs
*and* **summary, which already carried those same keys -- a guaranteed
TypeError on every real call, never caught because scan_classification_run
itself needs torch and was never executed here.
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import approval, logging_utils
from gotrain.dataiq import loss_scan


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setattr(logging_utils, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", logs_dir / "events.jsonl")
    monkeypatch.setattr(logging_utils, "STATUS_PATH", logs_dir / "status.json")
    monkeypatch.setattr(approval, "APPROVALS_STATE_PATH", logs_dir / "approvals_state.json")
    monkeypatch.setattr(approval, "APPROVALS_LOG_PATH", logs_dir / "approvals.jsonl")
    monkeypatch.setattr(loss_scan, "SAMPLE_LOSSES_DIR", logs_dir / "dataiq" / "sample_losses")
    return logs_dir


def test_record_and_flag_does_not_raise_on_duplicate_keys(isolated_logs):
    """The exact bug: summary already contains dataset/run_id/task_type;
    log_event must not be called with those as explicit kwargs too.
    """
    rows = [{"sample_id": 0, "possible_issue": None}]
    summary = loss_scan._record_and_flag("m-eurosat", "run-1", "classification", rows)  # must not raise
    assert summary["dataset"] == "m-eurosat"
    assert summary["run_id"] == "run-1"
    assert summary["n_samples"] == 1
    assert summary["n_flagged"] == 0


def test_record_and_flag_logs_the_event(isolated_logs):
    rows = [{"sample_id": 0, "possible_issue": None}]
    loss_scan._record_and_flag("m-eurosat", "run-1", "classification", rows)

    events = [
        json.loads(line)
        for line in (isolated_logs / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    finished_events = [e for e in events if e["event"] == "dataiq_loss_scan_finished"]
    assert len(finished_events) == 1
    assert finished_events[0]["dataset"] == "m-eurosat"


def test_record_and_flag_merges_extra_summary(isolated_logs):
    rows = [{"sample_id": 0, "possible_issue": None}]
    summary = loss_scan._record_and_flag(
        "m-eurosat", "run-1", "classification", rows,
        extra_summary={"calibration": {"ece": 0.12, "verdict": "moderately overconfident"}, "calibration_split": "train"},
    )
    assert summary["calibration"]["ece"] == 0.12
    assert summary["calibration_split"] == "train"

    # the written report file carries it too, not just the return value
    written = json.loads((isolated_logs / "dataiq" / "sample_losses" / "m-eurosat" / "run-1.json").read_text())
    assert written["summary"]["calibration"]["verdict"] == "moderately overconfident"


def test_record_and_flag_raises_proposals_for_flagged_rows(isolated_logs):
    rows = [
        {"sample_id": 0, "possible_issue": None},
        {"sample_id": 1, "possible_issue": "high_confidence_wrong_prediction", "loss": 4.0, "confidence": 0.95},
        {"sample_id": 2, "possible_issue": "high_loss_hard_or_noisy", "loss": 3.5, "confidence": 0.4},
    ]
    summary = loss_scan._record_and_flag("m-eurosat", "run-1", "classification", rows)
    assert summary["n_flagged"] == 2

    from gotrain.dataiq import decision as dataiq_decision

    proposals = dataiq_decision.list_data_proposals(dataset="m-eurosat")
    assert len(proposals) == 2
    actions = {p["sample_id"]: p["action"] for p in proposals}
    assert actions[1] == "CORRECT_LABEL_CANDIDATE"  # high-confidence wrong -> suspected mislabel
    assert actions[2] == "EXCLUDE_FROM_TRAINING_CANDIDATE"  # high loss, not confidently wrong -> hard/noisy
    # every proposal from a model-assisted signal must be PENDING, never auto-approved
    assert all(p["status"] == approval.PENDING for p in proposals)


def test_record_and_flag_without_extra_summary_omits_calibration_key(isolated_logs):
    rows = [{"sample_id": 0, "possible_issue": None}]
    summary = loss_scan._record_and_flag("m-pv4ger-seg", "run-2", "segmentation", rows)
    assert "calibration" not in summary
