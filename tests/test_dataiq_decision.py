"""Tests for gotrain/dataiq/decision.py -- the data-quality decision
system, and its sharing of gotrain.approval's review queue/audit trail.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import approval, logging_utils
from gotrain.dataiq import decision


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setattr(logging_utils, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", logs_dir / "events.jsonl")
    monkeypatch.setattr(logging_utils, "STATUS_PATH", logs_dir / "status.json")
    monkeypatch.setattr(approval, "APPROVALS_STATE_PATH", logs_dir / "approvals_state.json")
    monkeypatch.setattr(approval, "APPROVALS_LOG_PATH", logs_dir / "approvals.jsonl")
    return logs_dir


def test_quarantine_is_auto_approved(isolated_logs):
    record = decision.propose_action(
        "m-eurosat", sample_id=1, action="QUARANTINE_INVALID_SAMPLE",
        evidence={"problems": ["NaN pixel"]}, severity="high", reason="corrupted",
    )
    assert record["status"] == approval.APPROVED
    assert record["decided_by"] == "policy"


def test_exclude_candidate_is_pending_not_auto_approved(isolated_logs):
    record = decision.propose_action(
        "m-eurosat", sample_id=2, action="EXCLUDE_FROM_TRAINING_CANDIDATE",
        evidence={"loss": 4.2}, severity="medium", reason="high loss, high-confidence wrong prediction",
    )
    assert record["status"] == approval.PENDING


def test_correct_label_candidate_is_pending_not_auto_approved(isolated_logs):
    record = decision.propose_action(
        "m-eurosat", sample_id=3, action="CORRECT_LABEL_CANDIDATE",
        evidence={"predicted_class": 5, "assigned_label": 1}, severity="medium", reason="model disagrees",
    )
    assert record["status"] == approval.PENDING


def test_keep_and_flag_for_review_are_informational_only(isolated_logs):
    keep = decision.propose_action("m-eurosat", 4, "KEEP", evidence={}, severity="none", reason="looks fine")
    flag = decision.propose_action("m-eurosat", 5, "FLAG_FOR_REVIEW", evidence={}, severity="low", reason="worth a look")
    assert keep["status"] == approval.APPROVED
    assert flag["status"] == approval.APPROVED
    assert keep["decided_by"] == "system"


def test_rejects_unknown_action(isolated_logs):
    with pytest.raises(ValueError):
        decision.propose_action("m-eurosat", 1, "DELETE_IT", evidence={}, severity="high", reason="x")


def test_reproposing_does_not_reset_an_existing_decision(isolated_logs):
    first = decision.propose_action(
        "m-eurosat", sample_id=6, action="EXCLUDE_FROM_TRAINING_CANDIDATE",
        evidence={}, severity="medium", reason="first pass",
    )
    approval.decide(first["key"], approve=False, decided_by="engineer", decision_reason="legitimate hard example")

    second = decision.propose_action(
        "m-eurosat", sample_id=6, action="EXCLUDE_FROM_TRAINING_CANDIDATE",
        evidence={}, severity="medium", reason="resubmitted with different wording",
    )
    assert second["status"] == approval.REJECTED
    assert second["decision_reason"] == "legitimate hard example"


def test_get_approved_action_none_when_nothing_decided(isolated_logs):
    decision.propose_action(
        "m-eurosat", sample_id=7, action="EXCLUDE_FROM_TRAINING_CANDIDATE", evidence={}, severity="medium", reason="x",
    )
    assert decision.get_approved_action("m-eurosat", 7) is None


def test_get_approved_action_returns_approved_exclude(isolated_logs):
    proposal = decision.propose_action(
        "m-eurosat", sample_id=8, action="EXCLUDE_FROM_TRAINING_CANDIDATE", evidence={}, severity="medium", reason="x",
    )
    approval.decide(proposal["key"], approve=True)
    assert decision.get_approved_action("m-eurosat", 8) == "EXCLUDE_FROM_TRAINING_CANDIDATE"


def test_get_approved_action_returns_auto_approved_quarantine(isolated_logs):
    decision.propose_action(
        "m-eurosat", sample_id=9, action="QUARANTINE_INVALID_SAMPLE", evidence={}, severity="high", reason="x",
    )
    assert decision.get_approved_action("m-eurosat", 9) == "QUARANTINE_INVALID_SAMPLE"


def test_list_data_proposals_filters_by_dataset_and_status(isolated_logs):
    decision.propose_action("m-eurosat", 1, "EXCLUDE_FROM_TRAINING_CANDIDATE", {}, "medium", "x")
    decision.propose_action("m-so2sat", 2, "EXCLUDE_FROM_TRAINING_CANDIDATE", {}, "medium", "x")
    decision.propose_action("m-eurosat", 3, "QUARANTINE_INVALID_SAMPLE", {}, "high", "x")

    eurosat_only = decision.list_data_proposals(dataset="m-eurosat")
    assert len(eurosat_only) == 2

    pending_only = decision.list_data_proposals(status=approval.PENDING)
    assert len(pending_only) == 2  # the two EXCLUDE_FROM_TRAINING_CANDIDATE proposals


def test_data_proposals_do_not_leak_into_config_gate_listing(isolated_logs):
    """gotrain.approval.list_proposals(kind="config_gate") must never
    show a DataIQ sample-level proposal, and vice versa -- the dashboard
    relies on this separation to render two distinct, non-confusing
    panels.
    """
    decision.propose_action("m-eurosat", 1, "EXCLUDE_FROM_TRAINING_CANDIDATE", {}, "medium", "x")
    with pytest.raises(approval.ApprovalRequiredError):
        approval.check_gate("m-eurosat", "classification", {"backbone": "resnet50"})

    config_gates = approval.list_proposals(kind="config_gate")
    data_actions = approval.list_proposals(kind="data_action")
    assert len(config_gates) == 1
    assert len(data_actions) == 1
    assert config_gates[0]["field"] != "sample_action"
    assert data_actions[0]["action"] == "EXCLUDE_FROM_TRAINING_CANDIDATE"
