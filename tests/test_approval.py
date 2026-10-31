"""Tests for the code-enforced approval gate (gotrain/approval.py).

Pure file-based logic -- no torch, geobench, or strands required.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import approval, logging_utils
from gotrain.config_schema import PreflightValidationError


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


def test_pre_approved_default_never_gated(isolated_logs):
    # resnet18 is the classification default -- must pass with no record created.
    approval.check_gate("m-eurosat", "classification", {"backbone": "resnet18"})
    assert approval.list_proposals() == []


def test_ungated_field_change_never_gated(isolated_logs):
    approval.check_gate("m-eurosat", "classification", {"lr": 0.0003, "optimizer": "adamw"})
    assert approval.list_proposals() == []


def test_first_use_of_gated_value_creates_pending_proposal_and_raises(isolated_logs):
    with pytest.raises(approval.ApprovalRequiredError):
        approval.check_gate("m-eurosat", "classification", {"backbone": "convnext_tiny"})

    pending = approval.list_proposals(status=approval.PENDING)
    assert len(pending) == 1
    assert pending[0]["field"] == "backbone"
    assert pending[0]["value"] == "convnext_tiny"
    assert pending[0]["dataset"] == "m-eurosat"


def test_approval_required_error_is_a_preflight_error(isolated_logs):
    assert issubclass(approval.ApprovalRequiredError, PreflightValidationError)


def test_still_pending_on_second_attempt_does_not_duplicate(isolated_logs):
    with pytest.raises(approval.ApprovalRequiredError):
        approval.check_gate("m-eurosat", "classification", {"backbone": "convnext_tiny"})
    with pytest.raises(approval.ApprovalRequiredError) as exc_info:
        approval.check_gate("m-eurosat", "classification", {"backbone": "convnext_tiny"})

    assert "still pending" in str(exc_info.value).lower()
    assert len(approval.list_proposals()) == 1  # not duplicated


def test_approved_value_passes_the_gate(isolated_logs):
    with pytest.raises(approval.ApprovalRequiredError):
        approval.check_gate("m-eurosat", "classification", {"backbone": "convnext_tiny"})

    key = approval.list_proposals()[0]["key"]
    approval.decide(key, approve=True, decided_by="engineer", decision_reason="looks fine")

    approval.check_gate("m-eurosat", "classification", {"backbone": "convnext_tiny"})  # must not raise
    assert approval.list_proposals(status=approval.APPROVED)[0]["key"] == key


def test_rejected_value_cannot_be_bypassed_by_resubmission(isolated_logs):
    """The worked example from the design brief: GoTrain proposes
    resnet50, the engineer rejects it for GPU limits, and no amount of
    resubmitting it -- under any config/rationale -- lets it through,
    because the gate is keyed by (dataset, field, value), not by call
    identity.
    """
    with pytest.raises(approval.ApprovalRequiredError):
        approval.check_gate("m-eurosat", "classification", {"backbone": "resnet50"})

    key = approval.list_proposals()[0]["key"]
    approval.decide(key, approve=False, decided_by="engineer", decision_reason="GPU budget limits")

    # Resubmitted with different surrounding config each time -- still blocked.
    for extra_config in [{}, {"lr": 0.0001}, {"epochs": 50}, {"augmentation": "strong"}]:
        with pytest.raises(approval.ApprovalRequiredError) as exc_info:
            approval.check_gate("m-eurosat", "classification", {"backbone": "resnet50", **extra_config})
        assert "rejected" in str(exc_info.value).lower()
        assert "gpu budget limits" in str(exc_info.value).lower()

    assert approval.list_proposals(status=approval.REJECTED)[0]["decision_reason"] == "GPU budget limits"


def test_segmentation_gates_architecture_and_encoder_independently(isolated_logs):
    # Changing architecture away from default requires approval...
    with pytest.raises(approval.ApprovalRequiredError):
        approval.check_gate("m-pv4ger-seg", "segmentation", {"architecture": "fpn"})
    # ...but the default encoder alongside it should not ALSO be gated.
    pending = approval.list_proposals(status=approval.PENDING)
    assert len(pending) == 1
    assert pending[0]["field"] == "architecture"

    # Changing encoder away from default is a SEPARATE gate.
    with pytest.raises(approval.ApprovalRequiredError):
        approval.check_gate("m-pv4ger-seg", "segmentation", {"encoder": "efficientnet-b3"})
    assert len(approval.list_proposals(status=approval.PENDING)) == 2


def test_default_architecture_and_encoder_together_not_gated(isolated_logs):
    approval.check_gate("m-pv4ger-seg", "segmentation", {"architecture": "unet", "encoder": "resnet34"})
    assert approval.list_proposals() == []


def test_decide_raises_on_unknown_key(isolated_logs):
    with pytest.raises(ValueError):
        approval.decide("does-not-exist", approve=True)


def test_gate_is_independent_per_dataset(isolated_logs):
    with pytest.raises(approval.ApprovalRequiredError):
        approval.check_gate("m-eurosat", "classification", {"backbone": "resnet50"})
    key = approval.list_proposals()[0]["key"]
    approval.decide(key, approve=False)

    # A different dataset's use of the same value is an independent gate.
    with pytest.raises(approval.ApprovalRequiredError):
        approval.check_gate("m-so2sat", "classification", {"backbone": "resnet50"})
    pending = approval.list_proposals(status=approval.PENDING)
    assert len(pending) == 1
    assert pending[0]["dataset"] == "m-so2sat"
