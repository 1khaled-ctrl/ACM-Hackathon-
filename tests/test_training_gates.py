"""Tests for gotrain/tools/training.py::launch_training's integration with
the control plane (gotrain.control) and the approval gate
(gotrain.approval) -- both checked before any subprocess is spawned, so
these are safe to exercise even for task_type="classification"/
"segmentation" (which would otherwise need torch+downloaded GEO-Bench data
to actually train): a rejection here never reaches the subprocess-spawn
code at all.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import approval, control, ledger, logging_utils
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
    return logs_dir


# ---------------------------------------------------------------------------
# Control-plane blocking (applies regardless of task_type)
# ---------------------------------------------------------------------------


def test_launch_training_refused_while_paused(isolated_logs):
    control.transition(control.PAUSE_REQUESTED)
    control.transition(control.PAUSED_FOR_ENGINEER)

    with pytest.raises(control.ControlBlockedError):
        training.launch_training("m-eurosat", "fake", {"lr": 0.01})

    # a blocked call must not have been recorded in the ledger at all
    assert training.query_history("m-eurosat") == []


def test_launch_training_refused_during_manual_control(isolated_logs):
    control.transition(control.MANUAL_CONTROL)
    with pytest.raises(control.ControlBlockedError):
        training.launch_training("m-eurosat", "fake", {"lr": 0.01})


def test_launch_training_succeeds_once_resumed(isolated_logs):
    control.transition(control.PAUSE_REQUESTED)
    control.transition(control.PAUSED_FOR_ENGINEER)
    with pytest.raises(control.ControlBlockedError):
        training.launch_training("m-eurosat", "fake", {"lr": 0.01})

    control.transition(control.RESUME_REQUESTED)
    control.transition(control.RUNNING_AUTONOMOUSLY)

    record = training.launch_training("m-eurosat", "fake", {"lr": 0.01})
    assert record["val_metric"] is not None


# ---------------------------------------------------------------------------
# Approval gate (classification/segmentation only; rejections happen
# before any subprocess is spawned, so these are safe without torch/data)
# ---------------------------------------------------------------------------


def test_launch_training_rejects_ungated_backbone_change(isolated_logs):
    with pytest.raises(approval.ApprovalRequiredError):
        training.launch_training("m-eurosat", "classification", {"backbone": "convnext_tiny"})
    assert training.query_history("m-eurosat") == []
    assert len(approval.list_proposals(status=approval.PENDING)) == 1


def test_launch_training_rejects_resubmission_after_engineer_rejection(isolated_logs):
    """The brief's worked example end to end through launch_training
    itself: propose resnet50, engineer rejects for GPU limits, agent
    cannot bypass it, falls back to the default and succeeds.
    """
    with pytest.raises(approval.ApprovalRequiredError):
        training.launch_training("m-eurosat", "classification", {"backbone": "resnet50"})

    key = approval.list_proposals()[0]["key"]
    approval.decide(key, approve=False, decided_by="engineer", decision_reason="GPU budget limits")

    with pytest.raises(approval.ApprovalRequiredError) as exc_info:
        training.launch_training("m-eurosat", "classification", {"backbone": "resnet50", "lr": 0.0005})
    assert "gpu budget limits" in str(exc_info.value).lower()

    # No ledger record was ever created for the rejected backbone.
    assert training.query_history("m-eurosat") == []

    # Falling back to the default (resnet18) and an ungated change (fake
    # task_type stands in for "training actually runs") must succeed.
    record = training.launch_training("m-eurosat", "fake", {"lr": 0.0005})
    assert record["val_metric"] is not None


def test_launch_training_default_backbone_never_gated(isolated_logs):
    # Classification would try to spawn a real subprocess past this point
    # if it weren't gated -- but the default is pre-approved, so this
    # specific call must get past approval.check_gate. We don't assert
    # success (that needs torch/data); we assert it did NOT raise
    # ApprovalRequiredError, proving the gate itself let it through.
    try:
        # Bounded: where torch + m-eurosat exist this really trains, so keep
        # it tiny and offline (no pretrained-weight download).
        training.launch_training("m-eurosat", "classification", {
            "backbone": "resnet18", "pretrained": False, "epochs": 1, "batch_size": 4,
            "max_train_samples": 8, "max_val_samples": 8, "amp": False,
        })
    except approval.ApprovalRequiredError:
        pytest.fail("the default backbone value must never be gated")
    except Exception:
        pass  # any failure past this point is expected (no torch/data here)
