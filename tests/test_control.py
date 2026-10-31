"""Tests for the engineer control-plane state machine (gotrain/control.py).

Pure file-based logic -- no torch, geobench, or strands required, so this
exercises the actual module, not a stand-in.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import control, logging_utils
from gotrain.config_schema import PreflightValidationError


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setattr(logging_utils, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", logs_dir / "events.jsonl")
    monkeypatch.setattr(logging_utils, "STATUS_PATH", logs_dir / "status.json")
    monkeypatch.setattr(control, "CONTROL_PATH", logs_dir / "control.json")
    return logs_dir


def test_default_state_is_running_autonomously(isolated_logs):
    assert control.get_state() == control.RUNNING_AUTONOMOUSLY
    assert control.is_blocking() is False


def test_valid_transition_pause_then_acknowledge(isolated_logs):
    control.transition(control.PAUSE_REQUESTED, reason="engineer clicked pause")
    assert control.get_state() == control.PAUSE_REQUESTED
    assert control.is_blocking() is True

    control.acknowledge_pause()
    assert control.get_state() == control.PAUSED_FOR_ENGINEER


def test_acknowledge_pause_is_noop_when_not_pause_requested(isolated_logs):
    before = control.get_control()
    control.acknowledge_pause()
    after = control.get_control()
    assert before["state"] == after["state"] == control.RUNNING_AUTONOMOUSLY


def test_illegal_transition_raises(isolated_logs):
    # Cannot jump straight from RUNNING_AUTONOMOUSLY to PAUSED_FOR_ENGINEER
    # (must go through PAUSE_REQUESTED first).
    with pytest.raises(control.IllegalTransitionError):
        control.transition(control.PAUSED_FOR_ENGINEER)


def test_completed_is_terminal(isolated_logs):
    control.transition(control.COMPLETED, reason="done")
    with pytest.raises(control.IllegalTransitionError):
        control.transition(control.PAUSE_REQUESTED)
    with pytest.raises(control.IllegalTransitionError):
        control.transition(control.RUNNING_AUTONOMOUSLY)


def test_manual_control_round_trip(isolated_logs):
    control.transition(control.MANUAL_CONTROL, reason="engineer took over")
    assert control.is_blocking() is True
    control.transition(control.RESUME_REQUESTED, requested_by="engineer")
    control.transition(control.RUNNING_AUTONOMOUSLY, requested_by="agent")
    assert control.get_state() == control.RUNNING_AUTONOMOUSLY
    assert control.is_blocking() is False


def test_failed_safe_requires_paused_before_resume(isolated_logs):
    control.transition(control.FAILED_SAFE, reason="rollback failed")
    with pytest.raises(control.IllegalTransitionError):
        control.transition(control.RESUME_REQUESTED)
    control.transition(control.PAUSED_FOR_ENGINEER)
    control.transition(control.RESUME_REQUESTED)
    control.transition(control.RUNNING_AUTONOMOUSLY)
    assert control.get_state() == control.RUNNING_AUTONOMOUSLY


def test_reset_forces_running_regardless_of_prior_state(isolated_logs):
    control.transition(control.MANUAL_CONTROL)
    control.reset()
    assert control.get_state() == control.RUNNING_AUTONOMOUSLY


def test_raise_if_blocking_raises_preflight_error_when_blocked(isolated_logs):
    control.transition(control.PAUSE_REQUESTED)
    with pytest.raises(control.ControlBlockedError):
        control.raise_if_blocking()
    assert issubclass(control.ControlBlockedError, PreflightValidationError)


def test_raise_if_blocking_passes_when_running(isolated_logs):
    control.raise_if_blocking()  # must not raise


@pytest.mark.parametrize(
    "state,expected_reason",
    [
        (control.PAUSE_REQUESTED, "paused_by_engineer"),
        (control.PAUSED_FOR_ENGINEER, "paused_by_engineer"),
        (control.MANUAL_CONTROL, "manual_control_takeover"),
        (control.ROLLBACK_REQUESTED, "rollback_in_progress"),
        (control.ROLLBACK_VERIFYING, "rollback_in_progress"),
    ],
)
def test_should_stop_training_reflects_state(isolated_logs, state, expected_reason):
    if state == control.PAUSED_FOR_ENGINEER:
        control.transition(control.PAUSE_REQUESTED)
        control.transition(state)
    elif state == control.ROLLBACK_VERIFYING:
        control.transition(control.ROLLBACK_REQUESTED)
        control.transition(state)
    else:
        control.transition(state)
    assert control.should_stop_training() == expected_reason


def test_should_stop_training_none_when_running(isolated_logs):
    assert control.should_stop_training() is None


def test_wait_for_resume_blocks_until_resume_requested_then_transitions(isolated_logs):
    control.transition(control.PAUSE_REQUESTED)
    control.transition(control.PAUSED_FOR_ENGINEER)

    calls = {"n": 0}

    def fake_sleep(_seconds):
        calls["n"] += 1
        if calls["n"] == 3:
            # Simulate the dashboard flipping the state while we "sleep".
            control.transition(control.RESUME_REQUESTED, requested_by="engineer")

    control.wait_for_resume(poll_seconds=0, sleep_fn=fake_sleep)

    assert control.get_state() == control.RUNNING_AUTONOMOUSLY
    assert calls["n"] == 3


def test_wait_for_resume_returns_immediately_if_not_blocking(isolated_logs):
    calls = {"n": 0}

    def fake_sleep(_seconds):
        calls["n"] += 1

    control.wait_for_resume(poll_seconds=0, sleep_fn=fake_sleep)
    assert calls["n"] == 0


def test_every_transition_is_logged(isolated_logs):
    import json

    control.transition(control.PAUSE_REQUESTED, reason="test")
    control.transition(control.PAUSED_FOR_ENGINEER, reason="ack")

    events = [
        json.loads(line)
        for line in (isolated_logs / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    transitions = [e for e in events if e["event"] == "control_transition"]
    assert len(transitions) == 2
    assert transitions[0]["to_state"] == control.PAUSE_REQUESTED
    assert transitions[1]["to_state"] == control.PAUSED_FOR_ENGINEER
