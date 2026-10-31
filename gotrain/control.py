"""Authoritative run-state controller for GoTrain's engineer control plane.

logs/control.json is the single source of truth for "what mode is the
system in right now": RUNNING_AUTONOMOUSLY (the agent drives), or one of
several engineer-initiated states (paused, manual control, rollback in
progress, failed-safe). The dashboard (a separate OS process) and the
agent process (gotrain/agent.py's LimitsAndLoggingHooks, and
gotrain/tools/training.py as a second, defense-in-depth check) both
read/write this file -- same atomic tempfile+os.replace pattern as
status.json/ledger.jsonl, so either side can act on it without a
concurrent write corrupting the file.

This is a STATE MACHINE, not a free-for-all: transition() only allows the
moves listed in VALID_TRANSITIONS and raises IllegalTransitionError
otherwise, so a dashboard bug can't quietly leave the system in a state
nobody intended. Every transition is logged to events.jsonl
(control_transition) for auditability.

No agent tool ever calls transition() with requested_by="agent" except
acknowledging a pause and marking COMPLETED on submit_final_model --
pause/resume/manual-takeover/rollback are only ever engineer-initiated,
from the dashboard. Giving the agent a "resume yourself" tool would defeat
the entire point of this module.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from gotrain.config_schema import PreflightValidationError
from gotrain.logging_utils import LOGS_DIR, log_event

CONTROL_PATH = LOGS_DIR / "control.json"

RUNNING_AUTONOMOUSLY = "RUNNING_AUTONOMOUSLY"
PAUSE_REQUESTED = "PAUSE_REQUESTED"
PAUSED_FOR_ENGINEER = "PAUSED_FOR_ENGINEER"
MANUAL_CONTROL = "MANUAL_CONTROL"
RESUME_REQUESTED = "RESUME_REQUESTED"
ROLLBACK_REQUESTED = "ROLLBACK_REQUESTED"
ROLLBACK_VERIFYING = "ROLLBACK_VERIFYING"
FAILED_SAFE = "FAILED_SAFE"
COMPLETED = "COMPLETED"

ALL_STATES = {
    RUNNING_AUTONOMOUSLY, PAUSE_REQUESTED, PAUSED_FOR_ENGINEER, MANUAL_CONTROL,
    RESUME_REQUESTED, ROLLBACK_REQUESTED, ROLLBACK_VERIFYING, FAILED_SAFE, COMPLETED,
}

# States in which the agent must not be allowed to take any action at all
# (training, decisions, submission) -- checked by gotrain/agent.py's hook
# before every single tool call, and redundantly inside
# gotrain/tools/training.py::launch_training for anything that calls it
# outside the agent/hook pipeline.
BLOCKING_STATES = {
    PAUSE_REQUESTED, PAUSED_FOR_ENGINEER, MANUAL_CONTROL,
    ROLLBACK_REQUESTED, ROLLBACK_VERIFYING, FAILED_SAFE,
    # COMPLETED is terminal: once a final model is submitted, nothing --
    # including the agent itself -- should be able to take any further
    # action for this dataset's session.
    COMPLETED,
}

VALID_TRANSITIONS = {
    RUNNING_AUTONOMOUSLY: {PAUSE_REQUESTED, MANUAL_CONTROL, ROLLBACK_REQUESTED, FAILED_SAFE, COMPLETED},
    # PAUSE_REQUESTED can still fall back to RUNNING_AUTONOMOUSLY if the
    # engineer changes their mind before the agent has acknowledged it.
    PAUSE_REQUESTED: {PAUSED_FOR_ENGINEER, RUNNING_AUTONOMOUSLY},
    PAUSED_FOR_ENGINEER: {RESUME_REQUESTED, MANUAL_CONTROL, ROLLBACK_REQUESTED},
    MANUAL_CONTROL: {RESUME_REQUESTED, ROLLBACK_REQUESTED},
    RESUME_REQUESTED: {RUNNING_AUTONOMOUSLY},
    ROLLBACK_REQUESTED: {ROLLBACK_VERIFYING},
    # Verification success -> PAUSED_FOR_ENGINEER (never auto-resumes after
    # a rollback; an engineer must explicitly resume). Failure -> FAILED_SAFE.
    ROLLBACK_VERIFYING: {PAUSED_FOR_ENGINEER, FAILED_SAFE},
    # Only an engineer acknowledging a failure can move past it, and only
    # back to paused (never straight back to autonomous).
    FAILED_SAFE: {PAUSED_FOR_ENGINEER},
    COMPLETED: set(),  # terminal
}


class IllegalTransitionError(ValueError):
    pass


class ControlBlockedError(PreflightValidationError):
    """Raised by gotrain/tools/training.py when launch_training is called
    while the control state is one of BLOCKING_STATES. A
    PreflightValidationError like ConfigValidationError/
    HypothesisValidationError/ApprovalRequiredError, so it never counts
    against --max-runs.
    """


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_control() -> dict:
    if not CONTROL_PATH.exists():
        return {"state": RUNNING_AUTONOMOUSLY, "previous_state": None, "reason": None, "requested_by": None, "updated_at": None}
    return json.loads(CONTROL_PATH.read_text(encoding="utf-8"))


def get_state() -> str:
    return get_control().get("state", RUNNING_AUTONOMOUSLY)


def is_blocking() -> bool:
    return get_state() in BLOCKING_STATES


def _write_control(data: dict) -> dict:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=LOGS_DIR, prefix=".control_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, CONTROL_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return data


def transition(new_state: str, reason: Optional[str] = None, requested_by: str = "engineer") -> dict:
    """Move to `new_state` if the current state allows it; raise
    IllegalTransitionError otherwise. A same-state "transition" is always
    a harmless no-op (re-written with a fresh timestamp/reason).
    """
    if new_state not in ALL_STATES:
        raise ValueError(f"unknown control state {new_state!r}")
    current = get_state()
    if new_state != current and new_state not in VALID_TRANSITIONS.get(current, set()):
        raise IllegalTransitionError(
            f"cannot transition from {current!r} to {new_state!r}; "
            f"valid next states are {sorted(VALID_TRANSITIONS.get(current, set()))}"
        )
    data = {
        "state": new_state,
        "previous_state": current,
        "reason": reason,
        "requested_by": requested_by,
        "updated_at": _now(),
    }
    _write_control(data)
    log_event("control_transition", from_state=current, to_state=new_state, reason=reason, requested_by=requested_by)
    return data


def reset() -> dict:
    """Force the state back to RUNNING_AUTONOMOUSLY unconditionally --
    used only once, at the start of a fresh run_gotrain.py invocation, so
    a stale control.json left over from a previous session's pause/manual
    takeover doesn't immediately block a brand new run.
    """
    current = get_state()
    data = {
        "state": RUNNING_AUTONOMOUSLY,
        "previous_state": current,
        "reason": "session start",
        "requested_by": "system",
        "updated_at": _now(),
    }
    _write_control(data)
    log_event(
        "control_transition", from_state=current, to_state=RUNNING_AUTONOMOUSLY,
        reason="session start", requested_by="system",
    )
    return data


def acknowledge_pause() -> dict:
    """Agent-side: move PAUSE_REQUESTED -> PAUSED_FOR_ENGINEER the moment
    the agent actually stops (its next tool call attempt is blocked). A
    no-op (returns the current control dict unchanged) if the state isn't
    PAUSE_REQUESTED.
    """
    if get_state() == PAUSE_REQUESTED:
        return transition(PAUSED_FOR_ENGINEER, reason="agent acknowledged pause", requested_by="agent")
    return get_control()


def raise_if_blocking() -> None:
    """Defense-in-depth check for gotrain/tools/training.py: raise
    ControlBlockedError if the control state currently blocks training.
    The agent/hook pipeline (gotrain/agent.py) already blocks every tool
    call in these states before launch_training is ever reached, but this
    guards any other caller (a script, a test, a future tool) too.
    """
    state = get_state()
    if state in BLOCKING_STATES:
        raise ControlBlockedError(
            f"GoTrain is currently {state} -- no training may be launched until "
            "an engineer resumes."
        )


def should_stop_training() -> Optional[str]:
    """Called once per epoch by the training templates (best-effort --
    never allowed to fail a real training run; see report_progress in the
    templates for the same pattern). Returns a short machine-readable
    reason if the current control state means an in-progress run should
    stop early and preserve whatever checkpoint it already has, else None.
    """
    state = get_state()
    if state in (PAUSE_REQUESTED, PAUSED_FOR_ENGINEER):
        return "paused_by_engineer"
    if state == MANUAL_CONTROL:
        return "manual_control_takeover"
    if state in (ROLLBACK_REQUESTED, ROLLBACK_VERIFYING):
        return "rollback_in_progress"
    return None


def wait_for_resume(poll_seconds: float = 3.0, sleep_fn=None) -> None:
    """Block until the control state leaves BLOCKING_STATES, completing a
    pending RESUME_REQUESTED -> RUNNING_AUTONOMOUSLY transition the moment
    it's observed. `sleep_fn` defaults to time.sleep; tests inject a fake
    clock instead of a real sleep so this polling loop's logic (not its
    timing) is what's actually verified.

    Used by run_gotrain.py's main loop to implement real pause/resume
    across multiple agent() calls -- see that file for the caveat that
    this specific integration has not been exercised against a live
    Bedrock call in this repo's test suite (no AWS access in CI/dev).
    """
    import time as _time

    sleep_fn = sleep_fn or _time.sleep
    while True:
        state = get_state()
        if state == RESUME_REQUESTED:
            transition(RUNNING_AUTONOMOUSLY, reason="resumed", requested_by="agent")
            return
        if state not in BLOCKING_STATES:
            return
        sleep_fn(poll_seconds)
