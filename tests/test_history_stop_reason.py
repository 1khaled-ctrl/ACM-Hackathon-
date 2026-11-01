"""Tests for gotrain.history.derive_stop_reason -- the fix for mission
history showing only "STOPPED, 0/3" with no indication of why. Reuses the
history_env fixture pattern from tests/test_history.py (isolated
GOTRAIN_LOGS_DIR + module reload, since gotrain.logging_utils/ledger/
history all capture LOGS_DIR at import time).

The underlying bug: gotrain.tools.training.launch_training already logs
exactly what went wrong (training_timed_out / training_failed, with the
timeout/stderr detail) every time a run fails -- but
dashboard/api/server.py's reconcile_open_session only ever recorded the
generic control-plane state, never looked at those events, so the specific
reason was computed and then thrown away. derive_stop_reason is the piece
that reads it back.
"""

import importlib

import pytest


@pytest.fixture
def history_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GOTRAIN_LOGS_DIR", str(tmp_path))

    from gotrain import logging_utils

    importlib.reload(logging_utils)
    from gotrain import ledger

    importlib.reload(ledger)
    from gotrain import history

    importlib.reload(history)

    yield history, ledger, logging_utils

    monkeypatch.delenv("GOTRAIN_LOGS_DIR", raising=False)
    importlib.reload(logging_utils)
    importlib.reload(ledger)
    importlib.reload(history)


def test_stop_reason_names_a_timeout_with_zero_completed_runs(history_env):
    """The m-bigearthnet pattern from the task brief: every launch_training
    attempt timed out before completing, so the ledger has nothing, but
    events.jsonl recorded the timeout every time.
    """
    history, _, logging_utils = history_env
    rec = history.start_session(dataset="m-bigearthnet", max_runs=3, run_minutes=5)

    logging_utils.log_event("training_started", dataset="m-bigearthnet", task_type="classification", config={})
    logging_utils.log_event("training_timed_out", dataset="m-bigearthnet", task_type="classification", timeout_seconds=300)

    reason = history.derive_stop_reason("m-bigearthnet", rec["started_at"])
    assert reason is not None
    assert "timed out" in reason
    assert "300" in reason


def test_stop_reason_names_a_crash_with_the_stderr_tail(history_env):
    """The m-cashew-plant / m-SA-crop-type pattern: training_failed was
    logged with the subprocess's captured stderr -- the reason should
    surface the actual error, not just "something went wrong".
    """
    history, _, logging_utils = history_env
    rec = history.start_session(dataset="m-cashew-plant", max_runs=3, run_minutes=5)

    logging_utils.log_event(
        "training_failed", dataset="m-cashew-plant", task_type="segmentation",
        stderr="Traceback (most recent call last):\n  ...\nRuntimeError: CUDA out of memory",
    )

    reason = history.derive_stop_reason("m-cashew-plant", rec["started_at"])
    assert reason is not None
    assert "crashed" in reason
    assert "CUDA out of memory" in reason


def test_stop_reason_uses_the_most_recent_training_event(history_env):
    """Several attempts happened -- the reason should reflect the LAST
    one (what actually ended the mission), not the first.
    """
    history, _, logging_utils = history_env
    rec = history.start_session(dataset="m-bigearthnet", max_runs=3, run_minutes=5)

    logging_utils.log_event("training_failed", dataset="m-bigearthnet", task_type="classification", stderr="ImportError: bad import")
    logging_utils.log_event("training_timed_out", dataset="m-bigearthnet", task_type="classification", timeout_seconds=300)

    reason = history.derive_stop_reason("m-bigearthnet", rec["started_at"])
    assert "timed out" in reason  # the LAST event, not the ImportError


def test_stop_reason_none_when_training_completed_and_submitted(history_env):
    """A mission that actually finished normally (training_finished +
    final_model_submitted) should not get a fabricated stop reason --
    this function is for STOPPED missions, and a caller only calls it in
    that case, but it should still refuse to invent one here.
    """
    history, _, logging_utils = history_env
    rec = history.start_session(dataset="m-eurosat", max_runs=1, run_minutes=5)

    logging_utils.log_event("training_finished", dataset="m-eurosat", run_id="run-1", val_metric=0.9)
    logging_utils.log_event("final_model_submitted", dataset="m-eurosat", run_id="run-1")

    reason = history.derive_stop_reason("m-eurosat", rec["started_at"])
    assert reason is None


def test_stop_reason_none_when_nothing_happened_at_all(history_env):
    """The agent process died before ever calling launch_training (e.g.
    crashed on startup) -- there's no training event to point to, so this
    must return None rather than guess.
    """
    history, _, _ = history_env
    rec = history.start_session(dataset="m-eurosat", max_runs=3, run_minutes=5)

    reason = history.derive_stop_reason("m-eurosat", rec["started_at"])
    assert reason is None


def test_stop_reason_none_for_missing_dataset(history_env):
    history, _, _ = history_env
    assert history.derive_stop_reason(None, "2026-01-01T00:00:00+00:00") is None


def test_stop_reason_is_persisted_on_the_session_and_visible_in_list_sessions(history_env):
    """End-to-end: finish_session(status=STOPPED, stop_reason=...) must
    make it back out through list_sessions()/load_session() the same way
    ended_in_state already does, since that's what the History UI reads.
    """
    history, _, logging_utils = history_env
    rec = history.start_session(dataset="m-bigearthnet", max_runs=3, run_minutes=5)
    logging_utils.log_event("training_timed_out", dataset="m-bigearthnet", task_type="classification", timeout_seconds=300)

    reason = history.derive_stop_reason("m-bigearthnet", rec["started_at"])
    history.finish_session(rec["session_id"], status=history.STOPPED, stop_reason=reason)

    sessions = history.list_sessions()
    assert sessions[0]["stop_reason"] == reason
    assert sessions[0]["status"] == history.STOPPED

    loaded = history.load_session(rec["session_id"])
    assert loaded["stop_reason"] == reason
