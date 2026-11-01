"""Tests for dashboard/api/server.py::reconcile_open_session's stop_reason
wiring -- confirms the actual end-to-end path a real client hits: an open
session whose process is gone gets closed with a reason attached, not just
a bare STOPPED status. Reuses the server_env fixture pattern from
tests/test_server_current_mission.py.

agent_process_alive() returns False whenever logs/agent.pid doesn't exist
(the default state in a fresh tmp_path), so these tests don't need to mock
psutil -- the reconciler proceeds straight to "process is gone, close the
session" exactly as it would after a real crash/timeout/kill.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def server_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GOTRAIN_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("GEO_BENCH_DIR", str(tmp_path / "geobench"))

    import importlib

    from gotrain import control, history, ledger, logging_utils

    importlib.reload(logging_utils)
    importlib.reload(ledger)
    importlib.reload(control)
    importlib.reload(history)

    sys.modules.pop("dashboard.api.server", None)
    # import_module, not `from dashboard.api import server`: the latter returns
    # the stale module cached as an attribute on the dashboard.api package.
    server = importlib.import_module("dashboard.api.server")

    yield server, history, ledger, control, logging_utils

    sys.modules.pop("dashboard.api.server", None)
    monkeypatch.delenv("GOTRAIN_LOGS_DIR", raising=False)
    monkeypatch.delenv("GEO_BENCH_DIR", raising=False)
    importlib.reload(logging_utils)
    importlib.reload(ledger)
    importlib.reload(control)
    importlib.reload(history)


def test_reconcile_closes_a_dead_session_as_stopped_with_a_reason(server_env):
    """The m-bigearthnet pattern: a mission timed out on every attempt
    (nothing ever got submitted, control state stays whatever it was
    left at, never COMPLETED/FAILED_SAFE), its process is gone, and the
    session must close as STOPPED with the actual reason attached -- not
    just "0/3" with no explanation.
    """
    server, history, _, _, logging_utils = server_env

    session = history.start_session(dataset="m-bigearthnet", max_runs=3, run_minutes=5)
    logging_utils.log_event(
        "training_timed_out", dataset="m-bigearthnet", task_type="classification", timeout_seconds=300
    )

    server.reconcile_open_session()

    loaded = history.load_session(session["session_id"])
    assert loaded["status"] == history.STOPPED
    assert loaded["stop_reason"] is not None
    assert "timed out" in loaded["stop_reason"]


def test_reconcile_closes_a_dead_session_with_crash_reason(server_env):
    server, history, _, _, logging_utils = server_env

    session = history.start_session(dataset="m-cashew-plant", max_runs=3, run_minutes=5)
    logging_utils.log_event(
        "training_failed", dataset="m-cashew-plant", task_type="segmentation",
        stderr="RuntimeError: mat1 and mat2 shapes cannot be multiplied",
    )

    server.reconcile_open_session()

    loaded = history.load_session(session["session_id"])
    assert loaded["status"] == history.STOPPED
    assert "crashed" in loaded["stop_reason"]
    assert "mat1 and mat2" in loaded["stop_reason"]


def test_reconcile_completed_session_gets_no_stop_reason(server_env):
    """A mission that actually finished (control state COMPLETED) must
    not get a stop_reason field treated as meaningful -- stop_reason is
    only computed/attached for the STOPPED branch.
    """
    server, history, _, control, _ = server_env

    session = history.start_session(dataset="m-eurosat", max_runs=1, run_minutes=5)
    from gotrain import logging_utils

    logging_utils.log_event("final_model_submitted", dataset="m-eurosat", run_id="run-1")
    control.transition(control.COMPLETED, reason="final model submitted: run-1", requested_by="agent")

    server.reconcile_open_session()

    loaded = history.load_session(session["session_id"])
    assert loaded["status"] == history.COMPLETED
    assert loaded.get("stop_reason") is None


def test_stale_completed_control_state_does_not_mark_an_empty_session_completed(server_env):
    """control.json persists across missions. If the new agent process dies
    before run_gotrain.py's control.reset() (bad credentials, import error),
    the previous mission's COMPLETED is still there. A session is COMPLETED
    only if IT submitted a final model."""
    server, history, _, control, _ = server_env
    control.transition(control.COMPLETED, reason="previous mission", requested_by="agent")

    session = history.start_session(dataset="m-pv4ger-seg", max_runs=3, run_minutes=5)
    server.reconcile_open_session()

    loaded = history.load_session(session["session_id"])
    assert loaded["status"] == history.STOPPED
    assert loaded["stop_reason"]


def test_agent_log_tail_explains_a_crash_before_any_tool_call(server_env, tmp_path):
    server, history, _, _, _ = server_env
    session = history.start_session(dataset="m-eurosat", max_runs=3, run_minutes=5)
    server.AGENT_LOG_PATH.write_text(
        "Traceback (most recent call last):\n  ...\nbotocore.exceptions.NoCredentialsError: Unable to locate credentials\n",
        encoding="utf-8",
    )
    server.reconcile_open_session()

    reason = history.load_session(session["session_id"])["stop_reason"]
    assert "NoCredentialsError" in reason


def test_early_exit_after_one_of_three_runs_is_stopped_with_counts(server_env):
    server, history, ledger, _, logging_utils = server_env
    session = history.start_session(dataset="m-eurosat", max_runs=3, run_minutes=5)
    run = ledger.add_run(dataset="m-eurosat", config={}, val_metric=0.71, task_type="classification")
    logging_utils.log_event("training_finished", dataset="m-eurosat", run_id=run["run_id"], val_metric=0.71)
    logging_utils.log_event("tool_call", tool="query_history", status="success")

    server.reconcile_open_session()
    loaded = history.load_session(session["session_id"])
    assert loaded["status"] == history.STOPPED
    assert loaded["runs_completed"] == 1 and loaded["runs_requested"] == 3
    assert loaded["best_run_id"] == run["run_id"]
    assert "before submitting" in loaded["stop_reason"]


def test_diminishing_returns_submit_after_two_of_three_is_completed(server_env):
    server, history, ledger, control, logging_utils = server_env
    session = history.start_session(dataset="m-eurosat", max_runs=3, run_minutes=5)
    a = ledger.add_run(dataset="m-eurosat", config={}, val_metric=0.80, task_type="classification")
    b = ledger.add_run(dataset="m-eurosat", config={"lr": 2e-3}, val_metric=0.801, task_type="classification")
    logging_utils.log_event("final_model_submitted", dataset="m-eurosat", run_id=b["run_id"])
    control.transition(control.COMPLETED, reason=f"final model submitted: {b['run_id']}", requested_by="agent")

    server.reconcile_open_session()
    loaded = history.load_session(session["session_id"])
    assert loaded["status"] == history.COMPLETED
    assert loaded["runs_completed"] == 2
    assert loaded["final_model_run_id"] == b["run_id"]
    assert loaded.get("stop_reason") is None
    assert a["run_id"] in loaded["run_ids"]


def test_failed_safe_is_failed_with_the_control_reason(server_env):
    server, history, _, control, _ = server_env
    session = history.start_session(dataset="m-eurosat", max_runs=3, run_minutes=5)
    control.transition(control.ROLLBACK_REQUESTED, reason="engineer rollback to v1", requested_by="engineer")
    control.transition(control.ROLLBACK_VERIFYING, reason="verifying v1", requested_by="engineer")
    control.transition(control.FAILED_SAFE, reason="rollback verification failed: model.pt missing", requested_by="system")

    server.reconcile_open_session()
    loaded = history.load_session(session["session_id"])
    assert loaded["status"] == history.FAILED
    assert "model.pt missing" in loaded["stop_reason"]


def test_reconcile_with_no_open_session_is_a_safe_noop(server_env):
    server, history, _, _, _ = server_env
    server.reconcile_open_session()  # must not raise
    assert history.get_open_session() is None
