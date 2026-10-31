"""Tests for gotrain/history.py -- the per-mission session records.

The risky logic here is time-window attribution: a session claims the
ledger runs and events that fall between its started_at and finished_at.
Get that wrong and one mission's results show up under another's, which
would be worse than having no history at all -- a wrong number presented
confidently. These tests pin that boundary behaviour down.
"""

import importlib
import json

import pytest


@pytest.fixture
def history_env(tmp_path, monkeypatch):
    """Point every logs/ path at a temp dir and reload the modules that
    captured LOGS_DIR at import time.
    """
    monkeypatch.setenv("GOTRAIN_LOGS_DIR", str(tmp_path))

    from gotrain import logging_utils

    importlib.reload(logging_utils)
    from gotrain import ledger

    importlib.reload(ledger)
    from gotrain import history

    importlib.reload(history)

    yield history, ledger, logging_utils

    # Restore module state for any test that follows.
    monkeypatch.delenv("GOTRAIN_LOGS_DIR", raising=False)
    importlib.reload(logging_utils)
    importlib.reload(ledger)
    importlib.reload(history)


def test_start_session_writes_a_running_record(history_env):
    history, _, _ = history_env
    rec = history.start_session(dataset="m-eurosat", max_runs=3, run_minutes=4, max_tool_calls=30, pid=123)

    assert rec["status"] == history.RUNNING
    assert rec["dataset"] == "m-eurosat"
    assert rec["runs_requested"] == 3
    assert rec["minutes_per_run"] == 4
    assert rec["max_tool_calls"] == 30
    assert rec["started_at"]
    assert rec["finished_at"] is None

    # And it is readable back from disk, not just returned in memory.
    on_disk = history.list_sessions()
    assert len(on_disk) == 1
    assert on_disk[0]["session_id"] == rec["session_id"]


def test_finish_session_stamps_status_and_duration(history_env):
    history, _, _ = history_env
    rec = history.start_session(dataset="m-eurosat", max_runs=1, run_minutes=1)

    done = history.finish_session(rec["session_id"], status=history.COMPLETED)

    assert done["status"] == history.COMPLETED
    assert done["finished_at"]
    assert done["duration_seconds"] >= 0


def test_finish_session_never_restamps_a_closed_session(history_env):
    """A second close must not overwrite the real end time -- the
    snapshot reconciler can fire repeatedly, and a mission's recorded
    duration must not grow every time someone loads the dashboard.
    """
    history, _, _ = history_env
    rec = history.start_session(dataset="m-eurosat", max_runs=1, run_minutes=1)

    first = history.finish_session(rec["session_id"], status=history.COMPLETED)
    second = history.finish_session(rec["session_id"], status=history.STOPPED)

    assert second["finished_at"] == first["finished_at"]
    assert second["status"] == history.COMPLETED  # not downgraded to STOPPED


def test_finish_session_on_unknown_id_is_a_noop(history_env):
    history, _, _ = history_env
    assert history.finish_session("session_does_not_exist") == {}


def test_get_open_session_finds_only_running_ones(history_env):
    history, _, _ = history_env
    closed = history.start_session(dataset="m-eurosat", max_runs=1, run_minutes=1)
    history.finish_session(closed["session_id"])
    still_open = history.start_session(dataset="m-nz-cattle", max_runs=2, run_minutes=2)

    found = history.get_open_session()
    assert found is not None
    assert found["session_id"] == still_open["session_id"]


def test_runs_are_attributed_to_the_session_that_was_open(history_env):
    """The core guarantee: a run recorded during session A belongs to A,
    and a run recorded after A closed does not.
    """
    history, ledger, _ = history_env

    first = history.start_session(dataset="m-eurosat", max_runs=1, run_minutes=1)
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.90)
    history.finish_session(first["session_id"], status=history.COMPLETED)

    second = history.start_session(dataset="m-eurosat", max_runs=1, run_minutes=1)
    ledger.add_run(dataset="m-eurosat", config={"backbone": "convnext_tiny"}, val_metric=0.95)
    history.finish_session(second["session_id"], status=history.COMPLETED)

    a = history.load_session(first["session_id"])
    b = history.load_session(second["session_id"])

    assert a["runs_completed"] == 1
    assert b["runs_completed"] == 1
    assert a["best_val_metric"] == 0.90
    assert b["best_val_metric"] == 0.95
    assert a["best_model"] == "resnet18"
    assert b["best_model"] == "convnext_tiny"
    # The two sessions must not share a run id.
    assert set(a["run_ids"]).isdisjoint(b["run_ids"])


def test_an_open_session_claims_runs_recorded_after_it_started(history_env):
    """A session with no finished_at yet is still accumulating -- it must
    pick up runs as they land, so a live mission's count isn't stuck at 0.
    """
    history, ledger, _ = history_env
    rec = history.start_session(dataset="m-eurosat", max_runs=2, run_minutes=1)
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.80)

    live = history.load_session(rec["session_id"])
    assert live["status"] == history.RUNNING
    assert live["runs_completed"] == 1


def test_runs_recorded_before_the_session_are_excluded(history_env):
    """Pre-existing ledger history must not be absorbed into a brand new
    mission -- that would show a fresh session claiming someone else's
    best score.
    """
    history, ledger, _ = history_env
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.99)

    rec = history.start_session(dataset="m-eurosat", max_runs=1, run_minutes=1)
    loaded = history.load_session(rec["session_id"])

    assert loaded["runs_completed"] == 0
    assert loaded["best_val_metric"] is None


def test_session_detail_includes_events_and_curves(history_env):
    history, ledger, logging_utils = history_env
    rec = history.start_session(dataset="m-eurosat", max_runs=1, run_minutes=1)

    run = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.9)
    logging_utils.log_live_metric(
        run_id=run["run_id"], dataset="m-eurosat", epoch=1, train_loss=0.5, val_metric=0.9
    )
    logging_utils.log_event("final_model_submitted", dataset="m-eurosat", run_id=run["run_id"])
    history.finish_session(rec["session_id"], status=history.COMPLETED)

    loaded = history.load_session(rec["session_id"])
    assert len(loaded["runs"]) == 1
    assert len(loaded["live_metrics"]) == 1
    assert loaded["final_model_run_id"] == run["run_id"]


def test_timed_out_runs_and_engineer_interventions_are_surfaced(history_env):
    history, _, logging_utils = history_env
    rec = history.start_session(dataset="m-eurosat", max_runs=1, run_minutes=1)

    logging_utils.log_event("training_timed_out", dataset="m-eurosat", timeout_seconds=60)
    logging_utils.log_event(
        "control_transition", from_state="RUNNING_AUTONOMOUSLY", to_state="PAUSE_REQUESTED",
        reason="engineer clicked pause", requested_by="engineer",
    )
    # An agent-initiated transition is not an engineer intervention.
    logging_utils.log_event(
        "control_transition", from_state="PAUSE_REQUESTED", to_state="PAUSED_FOR_ENGINEER",
        reason="agent acknowledged pause", requested_by="agent",
    )
    history.finish_session(rec["session_id"], status=history.STOPPED)

    loaded = history.load_session(rec["session_id"])
    assert loaded["timed_out_runs"] == 1
    assert len(loaded["interventions"]) == 1
    assert loaded["interventions"][0]["requested_by"] == "engineer"


def test_load_session_returns_none_for_unknown_id(history_env):
    history, _, _ = history_env
    assert history.load_session("session_nope") is None


def test_session_id_cannot_escape_the_history_directory(history_env):
    """session_id reaches _session_path from an HTTP path parameter, so a
    traversal attempt must not resolve outside logs/history/.
    """
    history, _, _ = history_env
    assert history.load_session("../../etc/passwd") is None
    assert history.finish_session("../../etc/passwd") == {}


def test_a_corrupt_session_file_does_not_break_the_listing(history_env):
    history, _, _ = history_env
    good = history.start_session(dataset="m-eurosat", max_runs=1, run_minutes=1)
    (history.HISTORY_DIR / "session_corrupt.json").write_text("{not json", encoding="utf-8")

    sessions = history.list_sessions()
    assert [s["session_id"] for s in sessions] == [good["session_id"]]


def test_sessions_are_listed_newest_first(history_env):
    history, _, _ = history_env
    first = history.start_session(dataset="m-eurosat", max_runs=1, run_minutes=1)
    second = history.start_session(dataset="m-nz-cattle", max_runs=1, run_minutes=1)

    ids = [s["session_id"] for s in history.list_sessions()]
    assert ids == [second["session_id"], first["session_id"]]


def test_history_module_never_touches_the_test_split(history_env):
    """gotrain/history.py must not import the final-eval tool or name the
    test-lock files in executable code. Checked via the AST rather than a
    text grep, so the module's own docstring explaining this guarantee
    doesn't trip the assertion (and so a comment can't hide a violation).
    """
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "gotrain" / "history.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported += [f"{node.module or ''}.{a.name}" for a in node.names]
    assert not any("final_eval" in name for name in imported), imported

    # String constants in executable positions. A docstring is also a
    # Constant node, so collect the actual docstring nodes (by identity,
    # not by normalized text) and skip exactly those -- the module's own
    # prose explaining this guarantee must not count as a violation.
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    docstring_nodes.add(id(body[0].value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstring_nodes:
                continue
            assert "test_lock" not in node.value, node.value
            assert "final_eval" not in node.value, node.value
