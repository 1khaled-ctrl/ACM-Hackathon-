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


# ---------------------------------------------------------------------
# Regression: a newer mission on a DIFFERENT dataset must never surface
# an older mission's runs. Hit for real: NEXUS showed m-eurosat runs
# while m-nz-cattle was the active mission, traced to
# dashboard/api/server.py's /api/snapshot building its "history" field
# from ledger.get_history() with no session/dataset scoping at all. The
# fix added build_current_mission() (server.py), which calls exactly
# this module's get_current_session() + load_session() -- these tests
# pin down that those two functions can't blend two missions' runs.
# ---------------------------------------------------------------------

def test_a_newer_different_dataset_mission_cannot_show_an_older_missions_runs(history_env):
    """The exact bug: m-eurosat mission finishes, m-nz-cattle mission
    starts later and is now "current" -- get_current_session() +
    load_session() must return ONLY m-nz-cattle's own run, never the
    m-eurosat one, no matter how strong or recent-looking it is.
    """
    history, ledger, _ = history_env

    old = history.start_session(dataset="m-eurosat", max_runs=2, run_minutes=4)
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.95,
                    task_type="classification", decision="kept")
    history.finish_session(old["session_id"], status=history.COMPLETED)

    new = history.start_session(dataset="m-nz-cattle", max_runs=3, run_minutes=10)
    ledger.add_run(dataset="m-nz-cattle", config={"encoder": "resnet34"}, val_metric=0.61,
                    task_type="segmentation", decision="kept")

    current = history.get_current_session()
    assert current is not None
    assert current["session_id"] == new["session_id"]
    assert current["dataset"] == "m-nz-cattle"

    detail = history.load_session(current["session_id"])
    assert detail["dataset"] == "m-nz-cattle"
    assert len(detail["runs"]) == 1
    assert all(r["dataset"] == "m-nz-cattle" for r in detail["runs"])
    # The specific failure mode that was actually observed: the old
    # dataset's name must not appear anywhere in the current mission's
    # scoped runs.
    assert not any(r["dataset"] == "m-eurosat" for r in detail["runs"])


def test_get_current_session_prefers_the_running_one_over_a_newer_finished_one(history_env):
    """If an OLDER session is still RUNNING while a NEWER one has already
    finished (e.g. two dashboards, or a race during reconciliation), the
    live dashboard must attribute to the one actually in progress --
    showing a finished mission's data while a different one is training
    would be just as wrong as the cross-dataset bug.
    """
    history, _, _ = history_env

    still_running = history.start_session(dataset="m-eurosat", max_runs=2, run_minutes=4)
    finished_after = history.start_session(dataset="m-nz-cattle", max_runs=1, run_minutes=1)
    history.finish_session(finished_after["session_id"], status=history.COMPLETED)

    current = history.get_current_session()
    assert current["session_id"] == still_running["session_id"]
    assert current["status"] == history.RUNNING


def test_get_current_session_falls_back_to_most_recent_when_nothing_is_running(history_env):
    """No RUNNING session -> the live panels should keep showing the last
    mission's final state rather than going blank, so completing a
    mission doesn't make the dashboard look broken.
    """
    history, _, _ = history_env

    first = history.start_session(dataset="m-eurosat", max_runs=1, run_minutes=1)
    history.finish_session(first["session_id"], status=history.COMPLETED)
    second = history.start_session(dataset="m-nz-cattle", max_runs=1, run_minutes=1)
    history.finish_session(second["session_id"], status=history.COMPLETED)

    current = history.get_current_session()
    assert current["session_id"] == second["session_id"]


def test_get_current_session_returns_none_when_no_mission_ever_launched(history_env):
    history, _, _ = history_env
    assert history.get_current_session() is None


def test_approval_decided_events_are_merged_into_the_session_event_stream(history_env):
    """approval_requested/approval_decided live in logs/approvals.jsonl,
    a separate file from events.jsonl (see gotrain/approval.py). History
    must record engineer approve/reject actions, so load_session()'s
    event stream has to merge both files, not just read events.jsonl.
    """
    history, _, logging_utils = history_env
    from gotrain import approval
    import importlib
    importlib.reload(approval)

    session = history.start_session(dataset="m-eurosat", max_runs=2, run_minutes=4)
    proposal = approval.request_approval("m-eurosat", "classification", "backbone", "convnext_tiny", "try a bigger backbone")
    approval.decide(proposal["key"], True, decided_by="engineer", decision_reason="worth trying")
    history.finish_session(session["session_id"], status=history.COMPLETED)

    detail = history.load_session(session["session_id"])
    event_types = [e["event"] for e in detail["events"]]
    assert "approval_requested" in event_types
    assert "approval_decided" in event_types

    decided = [e for e in detail["events"] if e["event"] == "approval_decided"][0]
    assert decided["status"] == "approved"
    assert decided["decided_by"] == "engineer"

    # approval_decided counts as an engineer intervention for History's
    # "pause/resume/approval/reject/rollback/manual-control" requirement.
    intervention_types = [e["event"] for e in detail["interventions"]]
    assert "approval_decided" in intervention_types
    # approval_requested is the agent's own proposal, not an engineer
    # action -- it must NOT be counted as an intervention.
    assert "approval_requested" not in intervention_types


def test_rollback_events_count_as_interventions(history_env):
    history, _, logging_utils = history_env

    session = history.start_session(dataset="m-eurosat", max_runs=2, run_minutes=4)
    logging_utils.log_event("rollback_succeeded", dataset="m-eurosat", version="v2", verification={})
    history.finish_session(session["session_id"], status=history.COMPLETED)

    detail = history.load_session(session["session_id"])
    intervention_types = [e["event"] for e in detail["interventions"]]
    assert "rollback_succeeded" in intervention_types


def test_windowed_events_are_deduplicated(history_env):
    """Two identical events (same timestamp/type/run_id) landing in the
    window -- e.g. a reconciler re-observing something already logged --
    must appear once in the merged stream, not twice.
    """
    history, _, _ = history_env

    session = history.start_session(dataset="m-eurosat", max_runs=1, run_minutes=1)
    start = history._parse(session["started_at"])
    end = None

    dup_event = {"timestamp": session["started_at"], "event": "training_started", "dataset": "m-eurosat"}
    from gotrain.logging_utils import EVENTS_PATH
    import json as _json
    with open(EVENTS_PATH, "a", encoding="utf-8") as f:
        f.write(_json.dumps(dup_event) + "\n")
        f.write(_json.dumps(dup_event) + "\n")

    merged = history._windowed_events(start, end)
    matching = [e for e in merged if e.get("event") == "training_started"]
    assert len(matching) == 1


# ---------------------------------------------------------------------
# Training Health card (gotrain/history.py::_training_health). Reuses
# gotrain/experiment_planner.py::get_training_diagnostics exactly --
# these tests pin down that the mapping from its real diagnoses
# (possible_overfitting / possible_underfitting / ...) to the card's
# HEALTHY/OVERFITTING/UNDERFITTING/UNKNOWN states is correct, and that a
# missing/unavailable diagnosis degrades to UNKNOWN rather than raising.
# UNSTABLE is intentionally not exercised here: the backend has no
# diagnosis that maps to it (see _training_health's own docstring), so
# there is nothing yet to test beyond "the constant exists".
# ---------------------------------------------------------------------

@pytest.fixture
def diagnostics_env(history_env, monkeypatch):
    """Extends history_env with experiment_planner reloaded against the
    same temp LOGS_DIR, and its RUNS_DIR redirected into tmp_path -- the
    same isolation pattern tests/test_experiment_planner.py already uses
    for this exact module.
    """
    history, ledger, logging_utils = history_env

    from gotrain import experiment_planner as planner

    importlib.reload(planner)
    monkeypatch.setattr(planner, "RUNS_DIR", logging_utils.LOGS_DIR / "runs")

    yield history, ledger, planner


def _write_run_metrics(planner, dataset, run_id, train_loss):
    run_dir = planner.RUNS_DIR / dataset / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps({"best": {"train_loss": train_loss}}), encoding="utf-8")


def test_training_health_unknown_with_no_runs(diagnostics_env):
    history, _, _ = diagnostics_env
    assert history._training_health("m-eurosat", []) == {"state": "UNKNOWN", "evidence": []}


def test_training_health_unknown_with_no_dataset(diagnostics_env):
    """mission.dataset can be falsy on a half-written session record --
    must degrade to UNKNOWN, not raise, even with runs present.
    """
    history, ledger, _ = diagnostics_env
    run = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.8,
                          task_type="classification", decision="kept")
    assert history._training_health(None, [run]) == {"state": "UNKNOWN", "evidence": []}


def test_training_health_healthy_when_no_diagnosis_fires(diagnostics_env):
    """A run with no metrics.json (train_loss unavailable) and a val_metric
    that doesn't trigger any threshold: get_training_diagnostics legitimately
    returns an empty diagnoses list -- that is HEALTHY, not UNKNOWN (the
    diagnosis ran successfully and found nothing concerning).
    """
    history, ledger, _ = diagnostics_env
    run = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.85,
                          task_type="classification", decision="kept")
    result = history._training_health("m-eurosat", [run])
    assert result["state"] == "HEALTHY"
    assert result["run_id"] == run["run_id"]


def test_training_health_overfitting_maps_from_possible_overfitting(diagnostics_env):
    """The real overfitting heuristic: train_loss below threshold (0.15)
    AND val_metric below the classification ceiling (0.90).
    """
    history, ledger, planner = diagnostics_env
    run = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.82,
                          task_type="classification", decision="pending")
    _write_run_metrics(planner, "m-eurosat", run["run_id"], train_loss=0.08)

    result = history._training_health("m-eurosat", [run])
    assert result["state"] == "OVERFITTING"
    assert len(result["evidence"]) == 1
    assert "0.080" in result["evidence"][0]
    assert "0.820" in result["evidence"][0]


def test_training_health_underfitting_maps_from_possible_underfitting(diagnostics_env):
    """The real underfitting heuristic: train_loss above the floor (0.8)
    AND val_metric below the classification ceiling (0.5).
    """
    history, ledger, planner = diagnostics_env
    run = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.3,
                          task_type="classification", decision="pending")
    _write_run_metrics(planner, "m-eurosat", run["run_id"], train_loss=0.95)

    result = history._training_health("m-eurosat", [run])
    assert result["state"] == "UNDERFITTING"
    assert len(result["evidence"]) == 1


def test_training_health_diagnoses_the_most_recent_run(diagnostics_env):
    """Two runs in the session: health must reflect the LATEST one, not
    the first (e.g. an early underfitting run followed by a healthy fix
    must show HEALTHY, not the stale earlier diagnosis).
    """
    history, ledger, planner = diagnostics_env
    first = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.3,
                            task_type="classification", decision="discarded")
    _write_run_metrics(planner, "m-eurosat", first["run_id"], train_loss=0.95)

    second = ledger.add_run(dataset="m-eurosat", config={"backbone": "convnext_tiny"}, val_metric=0.91,
                             task_type="classification", decision="kept", parent_id=first["run_id"])

    result = history._training_health("m-eurosat", [first, second])
    assert result["run_id"] == second["run_id"]
    assert result["state"] == "HEALTHY"


def test_training_health_unknown_when_run_id_not_in_ledger(diagnostics_env):
    """get_training_diagnostics raises ValueError for an unknown run_id
    (e.g. the run record passed in came from a different dataset's
    ledger, or was deleted) -- must degrade to UNKNOWN, not propagate.
    """
    history, _, _ = diagnostics_env
    fake_run = {"run_id": "does-not-exist", "timestamp": "2026-01-01T00:00:00+00:00"}
    result = history._training_health("m-eurosat", [fake_run])
    assert result == {"state": "UNKNOWN", "evidence": []}


def test_training_health_is_present_in_derived_session_summary(diagnostics_env):
    """End-to-end through the real entry points list_sessions() and
    load_session() both call (_derive) -- the field this whole feature
    depends on must actually be attached, not just computable in
    isolation.
    """
    history, ledger, planner = diagnostics_env
    session = history.start_session(dataset="m-eurosat", max_runs=2, run_minutes=4)
    run = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.82,
                          task_type="classification", decision="kept")
    _write_run_metrics(planner, "m-eurosat", run["run_id"], train_loss=0.08)
    history.finish_session(session["session_id"], status=history.COMPLETED)

    listed = history.list_sessions()[0]
    assert listed["training_health"]["state"] == "OVERFITTING"

    loaded = history.load_session(session["session_id"])
    assert loaded["training_health"]["state"] == "OVERFITTING"
    assert loaded["training_health"] == listed["training_health"]


def test_training_health_states_include_unstable_for_future_use(diagnostics_env):
    """UNSTABLE is a defined state constant even though nothing currently
    produces it (see _training_health's docstring) -- this just confirms
    the constant exists with the expected string value, so the frontend's
    HEALTH_TONE mapping has something real to key off if/when a backend
    diagnosis for it is added.
    """
    history, _, _ = diagnostics_env
    assert history.HEALTH_UNSTABLE == "UNSTABLE"


# ---------------------------------------------------------------------
# Statistical Improvement Verification (gotrain/history.py::
# _attach_statistical_verdicts). Reuses gotrain/statistics.py's
# estimate_noise_floor + is_improvement_significant exactly -- these
# tests pin down the VERIFIED/NOT_ABOVE_NOISE/INSUFFICIENT_EVIDENCE
# mapping from that module's existing significant=True/False/None.
# ---------------------------------------------------------------------

@pytest.fixture
def statistics_env(history_env):
    """gotrain/statistics.py does `from gotrain import ledger` at its own
    import time -- must be reloaded here too (not just ledger itself) so
    its bound `ledger` reference points at THIS test's isolated module,
    not whatever gotrain.statistics last saw in a previous test.
    """
    history, ledger, _ = history_env

    from gotrain import statistics

    importlib.reload(statistics)

    yield history, ledger, statistics


def test_statistical_verdict_insufficient_evidence_with_no_delta(statistics_env):
    """A root run (no parent_id, delta_vs_parent is None) can never be
    judged -- there is nothing to compare.
    """
    history, ledger, _ = statistics_env
    root = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.8,
                           task_type="classification", decision="kept")
    out = history._attach_statistical_verdicts("m-eurosat", [root])
    assert out[0]["statistical_verdict"]["state"] == history.STAT_INSUFFICIENT_EVIDENCE


def test_statistical_verdict_insufficient_evidence_with_no_repeated_configs(statistics_env):
    """A real delta exists, but no two runs share an identical (non-seed)
    config yet -- no noise floor can be estimated, so even a real-looking
    delta must not be called VERIFIED.
    """
    history, ledger, _ = statistics_env
    root = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.80,
                           task_type="classification", decision="kept")
    child = ledger.add_run(dataset="m-eurosat", config={"backbone": "convnext_tiny"}, val_metric=0.90,
                            parent_id=root["run_id"], task_type="classification", decision="kept")
    out = history._attach_statistical_verdicts("m-eurosat", [root, child])
    verdicts = {n["run_id"]: n["statistical_verdict"]["state"] for n in out}
    assert verdicts[child["run_id"]] == history.STAT_INSUFFICIENT_EVIDENCE


def test_statistical_verdict_verified_when_delta_exceeds_noise_floor(statistics_env):
    """Two repeated-config runs establish a small noise floor; a third
    run's delta clearly exceeds it -- that delta must be VERIFIED.
    """
    history, ledger, _ = statistics_env
    r1 = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "seed": 1}, val_metric=0.800,
                         task_type="classification", decision="kept")
    r2 = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "seed": 2}, val_metric=0.802,
                         task_type="classification", decision="kept")
    big_improvement = ledger.add_run(dataset="m-eurosat", config={"backbone": "convnext_tiny"}, val_metric=0.95,
                                      parent_id=r2["run_id"], task_type="classification", decision="kept")

    out = history._attach_statistical_verdicts("m-eurosat", [r1, r2, big_improvement])
    verdicts = {n["run_id"]: n["statistical_verdict"] for n in out}
    assert verdicts[big_improvement["run_id"]]["state"] == history.STAT_VERIFIED
    assert verdicts[big_improvement["run_id"]]["noise_floor"] == pytest.approx(0.002, abs=1e-6)


def test_statistical_verdict_not_above_noise_when_delta_is_within_noise_floor(statistics_env):
    """A repeated-config pair establishes a noise floor; a THIRD run with
    that same config (so its delta vs. one of the pair is itself part of
    the measured spread) must not be called a verified improvement.
    """
    history, ledger, _ = statistics_env
    r1 = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "seed": 1}, val_metric=0.800,
                         task_type="classification", decision="kept")
    r2 = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "seed": 2}, val_metric=0.840,
                         task_type="classification", decision="kept")
    # Delta vs parent (r1) is 0.01 -- well inside the 0.04 noise floor
    # established by r1/r2 sharing the identical non-seed config.
    small_change = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "seed": 3}, val_metric=0.810,
                                   parent_id=r1["run_id"], task_type="classification", decision="pending")

    out = history._attach_statistical_verdicts("m-eurosat", [r1, r2, small_change])
    verdicts = {n["run_id"]: n["statistical_verdict"]["state"] for n in out}
    assert verdicts[small_change["run_id"]] == history.STAT_NOT_ABOVE_NOISE


def test_statistical_verdict_is_present_in_mission_runs_end_to_end(statistics_env):
    """End-to-end through load_session(): every run in mission.runs must
    carry statistical_verdict, not just when _attach_statistical_verdicts
    is called directly.
    """
    history, ledger, _ = statistics_env
    session = history.start_session(dataset="m-eurosat", max_runs=1, run_minutes=1)
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.8,
                    task_type="classification", decision="kept")
    history.finish_session(session["session_id"], status=history.COMPLETED)

    loaded = history.load_session(session["session_id"])
    assert len(loaded["runs"]) == 1
    assert "statistical_verdict" in loaded["runs"][0]
    assert loaded["runs"][0]["statistical_verdict"]["state"] == history.STAT_INSUFFICIENT_EVIDENCE
