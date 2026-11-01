"""Tests for dashboard/api/server.py::build_current_mission -- the field
that fixes NEXUS's Experiment Timeline showing a PREVIOUS mission's runs
(e.g. m-eurosat) while a DIFFERENT mission (e.g. m-nz-cattle) is active.

Root cause that was actually hit: /api/snapshot's "history" field came
from ledger.get_history() with no session or dataset scoping at all --
the entire global ledger, every mission, every dataset, forever. The fix
adds build_current_mission(), which calls gotrain.history's
get_current_session() + load_session() (the exact functions
tests/test_history.py already pins down for cross-mission isolation) and
exposes the result as snapshot["mission"]. These tests confirm the
server-level wiring itself -- that build_current_mission() actually
returns the scoped detail and not the old unscoped global history.
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

    from gotrain import history, ledger, logging_utils

    importlib.reload(logging_utils)
    importlib.reload(ledger)
    importlib.reload(history)

    sys.modules.pop("dashboard.api.server", None)
    from dashboard.api import server

    yield server, history, ledger

    sys.modules.pop("dashboard.api.server", None)
    monkeypatch.delenv("GOTRAIN_LOGS_DIR", raising=False)
    monkeypatch.delenv("GEO_BENCH_DIR", raising=False)
    importlib.reload(logging_utils)
    importlib.reload(ledger)
    importlib.reload(history)


def test_no_mission_ever_launched_returns_none(server_env):
    server, _, _ = server_env
    assert server.build_current_mission() is None


def test_build_current_mission_scopes_to_the_newer_dataset_only(server_env):
    """The actual bug, reproduced through the server function a real
    client calls: an m-eurosat mission finishes, an m-nz-cattle mission
    starts after it -- build_current_mission() must return ONLY the
    m-nz-cattle mission's own run.
    """
    server, history, ledger = server_env

    old = history.start_session(dataset="m-eurosat", max_runs=2, run_minutes=4)
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.95,
                    task_type="classification", decision="kept")
    history.finish_session(old["session_id"], status=history.COMPLETED)

    new = history.start_session(dataset="m-nz-cattle", max_runs=3, run_minutes=10)
    ledger.add_run(dataset="m-nz-cattle", config={"encoder": "resnet34"}, val_metric=0.61,
                    task_type="segmentation", decision="kept")

    mission = server.build_current_mission()
    assert mission is not None
    assert mission["dataset"] == "m-nz-cattle"
    assert len(mission["runs"]) == 1
    assert mission["runs"][0]["dataset"] == "m-nz-cattle"
    assert not any(r["dataset"] == "m-eurosat" for r in mission["runs"])


def test_snapshot_mission_field_matches_build_current_mission(server_env):
    """build_snapshot()'s "mission" key -- what the live dashboard and WS
    feed actually consume -- must be exactly build_current_mission()'s
    output, not a second, possibly-different computation of the same
    thing.
    """
    server, history, ledger = server_env

    history.start_session(dataset="m-nz-cattle", max_runs=1, run_minutes=1)
    ledger.add_run(dataset="m-nz-cattle", config={"encoder": "resnet34"}, val_metric=0.5,
                    task_type="segmentation", decision="pending")

    snap = server.build_snapshot()
    direct = server.build_current_mission()
    assert snap["mission"] is not None
    assert snap["mission"]["session_id"] == direct["session_id"]
    assert snap["mission"]["runs"] == direct["runs"]


def test_build_current_mission_never_breaks_the_snapshot_on_a_corrupt_session(server_env, monkeypatch):
    """build_current_mission() is wrapped in try/except on purpose --
    mission-history bookkeeping failing must never take down the live
    snapshot the rest of the dashboard depends on every second.
    """
    server, history, ledger = server_env

    def boom():
        raise RuntimeError("simulated history failure")

    monkeypatch.setattr(server.history_mod, "get_current_session", boom)

    assert server.build_current_mission() is None
    snap = server.build_snapshot()  # must not raise
    assert snap["mission"] is None
