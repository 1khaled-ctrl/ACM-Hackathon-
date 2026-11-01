"""Tests for the 3 new read-only audit endpoints in dashboard/api/server.py:
/api/lineage/{dataset}, /api/stresslab/{dataset}/{run_id},
/api/goaudit/{dataset}[/{generated_at}]. Each is a thin passthrough to an
existing backend module (gotrain.ledger.get_lineage_tree,
gotrain.stresslab.runner, gotrain.goaudit) -- these tests confirm the
passthrough is correct and that every "nothing here yet" case degrades to
a clean, honest response rather than an error, matching how the rest of
this FastAPI bridge behaves.

Calls the route functions directly (same pattern as
test_server_agent_alive.py) rather than spinning up a live HTTP server.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def server_env(tmp_path, monkeypatch):
    """Point logs/ AND experiment_planner's RUNS_DIR at the same temp
    tree, then import a fresh dashboard.api.server so every module it
    touches (ledger, history, experiment_planner, stresslab, goaudit,
    statistics) sees the isolated paths.
    """
    monkeypatch.setenv("GOTRAIN_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("GEO_BENCH_DIR", str(tmp_path / "geobench"))

    from gotrain import logging_utils

    importlib.reload(logging_utils)
    from gotrain import ledger

    importlib.reload(ledger)
    from gotrain import experiment_planner as planner

    importlib.reload(planner)
    monkeypatch.setattr(planner, "RUNS_DIR", tmp_path / "runs")
    from gotrain.stresslab import runner as stresslab_runner

    importlib.reload(stresslab_runner)
    from gotrain import goaudit

    importlib.reload(goaudit)
    from gotrain import statistics

    importlib.reload(statistics)
    from gotrain import history as history_mod

    importlib.reload(history_mod)

    sys.modules.pop("dashboard.api.server", None)
    from dashboard.api import server

    yield server, ledger, planner, stresslab_runner, goaudit

    sys.modules.pop("dashboard.api.server", None)
    monkeypatch.delenv("GOTRAIN_LOGS_DIR", raising=False)
    monkeypatch.delenv("GEO_BENCH_DIR", raising=False)
    importlib.reload(logging_utils)
    importlib.reload(ledger)
    importlib.reload(planner)
    importlib.reload(stresslab_runner)
    importlib.reload(goaudit)
    importlib.reload(statistics)
    importlib.reload(history_mod)


# ---------------------------------------------------------------------
# /api/lineage/{dataset}
# ---------------------------------------------------------------------

def test_lineage_endpoint_empty_dataset(server_env):
    server, _, _, _, _ = server_env
    result = server.api_lineage("m-eurosat")
    assert result == {"dataset": "m-eurosat", "runs": []}


def test_lineage_endpoint_never_claims_warm_start_without_metrics_file(server_env):
    """A run whose config literally sets warm_start_from_parent=true but
    has no metrics.json yet (the subprocess may still be running, or
    crashed before writing it) must report warm_start_requested=true and
    warm_started_from=None -- NOT a fabricated warm_started_from.
    """
    server, ledger, _, _, _ = server_env
    root = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.80,
                           task_type="classification", decision="kept")
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "warm_start_from_parent": True},
                    val_metric=0.85, parent_id=root["run_id"], task_type="classification", decision="pending")

    result = server.api_lineage("m-eurosat")
    child_node = next(n for n in result["runs"] if n["generation"] == 1)
    assert child_node["warm_start_requested"] is True
    assert child_node["warm_started_from"] is None


def test_lineage_endpoint_reports_confirmed_warm_start_from_metrics_file(server_env):
    """When metrics.json DOES record warm_started_from (the training
    template actually loaded the parent checkpoint), the endpoint must
    surface that real record, not just echo the requested flag.
    """
    server, ledger, planner, _, _ = server_env
    root = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.80,
                           task_type="classification", decision="kept")
    child = ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "warm_start_from_parent": True},
                            val_metric=0.87, parent_id=root["run_id"], task_type="classification", decision="kept")

    run_dir = planner.RUNS_DIR / "m-eurosat" / child["run_id"]
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps({
        "best": {"train_loss": 0.1},
        "warm_started_from": {"parent_run_id": root["run_id"], "parent_version": None, "checkpoint_path": "x"},
    }), encoding="utf-8")

    result = server.api_lineage("m-eurosat")
    child_node = next(n for n in result["runs"] if n["run_id"] == child["run_id"])
    assert child_node["warm_started_from"]["parent_run_id"] == root["run_id"]


# ---------------------------------------------------------------------
# /api/stresslab/{dataset}/{run_id}
# ---------------------------------------------------------------------

def test_stresslab_endpoint_not_run(server_env):
    server, _, _, _, _ = server_env
    result = server.api_stresslab("m-eurosat", "some-run-id")
    assert result == {"dataset": "m-eurosat", "run_id": "some-run-id", "not_run": True}


def test_stresslab_endpoint_returns_real_report(server_env):
    server, _, _, stresslab_runner, _ = server_env
    battery = {
        "clean_metric": 0.9,
        "results": [{"perturbation": "sensor_noise", "severity": "low", "clean_metric": 0.9,
                      "perturbed_metric": 0.88, "relative_drop": 0.022, "fragility": "robust"}],
        "n_perturbations_tested": 1, "n_fragile": 0, "overall_verdict": "robust",
    }
    stresslab_runner._record_and_log("m-eurosat", "run-1", "classification", battery)

    result = server.api_stresslab("m-eurosat", "run-1")
    assert result["not_run"] is False
    assert result["summary"]["overall_verdict"] == "robust"
    assert len(result["results"]) == 1


# ---------------------------------------------------------------------
# /api/goaudit/{dataset}[/{generated_at}]
# ---------------------------------------------------------------------

def test_goaudit_endpoint_not_available(server_env):
    server, _, _, _, _ = server_env
    result = server.api_goaudit_latest("m-eurosat")
    assert result == {"dataset": "m-eurosat", "available": False, "report": None, "history": []}


def test_goaudit_endpoint_returns_latest_report(server_env):
    server, ledger, _, _, goaudit = server_env
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.8,
                    task_type="classification", decision="kept")
    report = goaudit.generate_report("m-eurosat")
    goaudit.write_report("m-eurosat", report)

    result = server.api_goaudit_latest("m-eurosat")
    assert result["available"] is True
    assert result["report"]["progress"]["n_runs"] == 1
    assert len(result["history"]) == 1


def test_goaudit_historical_report_endpoint(server_env):
    server, ledger, _, _, goaudit = server_env
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.8,
                    task_type="classification", decision="kept")
    report = goaudit.generate_report("m-eurosat")
    goaudit.write_report("m-eurosat", report)
    timestamp = goaudit.list_reports("m-eurosat")[0]

    fetched = server.api_goaudit_report("m-eurosat", timestamp)
    assert fetched["progress"]["n_runs"] == 1


def test_goaudit_historical_report_404_for_unknown_timestamp(server_env):
    from fastapi import HTTPException

    server, _, _, _, _ = server_env
    with pytest.raises(HTTPException) as exc_info:
        server.api_goaudit_report("m-eurosat", "does-not-exist")
    assert exc_info.value.status_code == 404
