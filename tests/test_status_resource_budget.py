"""Test for gotrain.status.set_resource_budget -- the new write path
resource-intelligence status relies on. Pure Python/JSON file I/O.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import logging_utils, status


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_utils, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(logging_utils, "STATUS_PATH", tmp_path / "status.json")
    return tmp_path


def test_set_resource_budget_round_trips(isolated_logs):
    budget = {"runs_used": 2, "max_runs": 5, "runs_remaining": 3}
    status.set_resource_budget(budget)
    assert status.get_status()["resource_budget"] == budget


def test_set_resource_budget_overwrites_the_whole_snapshot(isolated_logs):
    status.set_resource_budget({"runs_used": 1, "max_runs": 5})
    status.set_resource_budget({"runs_used": 2, "max_runs": 5})
    assert status.get_status()["resource_budget"]["runs_used"] == 2


def test_set_resource_budget_does_not_clobber_other_status_fields(isolated_logs):
    status.set_last_event("something happened")
    status.set_resource_budget({"runs_used": 1})
    current = status.get_status()
    assert current["last_event"] == "something happened"
    assert current["resource_budget"]["runs_used"] == 1
