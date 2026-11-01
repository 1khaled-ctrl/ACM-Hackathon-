"""Two Start requests at once must launch exactly one agent.

api_agent_start checked agent_process_alive(), listed datasets, spawned the
subprocess and only then wrote logs/agent.pid -- not atomic. FastAPI runs
sync endpoints in a thread pool, so two near-simultaneous Start clicks could
both pass the check and launch two agents racing on the same ledger.
"""

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("GOTRAIN_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("GEO_BENCH_DIR", str(tmp_path / "geobench"))
    import importlib

    from gotrain import history, ledger, logging_utils

    importlib.reload(logging_utils)
    importlib.reload(ledger)
    importlib.reload(history)
    sys.modules.pop("dashboard.api.server", None)
    # import_module, not `from dashboard.api import server`: the latter returns
    # the stale module cached as an attribute on the dashboard.api package.
    srv = importlib.import_module("dashboard.api.server")

    yield srv
    sys.modules.pop("dashboard.api.server", None)
    monkeypatch.delenv("GOTRAIN_LOGS_DIR", raising=False)
    monkeypatch.delenv("GEO_BENCH_DIR", raising=False)
    importlib.reload(logging_utils)
    importlib.reload(ledger)
    importlib.reload(history)


def test_two_simultaneous_starts_launch_one_agent(server, monkeypatch):
    from fastapi import HTTPException

    from gotrain.tools import datasets as datasets_tools

    spawned = []

    def slow_popen(cmd, **kwargs):
        time.sleep(0.3)  # the window between the alive-check and writing agent.pid
        spawned.append(cmd)
        return SimpleNamespace(pid=40000 + len(spawned))

    monkeypatch.setattr(datasets_tools, "list_datasets", lambda: ["m-eurosat"])
    monkeypatch.setattr(server.subprocess, "Popen", slow_popen)
    monkeypatch.setattr(server, "agent_process_alive", lambda: server.AGENT_PID_PATH.exists())

    results = []

    def start():
        try:
            server.api_agent_start(server.StartRequest(dataset="m-eurosat", max_runs=1, run_minutes=1, max_tool_calls=5))
            results.append("started")
        except HTTPException as e:
            results.append(e.status_code)
        except Exception as e:  # surfaced in the assertion message, not lost in a thread
            results.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=start) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(spawned) == 1, f"spawned={len(spawned)} results={results}"
    assert sorted(results, key=str) == [409, "started"]
