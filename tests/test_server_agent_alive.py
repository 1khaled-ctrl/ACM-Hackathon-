"""Tests for dashboard/api/server.py::agent_process_alive.

This is the check Start/Stop and the History reconciler all trust to
answer "is the agent subprocess actually still running". A pid alone is
not enough: run_gotrain.py never touches logs/agent.pid itself (only
api_agent_start writes it, only api_agent_stop removes it), so when a
mission finishes on its own -- the normal, successful path, agent exits
without anyone calling /api/agent/stop -- nothing cleans the file up.
psutil.pid_exists() alone would then report "running" forever
afterward, because the OS reuses pids: the next unrelated process that
happens to get that number makes a finished mission look permanently
stuck. This was hit for real (a completed mission's pid got reused,
"Start New Mission" returned 409 "already running" against an unrelated
live process). These tests reproduce that exact scenario with a real OS
process, not a mock, and confirm the fix self-heals the stale file.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def server_env(tmp_path, monkeypatch):
    """Point logs/ at a temp dir and get a fresh import of the server
    module so AGENT_PID_PATH resolves under it.
    """
    monkeypatch.setenv("GOTRAIN_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("GEO_BENCH_DIR", str(tmp_path / "geobench"))

    import importlib

    from gotrain import logging_utils

    importlib.reload(logging_utils)

    # dashboard.api.server reads LOGS_DIR at import time into
    # AGENT_PID_PATH -- must (re)import it AFTER the env var is set.
    sys.modules.pop("dashboard.api.server", None)
    from dashboard.api import server

    yield server

    sys.modules.pop("dashboard.api.server", None)
    monkeypatch.delenv("GOTRAIN_LOGS_DIR", raising=False)
    importlib.reload(logging_utils)


def test_no_pid_file_means_not_alive(server_env):
    server = server_env
    assert not server.AGENT_PID_PATH.exists()
    assert server.agent_process_alive() is False


def test_garbage_pid_file_means_not_alive(server_env):
    server = server_env
    server.AGENT_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    server.AGENT_PID_PATH.write_text("not-a-number", encoding="utf-8")
    assert server.agent_process_alive() is False


def test_pid_that_does_not_exist_is_cleaned_up(server_env):
    """A pid for a process that has already exited: not alive, and the
    stale file must be removed so it doesn't get re-checked forever.
    """
    server = server_env
    server.AGENT_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    # A pid essentially guaranteed not to exist.
    server.AGENT_PID_PATH.write_text("999999999", encoding="utf-8")

    assert server.agent_process_alive() is False
    assert not server.AGENT_PID_PATH.exists()


def test_a_real_but_unrelated_live_process_is_rejected_and_self_heals(server_env):
    """THE regression test: a pid that is genuinely alive right now, but
    is not a run_gotrain.py process -- e.g. the OS reused the number
    after the real agent exited. Must report not-alive AND remove the
    stale file, rather than reporting "running" against the wrong
    process forever. Uses this test's own interpreter process as the
    "real but unrelated" live process -- no mocking needed.
    """
    server = server_env
    own_pid = os.getpid()  # genuinely alive: it's pytest itself

    server.AGENT_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    server.AGENT_PID_PATH.write_text(str(own_pid), encoding="utf-8")

    assert server.agent_process_alive() is False
    assert not server.AGENT_PID_PATH.exists()


def test_a_process_whose_cmdline_names_run_gotrain_is_accepted(server_env, monkeypatch):
    """The positive case, to confirm the fix doesn't just always return
    False: a pid that IS alive and DOES look like run_gotrain.py must be
    reported as running, so a genuinely in-progress mission still blocks
    a second Start.

    agent_process_alive() does `import psutil` locally rather than at
    module scope (same lazy-import pattern used throughout server.py),
    so the patch target is the psutil module in sys.modules, not an
    attribute on the server module.
    """
    import psutil

    server = server_env
    own_pid = os.getpid()

    class FakeProc:
        def cmdline(self):
            return ["python", "run_gotrain.py", "--dataset", "m-eurosat"]

    monkeypatch.setattr(psutil, "Process", lambda pid: FakeProc())

    server.AGENT_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    server.AGENT_PID_PATH.write_text(str(own_pid), encoding="utf-8")

    assert server.agent_process_alive() is True
    # A genuinely running agent's pid file must NOT be touched.
    assert server.AGENT_PID_PATH.exists()
