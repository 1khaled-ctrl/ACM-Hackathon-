"""Mission-integrity regressions found in the pre-submission audit.

1. submit_final_model accepted an OLD historical run after this mission's
   only launch_training attempt had timed out or crashed. The guard checked
   hooks.run_count, which counts every non-preflight attempt (including
   failures), so a mission with zero usable runs could still end COMPLETED --
   the exact "m-pv4ger-seg: COMPLETED, 0/3, no best val, 1 timed out" state
   seen in History.
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import agent as agent_mod
from gotrain import approval, control, ledger, logging_utils, model_passport
from gotrain.dataiq import manifest as dataiq_manifest
from gotrain.tools import final_eval


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setattr(logging_utils, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", logs_dir / "events.jsonl")
    monkeypatch.setattr(logging_utils, "STATUS_PATH", logs_dir / "status.json")
    monkeypatch.setattr(ledger, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(ledger, "LEDGER_PATH", logs_dir / "ledger.jsonl")
    monkeypatch.setattr(final_eval, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(control, "CONTROL_PATH", logs_dir / "control.json")
    monkeypatch.setattr(model_passport, "EVENTS_PATH", logs_dir / "events.jsonl")
    monkeypatch.setattr(model_passport, "PASSPORTS_DIR", logs_dir / "model_passports")
    monkeypatch.setattr(approval, "APPROVALS_STATE_PATH", logs_dir / "approvals_state.json")
    monkeypatch.setattr(approval, "APPROVALS_LOG_PATH", logs_dir / "approvals.jsonl")
    monkeypatch.setattr(dataiq_manifest, "MANIFESTS_DIR", logs_dir / "dataiq" / "manifests")
    return logs_dir


def _text(result: dict) -> str:
    return "".join(block.get("text", "") for block in result.get("content", []))


def _old_run():
    return ledger.add_run(
        dataset="m-pv4ger-seg", config={"encoder": "resnet34"}, val_metric=0.81,
        task_type="segmentation", decision="kept",
    )


@pytest.mark.parametrize("failure", [
    TimeoutError("Training for 'm-pv4ger-seg' exceeded its 300s time budget and was killed."),
    RuntimeError("Training failed for m-pv4ger-seg (exit code -9): "),
])
def test_failed_attempt_does_not_unlock_submitting_an_old_run(isolated_logs, monkeypatch, failure):
    old = _old_run()
    agent = agent_mod.build_agent("m-pv4ger-seg", max_runs=3, run_minutes=5, max_tool_calls=40)

    def _fail(**kwargs):
        raise failure

    monkeypatch.setattr(agent_mod.training_tools, "launch_training", _fail)
    launch = agent.tool.launch_training(dataset="m-pv4ger-seg", task_type="segmentation", config={})
    assert launch["status"] == "error"

    result = agent.tool.submit_final_model(dataset="m-pv4ger-seg", run_id=old["run_id"])
    assert result["status"] == "error"
    assert "0 new training runs" in _text(result)
    assert control.get_state() != control.COMPLETED
    assert not [r for r in ledger.get_history("m-pv4ger-seg") if r.get("submitted_final")]


def test_failed_attempt_still_consumes_run_budget(isolated_logs, monkeypatch):
    """The fix must not make failures free: a timed-out attempt really
    spent GPU time, so it still counts against max_runs."""
    agent = agent_mod.build_agent("m-pv4ger-seg", max_runs=1, run_minutes=5, max_tool_calls=40)

    def _fail(**kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(agent_mod.training_tools, "launch_training", _fail)
    agent.tool.launch_training(dataset="m-pv4ger-seg", task_type="segmentation", config={})
    second = agent.tool.launch_training(dataset="m-pv4ger-seg", task_type="segmentation", config={})
    assert "Run budget exhausted" in _text(second)


def test_success_after_failure_unlocks_submission(isolated_logs, monkeypatch):
    agent = agent_mod.build_agent("m-eurosat", max_runs=3, run_minutes=5, max_tool_calls=40)
    real_launch = agent_mod.training_tools.launch_training
    calls = {"n": 0}

    def _first_fails(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("timed out")
        return real_launch(**kwargs)

    monkeypatch.setattr(agent_mod.training_tools, "launch_training", _first_fails)
    agent.tool.launch_training(dataset="m-eurosat", task_type="fake", config={"lr": 0.01})
    ok = agent.tool.launch_training(
        dataset="m-eurosat", task_type="fake", config={"lr": 0.02},
        hypothesis={"change": "retry with a lower-cost config", "why": "first attempt timed out"},
    )
    run_id = json.loads(_text(ok))["run_id"]

    result = agent.tool.submit_final_model(dataset="m-eurosat", run_id=run_id)
    assert result["status"] == "success"
    assert control.get_state() == control.COMPLETED


def test_second_run_without_parent_still_requires_a_hypothesis(isolated_logs):
    """The prompt promises 'every run after the first requires a hypothesis,
    enforced in code' -- but the check only fired when parent_id was given,
    so omitting parent_id skipped it. Rejected pre-flight, at no run cost."""
    agent = agent_mod.build_agent("m-eurosat", max_runs=3, run_minutes=5, max_tool_calls=40)
    first = agent.tool.launch_training(dataset="m-eurosat", task_type="fake", config={"lr": 0.01})
    assert first["status"] == "success"

    second = agent.tool.launch_training(dataset="m-eurosat", task_type="fake", config={"lr": 0.02})
    assert second["status"] == "error"
    assert "hypothesis" in _text(second).lower()

    third = agent.tool.launch_training(
        dataset="m-eurosat", task_type="fake", config={"lr": 0.02},
        hypothesis={"change": "doubled lr", "why": "loss plateaued early"},
    )
    assert third["status"] == "success"  # the rejected attempt did not consume run budget
