"""Tests for the GoTrain agent: hard limits enforced via hooks, and the
fact that final_test_eval is never exposed as an agent tool. No test here
constructs a live Bedrock call — build_agent() only constructs a
BedrockModel client object (no network call) and agent.tool.<name>(...)
invokes a tool directly through the hook pipeline, bypassing the LLM.
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
from gotrain import control, ledger, logging_utils, status
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

    return logs_dir


def _tool_result_text(result: dict) -> str:
    return "".join(block.get("text", "") for block in result.get("content", []))


def test_agent_tool_list_excludes_final_test_eval(isolated_logs):
    agent = agent_mod.build_agent("m-eurosat", max_runs=6, run_minutes=5, max_tool_calls=40)
    assert "final_test_eval" not in agent.tool_names
    # sanity: the rest of the expected tools ARE present
    for name in [
        "list_datasets", "get_dataset_info", "run_geoguard", "launch_training", "get_run_status",
        "get_run_metrics", "query_history", "compare_runs", "record_decision",
        "submit_final_model", "upload_to_s3", "download_from_s3",
    ]:
        assert name in agent.tool_names


def test_all_tools_module_excludes_final_test_eval():
    assert "final_test_eval" not in [t.tool_name for t in agent_mod.ALL_TOOLS]


def test_system_prompt_is_cp1252_safe():
    """run_gotrain.py --dry-run prints the system prompt via plain print(),
    which on Windows defaults to the cp1252 console codepage — a character
    outside that codepage (e.g. a Unicode arrow) crashes with
    UnicodeEncodeError instead of printing. Caught this exact bug once
    already (an arrow in the "suggested strategy" section); this guards
    against it coming back.
    """
    prompt = agent_mod.load_system_prompt()
    try:
        prompt.encode("cp1252")
    except UnicodeEncodeError as e:
        raise AssertionError(
            f"system_prompt.md contains a character not safe for Windows' "
            f"default console codepage (cp1252): {e}"
        )


def test_system_prompt_mentions_dataset_search_space():
    """Block 4: the prompt must name the valid config values and the
    maximize-validation-score goal for the agent to plan with."""
    prompt = agent_mod.load_system_prompt()
    assert "maximize" in prompt.lower()
    for dataset in ["m-eurosat", "m-so2sat", "m-pv4ger-seg", "m-nz-cattle"]:
        assert dataset in prompt
    assert "do not select bands by name on m-eurosat" in prompt.lower()


def test_max_tool_calls_enforced(isolated_logs):
    agent = agent_mod.build_agent("m-eurosat", max_runs=6, run_minutes=5, max_tool_calls=2)

    r1 = agent.tool.list_datasets()
    r2 = agent.tool.list_datasets()
    r3 = agent.tool.list_datasets()

    assert r1["status"] == "success"
    assert r2["status"] == "success"
    assert r3["status"] == "error"
    assert "budget exhausted" in _tool_result_text(r3).lower()


def test_max_runs_enforced(isolated_logs):
    agent = agent_mod.build_agent("m-eurosat", max_runs=1, run_minutes=5, max_tool_calls=40)

    r1 = agent.tool.launch_training(dataset="m-eurosat", task_type="fake", config={"lr": 0.01})
    assert r1["status"] == "success"

    r2 = agent.tool.launch_training(dataset="m-eurosat", task_type="fake", config={"lr": 0.001})
    assert r2["status"] == "error"
    assert "run budget exhausted" in _tool_result_text(r2).lower()

    # only one run should have actually been recorded
    history = agent.tool.query_history(dataset="m-eurosat")
    history_data = json.loads(_tool_result_text(history))
    assert len(history_data) == 1


def test_max_runs_does_not_block_other_tools(isolated_logs):
    """Hitting the run limit should only block launch_training, not every tool."""
    agent = agent_mod.build_agent("m-eurosat", max_runs=1, run_minutes=5, max_tool_calls=40)

    agent.tool.launch_training(dataset="m-eurosat", task_type="fake", config={"lr": 0.01})
    agent.tool.launch_training(dataset="m-eurosat", task_type="fake", config={"lr": 0.001})

    r = agent.tool.list_datasets()
    assert r["status"] == "success"


def test_tool_calls_are_logged(isolated_logs):
    agent = agent_mod.build_agent("m-eurosat", max_runs=6, run_minutes=5, max_tool_calls=40)
    agent.tool.list_datasets()

    events = [
        json.loads(line)
        for line in (isolated_logs / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tool_call_events = [e for e in events if e["event"] == "tool_call"]
    assert len(tool_call_events) == 1
    assert tool_call_events[0]["tool"] == "list_datasets"
    assert tool_call_events[0]["status"] == "success"
    assert "duration_seconds" in tool_call_events[0]


def test_submit_final_model_does_not_touch_test_split(isolated_logs):
    agent = agent_mod.build_agent("m-eurosat", max_runs=6, run_minutes=5, max_tool_calls=40)

    launch_result = agent.tool.launch_training(
        dataset="m-eurosat", task_type="fake", config={"lr": 0.01}
    )
    run_id = json.loads(_tool_result_text(launch_result))["run_id"]

    submit_result = agent.tool.submit_final_model(dataset="m-eurosat", run_id=run_id)
    assert submit_result["status"] == "success"

    # no test_lock file should exist — submit_final_model must never evaluate
    assert not list(isolated_logs.glob("test_lock_*.json"))


def test_submit_final_model_rejects_mismatched_dataset(isolated_logs):
    agent = agent_mod.build_agent("m-eurosat", max_runs=6, run_minutes=5, max_tool_calls=40)

    launch_result = agent.tool.launch_training(
        dataset="m-eurosat", task_type="fake", config={"lr": 0.01}
    )
    run_id = json.loads(_tool_result_text(launch_result))["run_id"]

    result = agent.tool.submit_final_model(dataset="m-so2sat", run_id=run_id)
    assert result["status"] == "error"


def test_bad_config_rejection_does_not_count_against_max_runs(isolated_logs):
    """A launch rejected for an unsupported config value (a nonsense
    optimizer name the trainer doesn't support) must not consume one of the
    agent's max_runs — only a launch that actually attempts training should.
    """
    agent = agent_mod.build_agent("m-eurosat", max_runs=1, run_minutes=5, max_tool_calls=40)

    rejected = agent.tool.launch_training(
        dataset="m-eurosat", task_type="classification", config={"optimizer": "not-a-real-optimizer"}
    )
    assert rejected["status"] == "error"
    rejected_text = _tool_result_text(rejected)
    assert "not-a-real-optimizer" in rejected_text
    # the error must name the valid values, not just say "invalid"
    assert "adam" in rejected_text and "sgd" in rejected_text

    # max_runs=1 and none consumed yet -> this real run must still succeed
    real_run = agent.tool.launch_training(dataset="m-eurosat", task_type="fake", config={})
    assert real_run["status"] == "success"

    # now the one real run budget is spent -> a second real attempt is blocked
    blocked = agent.tool.launch_training(dataset="m-eurosat", task_type="fake", config={"seed": 1})
    assert blocked["status"] == "error"
    assert "run budget exhausted" in _tool_result_text(blocked).lower()

    # exactly one ledger record should exist (the rejected call never ran)
    history = agent.tool.query_history(dataset="m-eurosat")
    assert len(json.loads(_tool_result_text(history))) == 1


def test_missing_hypothesis_rejection_does_not_count_against_max_runs(isolated_logs):
    """A launch_training call with a parent_id but no hypothesis must be
    rejected before any subprocess runs, and — like a bad config value —
    must not consume one of the agent's max_runs.
    """
    agent = agent_mod.build_agent("m-eurosat", max_runs=1, run_minutes=5, max_tool_calls=40)

    first = agent.tool.launch_training(dataset="m-eurosat", task_type="fake", config={"lr": 0.01})
    run_id = json.loads(_tool_result_text(first))["run_id"]

    rejected = agent.tool.launch_training(
        dataset="m-eurosat", task_type="fake", config={"lr": 0.001}, parent_id=run_id
    )
    assert rejected["status"] == "error"
    assert "hypothesis" in _tool_result_text(rejected).lower()

    # max_runs=1 was already spent by the first (real) run -- the budget
    # message should fire for a THIRD call, proving the rejected second
    # call above didn't silently consume the budget itself.
    third = agent.tool.launch_training(
        dataset="m-eurosat", task_type="fake", config={"lr": 0.002}, parent_id=run_id,
        hypothesis={"change": "lowered lr again", "why": "testing", "expected_direction": "increase"},
    )
    assert third["status"] == "error"
    assert "run budget exhausted" in _tool_result_text(third).lower()

    history = agent.tool.query_history(dataset="m-eurosat")
    assert len(json.loads(_tool_result_text(history))) == 1


def test_control_plane_blocks_every_tool_call_while_paused(isolated_logs):
    """Pausing (gotrain.control) must cancel EVERY tool call, not just
    launch_training -- a paused agent should not be able to do anything
    at all, including read-only exploration. The cancellation must not
    cost tool-call budget either (it isn't the agent's doing).
    """
    agent = agent_mod.build_agent("m-eurosat", max_runs=6, run_minutes=5, max_tool_calls=2)
    control.transition(control.PAUSE_REQUESTED, requested_by="engineer")

    blocked = agent.tool.list_datasets()
    assert blocked["status"] == "error"
    assert "engineer-controlled" in _tool_result_text(blocked).lower()

    # The agent should have acknowledged the pause the moment its first
    # call was blocked.
    assert control.get_state() == control.PAUSED_FOR_ENGINEER

    # Being blocked must not have consumed any of the tiny max_tool_calls=2
    # budget -- a second and third blocked call must behave identically,
    # not suddenly report budget exhaustion instead.
    for _ in range(3):
        blocked_again = agent.tool.list_datasets()
        assert blocked_again["status"] == "error"
        assert "engineer-controlled" in _tool_result_text(blocked_again).lower()


def test_control_plane_resumes_and_tool_calls_work_again(isolated_logs):
    agent = agent_mod.build_agent("m-eurosat", max_runs=6, run_minutes=5, max_tool_calls=40)
    control.transition(control.PAUSE_REQUESTED, requested_by="engineer")
    agent.tool.list_datasets()  # acknowledges the pause
    assert control.get_state() == control.PAUSED_FOR_ENGINEER

    control.transition(control.RESUME_REQUESTED, requested_by="engineer")
    control.transition(control.RUNNING_AUTONOMOUSLY, requested_by="engineer")

    result = agent.tool.list_datasets()
    assert result["status"] == "success"


def test_manual_control_also_blocks_tool_calls(isolated_logs):
    agent = agent_mod.build_agent("m-eurosat", max_runs=6, run_minutes=5, max_tool_calls=40)
    control.transition(control.MANUAL_CONTROL, requested_by="engineer")

    blocked = agent.tool.launch_training(dataset="m-eurosat", task_type="fake", config={})
    assert blocked["status"] == "error"
    assert "manual_control" in _tool_result_text(blocked).lower()


def test_submit_final_model_transitions_control_to_completed(isolated_logs):
    agent = agent_mod.build_agent("m-eurosat", max_runs=6, run_minutes=5, max_tool_calls=40)

    launch_result = agent.tool.launch_training(dataset="m-eurosat", task_type="fake", config={"lr": 0.01})
    run_id = json.loads(_tool_result_text(launch_result))["run_id"]

    agent.tool.submit_final_model(dataset="m-eurosat", run_id=run_id)
    assert control.get_state() == control.COMPLETED

    # Nothing further should be permitted once completed.
    blocked = agent.tool.list_datasets()
    assert blocked["status"] == "error"


def test_final_test_eval_lock_is_independent_per_dataset(isolated_logs):
    """All 4 GEO-Bench datasets must each be evaluable exactly once,
    independently — locking one dataset's test split must not affect any
    other dataset.
    """
    datasets = ["m-eurosat", "m-so2sat", "m-pv4ger-seg", "m-nz-cattle"]

    for ds in datasets:
        record = final_eval.final_test_eval(ds, f"run-{ds}", lambda: {"metric": 1.0})
        assert record["task"] == ds

    for ds in datasets:
        with pytest.raises(final_eval.TestAlreadyEvaluatedError):
            final_eval.final_test_eval(ds, f"run-{ds}-again", lambda: {"metric": 0.0})

    lock_files = sorted(p.name for p in isolated_logs.glob("test_lock_*.json"))
    assert lock_files == sorted(f"test_lock_{ds}.json" for ds in datasets)
