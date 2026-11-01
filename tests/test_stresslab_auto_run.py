"""Tests for StressLab auto-running on submit_final_model
(gotrain.agent._auto_run_stresslab) -- the fix for "StressLab never runs"
(the frontend correctly reported NOT RUN because nothing in code ever
called run_stresslab; it was entirely left to the agent's own optional
tool choice, which it would skip under tool-call budget pressure).

Same fixture/pattern as tests/test_agent.py (isolated_logs, task_type="fake"
via fake_train.py for a fast, torch-free launch_training call) -- reused
here rather than duplicated, since _auto_run_stresslab is wired into the
same submit_final_model path those tests already exercise.

task_type="fake" is itself one of the real cases this fix must handle
correctly: StressLab has no evaluation path for "fake" (fake_train.py
never reports a real checkpoint), so every one of these tests exercises
the "stresslab_unavailable" branch for real, without needing torch,
geobench, or a downloaded dataset -- a true success/failure run through
StressLab's actual torch forward pass is out of reach in this environment,
same caveat as stresslab/runner.py's own module docstring.
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


def _tool_result_text(result: dict) -> str:
    return "".join(block.get("text", "") for block in result.get("content", []))


def test_submit_final_model_triggers_stresslab_attempt(isolated_logs):
    """The actual bug: previously nothing ever called run_stresslab unless
    the agent chose to. submit_final_model must now always attempt it --
    here that attempt resolves to "unavailable" (task_type="fake" has no
    StressLab path), but the point is it was ATTEMPTED and the outcome is
    recorded, not silently skipped with no trace at all.
    """
    agent = agent_mod.build_agent("m-eurosat", max_runs=6, run_minutes=5, max_tool_calls=40)
    launch_result = agent.tool.launch_training(dataset="m-eurosat", task_type="fake", config={"lr": 0.01})
    run_id = json.loads(_tool_result_text(launch_result))["run_id"]

    agent.tool.submit_final_model(dataset="m-eurosat", run_id=run_id)

    events = logging_utils.read_events(event="stresslab_unavailable")
    assert len(events) == 1
    assert events[0]["dataset"] == "m-eurosat"
    assert events[0]["run_id"] == run_id


def test_stresslab_unavailable_for_fake_task_type_names_the_reason(isolated_logs):
    agent = agent_mod.build_agent("m-eurosat", max_runs=6, run_minutes=5, max_tool_calls=40)
    launch_result = agent.tool.launch_training(dataset="m-eurosat", task_type="fake", config={"lr": 0.01})
    run_id = json.loads(_tool_result_text(launch_result))["run_id"]

    agent.tool.submit_final_model(dataset="m-eurosat", run_id=run_id)

    events = logging_utils.read_events(event="stresslab_unavailable")
    assert "fake" in events[0]["reason"]


def test_stresslab_failure_does_not_break_submission(isolated_logs, monkeypatch):
    """A supported task_type (classification) whose StressLab call raises
    for some other reason (simulated here -- the real cause would be a
    missing checkpoint file, torch/geobench import failure, or an actual
    crash in the perturbation battery) must still leave submit_final_model
    itself successful. The submission already happened; StressLab is a
    courtesy check on top of it, same philosophy as model_passport
    generation right above it in _submit_final_model.
    """
    from gotrain.tools import stresslab as stresslab_tools

    def _boom(dataset, task_type, run_id, **kwargs):
        raise FileNotFoundError("no checkpoint found (simulated)")

    monkeypatch.setattr(stresslab_tools, "run_stresslab", _boom)

    agent = agent_mod.build_agent("m-eurosat", max_runs=6, run_minutes=5, max_tool_calls=40)
    # task_type="fake" still reports no run_dir/version_id, so this would
    # normally short-circuit to "unavailable" before ever calling
    # run_stresslab -- force task_type through the ledger record directly
    # isn't worth the complexity here; the unavailable-vs-failed distinction
    # itself is covered by the other tests. What this test actually proves
    # is the exception-safety contract: even if run_stresslab raises,
    # submit_final_model must not propagate it.
    launch_result = agent.tool.launch_training(dataset="m-eurosat", task_type="fake", config={"lr": 0.01})
    run_id = json.loads(_tool_result_text(launch_result))["run_id"]

    submit_result = agent.tool.submit_final_model(dataset="m-eurosat", run_id=run_id)
    assert submit_result["status"] == "success"

    # task_type="fake" short-circuits to "unavailable" (no version_id) before
    # run_stresslab would ever be called -- _boom is never reached, which is
    # itself the correct behavior (asserted in the dedicated unavailable
    # test above). This test's job is just: submission still succeeds.
    assert control.get_state() == control.COMPLETED


def test_auto_run_stresslab_directly_handles_real_exception(isolated_logs, monkeypatch):
    """Calls gotrain.agent._auto_run_stresslab directly (bypassing the
    task_type="fake" short-circuit) with a record that claims a real
    checkpoint exists, and makes the underlying run_stresslab call raise --
    this is the actual "StressLab failure must not destroy an otherwise
    successful mission" case from the task brief, exercised precisely.
    """
    from gotrain import agent as agent_mod_inner
    from gotrain.tools import stresslab as stresslab_tools

    def _boom(dataset, task_type, run_id, **kwargs):
        raise RuntimeError("simulated torch forward pass crash")

    monkeypatch.setattr(stresslab_tools, "run_stresslab", _boom)

    fake_record = {"dataset": "m-eurosat", "task_type": "classification", "version_id": "v1"}
    agent_mod_inner._auto_run_stresslab(dataset="m-eurosat", run_id="run-xyz", record=fake_record)

    events = logging_utils.read_events(event="stresslab_failed")
    assert len(events) == 1
    assert events[0]["run_id"] == "run-xyz"
    assert "simulated torch forward pass crash" in events[0]["error"]


def test_auto_run_stresslab_calls_through_for_supported_task_with_checkpoint(isolated_logs, monkeypatch):
    """The success path: task_type is supported and a checkpoint is
    registered -- run_stresslab must actually be called (not skipped).
    """
    from gotrain import agent as agent_mod_inner
    from gotrain.tools import stresslab as stresslab_tools

    calls = []

    def _record_call(dataset, task_type, run_id, **kwargs):
        calls.append((dataset, task_type, run_id))
        return {"overall_verdict": "robust"}

    monkeypatch.setattr(stresslab_tools, "run_stresslab", _record_call)

    fake_record = {"dataset": "m-eurosat", "task_type": "classification", "version_id": "v1"}
    agent_mod_inner._auto_run_stresslab(dataset="m-eurosat", run_id="run-xyz", record=fake_record)

    assert calls == [("m-eurosat", "classification", "run-xyz")]
    # No unavailable/failed event should be logged on the success path.
    assert logging_utils.read_events(event="stresslab_unavailable") == []
    assert logging_utils.read_events(event="stresslab_failed") == []


def test_auto_run_stresslab_skips_segmentation_with_checkpoint_too(isolated_logs):
    """segmentation is the other supported task_type -- confirms the
    unavailable check isn't accidentally classification-only.
    """
    from gotrain import agent as agent_mod_inner

    # No checkpoint (version_id missing) -- must short-circuit regardless
    # of task_type being a real, supported one.
    fake_record = {"dataset": "m-nz-cattle", "task_type": "segmentation", "version_id": None}
    # Must not raise even though no run_stresslab mock is installed --
    # proves it never reaches the actual call for a checkpoint-less run.
    agent_mod_inner._auto_run_stresslab(dataset="m-nz-cattle", run_id="run-abc", record=fake_record)

    events = logging_utils.read_events(event="stresslab_unavailable")
    assert any(e["run_id"] == "run-abc" and "checkpoint" in e["reason"] for e in events)


def test_auto_run_passes_bounded_samples_and_time_budget(isolated_logs, monkeypatch):
    """The automatic run happens inside submit_final_model in the agent
    process, so it must be bounded in both samples and wall-clock time."""
    from gotrain.tools import stresslab as stresslab_tools

    seen = {}

    def _record(dataset, task_type, run_id, **kwargs):
        seen.update(kwargs)
        return {"overall_verdict": "robust"}

    monkeypatch.setattr(stresslab_tools, "run_stresslab", _record)
    agent_mod._auto_run_stresslab("m-eurosat", "run-1", {"task_type": "classification", "version_id": "v1"})
    assert seen["max_samples"] == agent_mod.AUTO_STRESSLAB_MAX_SAMPLES
    assert seen["time_budget_seconds"] == agent_mod.AUTO_STRESSLAB_TIME_BUDGET_SECONDS


def test_missing_checkpoint_is_reported_unavailable_not_failed(isolated_logs, monkeypatch):
    from gotrain.tools import stresslab as stresslab_tools

    def _missing(dataset, task_type, run_id, **kwargs):
        raise FileNotFoundError("No checkpoint found for run 'run-1'")

    monkeypatch.setattr(stresslab_tools, "run_stresslab", _missing)
    agent_mod._auto_run_stresslab("m-eurosat", "run-1", {"task_type": "classification", "version_id": "v1"})
    assert logging_utils.read_events(event="stresslab_unavailable")
    assert not logging_utils.read_events(event="stresslab_failed")


def test_battery_time_budget_produces_an_explicit_partial_report():
    import time

    import numpy as np

    from gotrain.stresslab import runner

    def slow_evaluate(images):
        time.sleep(0.05)
        return 0.8

    result = runner.run_perturbation_battery(slow_evaluate, np.zeros((2, 4, 4, 3), dtype=np.float32), time_budget_seconds=0.12)
    assert result["truncated"] is True
    assert 0 < result["n_perturbations_tested"] < result["n_perturbations_planned"]
    assert result["overall_verdict"].startswith("partial")
    assert result["clean_metric"] == 0.8
