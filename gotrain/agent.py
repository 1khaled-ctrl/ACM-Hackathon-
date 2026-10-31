"""The GoTrain agent: a Strands Agent wired to the GEO-Bench training tools,
with hard limits enforced in code (not just in the prompt) and every tool
call logged to logs/events.jsonl + reflected in logs/status.json.

Confirmed against the installed strands-agents==1.63.0 API before writing
this module:
  - from strands import Agent, tool
  - from strands.models import BedrockModel(model_id=..., region_name=...)
  - from strands.hooks import HookProvider, HookRegistry, BeforeToolCallEvent, AfterToolCallEvent
    BeforeToolCallEvent.cancel_tool: bool | str — setting it to a string
    cancels the call and returns that string to the model as the result,
    which is how the hard limits below are enforced without trusting the
    model to respect the prompt.

The agent's tool list intentionally does NOT include final_test_eval. The
agent may only ever see validation metrics; the test split is evaluated
once, by a human, via run_final_eval.py (see gotrain/tools/final_eval.py).
"""

import json
from pathlib import Path
from typing import Any, Optional

from strands import Agent, tool
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent, HookRegistry

from gotrain import config, ledger, status
from gotrain.config_schema import ConfigValidationError
from gotrain.logging_utils import log_event
from gotrain.tools import datasets as datasets_tools
from gotrain.tools import s3 as s3_tools
from gotrain.tools import training as training_tools

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_PATH = REPO_ROOT / "prompts" / "system_prompt.md"

MAX_RESULT_LOG_CHARS = 1000


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tool wrappers. Each wraps an existing gotrain.tools function with @tool so
# the model can call it; docstrings here are what the model reads, so they
# describe the contract (inputs, what's returned, what split is used).
# ---------------------------------------------------------------------------


@tool
def list_datasets() -> list:
    """List the GEO-Bench dataset names available locally.

    Returns a list of dataset name strings (e.g. "m-eurosat").
    """
    return datasets_tools.list_datasets()


@tool
def get_dataset_info(dataset: str) -> dict:
    """Get metadata about a dataset: split sizes, band names, image shape,
    and label/class info. Returns metadata only — never raw pixel data or
    labels. The test split's size is reported as "locked", not a number,
    because it is never opened.

    Args:
        dataset: the GEO-Bench dataset name, e.g. "m-eurosat".
    """
    return datasets_tools.get_dataset_info(dataset)


@tool
def launch_training(
    dataset: str,
    task_type: str,
    config: dict,
    parent_id: Optional[str] = None,
) -> dict:
    """Launch one training run and evaluate it on the VALIDATION split only.

    Trains on the dataset's train split, evaluates on validation, and
    records the run in the ledger with its config, validation metric, and
    (if parent_id is given) the diff vs. the parent run's config. Never
    touches the test split.

    Args:
        dataset: the GEO-Bench dataset name, e.g. "m-eurosat".
        task_type: "classification" or "segmentation" — selects which
            training template to run.
        config: a dict of hyperparameters for this run (e.g. learning
            rate, epochs, batch size, backbone). Change exactly one field
            from the parent run's config.
        parent_id: the run_id this run is derived from, if any. Omit only
            for the very first run on this dataset.

    Returns:
        The ledger record for this run: run_id, val_metric,
        delta_vs_parent, config_diff, and more.
    """
    raise NotImplementedError  # replaced per-agent in build_agent()


@tool
def get_run_status(run_id: str) -> dict:
    """Get the full ledger record for one run: config, val_metric, decision,
    rationale, parent_id, and more.

    Args:
        run_id: the run to look up.
    """
    return training_tools.get_run_status(run_id)


@tool
def get_run_metrics(run_id: str) -> dict:
    """Get just the metrics for one run: val_metric, delta_vs_parent,
    split_used (always "validation"), gpu_minutes.

    Args:
        run_id: the run to look up.
    """
    return training_tools.get_run_metrics(run_id)


@tool
def query_history(dataset: Optional[str] = None) -> list:
    """List all past runs, optionally filtered to one dataset. Call this
    BEFORE every launch_training call so you never repeat a config that's
    already been tried.

    Args:
        dataset: if given, only return runs for this dataset.
    """
    return training_tools.query_history(dataset)


@tool
def compare_runs(run_id_a: str, run_id_b: str) -> dict:
    """Compare two runs: their config differences and the validation metric
    delta between them.

    Args:
        run_id_a: the first run to compare.
        run_id_b: the second run to compare.
    """
    return training_tools.compare_runs(run_id_a, run_id_b)


@tool
def record_decision(run_id: str, decision: str, rationale: str) -> dict:
    """Record your decision about a run: "kept" (it's your new best or a
    useful data point), "discarded" (clearly worse), or "pending".

    Args:
        run_id: the run this decision applies to.
        decision: one of "kept", "discarded", "pending".
        rationale: one sentence explaining why, in terms of the validation
            metric and what changed vs. its parent.
    """
    return training_tools.record_decision(run_id, decision, rationale)


@tool
def submit_final_model(dataset: str, run_id: str) -> dict:
    """Submit your final chosen model for this dataset. This only RECORDS
    your choice in the ledger — it does not evaluate anything and does not
    touch the test split. Call this once, when you are done iterating, with
    the run_id of your best validation run. A human will run the one-time
    test evaluation separately afterward.

    Args:
        dataset: the dataset this submission is for.
        run_id: the run_id of the model you are submitting as final.
    """
    record = training_tools.get_run_status(run_id)
    if record["dataset"] != dataset:
        raise ValueError(
            f"run_id {run_id!r} belongs to dataset {record['dataset']!r}, not {dataset!r}"
        )
    updated = ledger.update_run(run_id, decision="kept", submitted_final=True)
    status.write_status(stage="FINALIZE", action=f"submitted {run_id} as final model", where=dataset)
    log_event("final_model_submitted", dataset=dataset, run_id=run_id)
    return updated


@tool
def upload_to_s3(dataset: str, version: str) -> dict:
    """Upload a registered model version's files to S3, if an S3 bucket is
    configured for this environment. Returns {"status": "skipped", ...} if
    no bucket is configured (e.g. running locally without AWS access) —
    this is not an error, just means there's nowhere to upload to right now.

    Args:
        dataset: the dataset the model version belongs to.
        version: the version id (e.g. "v1") from a run's ledger record.
    """
    return s3_tools.upload_to_s3(dataset, version)


@tool
def download_from_s3(dataset: str, version: str) -> dict:
    """Download a model version's files from S3 into the local model
    registry, if an S3 bucket is configured. Returns {"status": "skipped",
    ...} if no bucket is configured.

    Args:
        dataset: the dataset the model version belongs to.
        version: the version id (e.g. "v1") to download.
    """
    return s3_tools.download_from_s3(dataset, version)


ALL_TOOLS = [
    list_datasets,
    get_dataset_info,
    launch_training,
    get_run_status,
    get_run_metrics,
    query_history,
    compare_runs,
    record_decision,
    submit_final_model,
    upload_to_s3,
    download_from_s3,
]


# ---------------------------------------------------------------------------
# Hard limits + logging, enforced via hooks (not just the system prompt).
# ---------------------------------------------------------------------------


class LimitsAndLoggingHooks:
    """Enforces max runs / tool calls and logs every tool call.

    Registered on the Agent via `hooks=[...]`. BeforeToolCallEvent fires
    before a tool executes; setting `event.cancel_tool` to a string cancels
    the call and sends that string back to the model as the tool's result,
    instead of running it. AfterToolCallEvent fires once a tool (that was
    allowed to run) has finished, with its result/duration/exception.
    """

    def __init__(self, dataset: str, max_runs: int, run_minutes: float, max_tool_calls: int):
        self.dataset = dataset
        self.max_runs = max_runs
        self.run_seconds = run_minutes * 60
        self.max_tool_calls = max_tool_calls
        self.tool_call_count = 0
        self.run_count = 0

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._on_before)
        registry.add_callback(AfterToolCallEvent, self._on_after)

    def _on_before(self, event: BeforeToolCallEvent) -> None:
        self.tool_call_count += 1
        tool_name = event.tool_use["name"]

        if self.tool_call_count > self.max_tool_calls:
            event.cancel_tool = (
                f"Tool-call budget exhausted ({self.max_tool_calls} calls used). "
                "Stop exploring: call query_history to find your best run, then "
                "call submit_final_model with that run's run_id."
            )
            return

        if tool_name == "launch_training" and self.run_count >= self.max_runs:
            event.cancel_tool = (
                f"Run budget exhausted ({self.max_runs} runs used for {self.dataset}). "
                "Do not launch another run. Call query_history to find your best "
                "run, then call submit_final_model with that run's run_id."
            )
            return

        if tool_name == "launch_training":
            # Run counting happens in _on_after, once we know whether the
            # call actually attempted training or was rejected pre-flight
            # for a bad config value (ConfigValidationError) — a rejected
            # launch must not cost the agent one of its max_runs.
            event.tool_use["input"]["timeout_seconds"] = self.run_seconds

    def _on_after(self, event: AfterToolCallEvent) -> None:
        tool_name = event.tool_use["name"]
        tool_input = event.tool_use.get("input", {})

        if tool_name == "launch_training" and not isinstance(event.exception, ConfigValidationError):
            self.run_count += 1

        result_text = ""
        if event.result is not None:
            for block in event.result.get("content", []):
                if "text" in block:
                    result_text += block["text"]
                elif "json" in block:
                    result_text += json.dumps(block["json"])
        truncated = result_text[:MAX_RESULT_LOG_CHARS]

        log_event(
            "tool_call",
            tool=tool_name,
            arguments=tool_input,
            result_truncated=truncated,
            status=event.result.get("status") if event.result else "error",
            duration_seconds=event.duration,
            exception=str(event.exception) if event.exception else None,
            counted_against_run_budget=(
                tool_name == "launch_training" and not isinstance(event.exception, ConfigValidationError)
            ),
        )
        status.write_status(last_event=f"called {tool_name}")


# ---------------------------------------------------------------------------
# Agent construction.
# ---------------------------------------------------------------------------


def build_agent(
    dataset: str,
    max_runs: int = config.DEFAULT_MAX_RUNS,
    run_minutes: float = config.DEFAULT_RUN_MINUTES,
    max_tool_calls: int = config.DEFAULT_MAX_TOOL_CALLS,
) -> Agent:
    """Build a Strands Agent wired to the GoTrain tools for one dataset.

    Hard limits (max runs, per-run time budget, max tool calls) are
    enforced by LimitsAndLoggingHooks, not left to the system prompt alone.
    The agent's tools never include final_test_eval — the test split is
    only ever touched by a human, via run_final_eval.py.
    """
    from strands.models import BedrockModel

    model = BedrockModel(model_id=config.BEDROCK_MODEL_ID, region_name=config.BEDROCK_REGION)

    hooks = LimitsAndLoggingHooks(
        dataset=dataset,
        max_runs=max_runs,
        run_minutes=run_minutes,
        max_tool_calls=max_tool_calls,
    )

    # launch_training's real implementation (not the @tool stub above) is
    # bound here so the per-call timeout_seconds injected by the hook reaches
    # gotrain.tools.training.launch_training.
    @tool(name="launch_training")
    def _launch_training(
        dataset: str,
        task_type: str,
        config: dict,
        parent_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> dict:
        """Launch one training run and evaluate it on the VALIDATION split only.

        Trains on the dataset's train split, evaluates on validation, and
        records the run in the ledger with its config, validation metric,
        and (if parent_id is given) the diff vs. the parent run's config.
        Never touches the test split.

        Args:
            dataset: the GEO-Bench dataset name, e.g. "m-eurosat".
            task_type: "classification" or "segmentation" — selects which
                training template to run.
            config: a dict of hyperparameters for this run. Change exactly
                one field from the parent run's config.
            parent_id: the run_id this run is derived from, if any. Omit
                only for the very first run on this dataset.
        """
        return training_tools.launch_training(
            dataset=dataset,
            task_type=task_type,
            config=config,
            parent_id=parent_id,
            timeout_seconds=timeout_seconds,
        )

    tool_list = [t for t in ALL_TOOLS if t.tool_name != "launch_training"] + [_launch_training]

    status.write_status(stage="EXPLORE", action="agent initialized", where=dataset, iteration=0)
    log_event("agent_initialized", dataset=dataset, max_runs=max_runs, run_minutes=run_minutes, max_tool_calls=max_tool_calls)

    return Agent(
        model=model,
        tools=tool_list,
        system_prompt=load_system_prompt(),
        hooks=[hooks],
    )
