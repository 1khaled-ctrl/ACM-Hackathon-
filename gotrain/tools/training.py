"""Training/run-management tools for the agent.

launch_training runs a training template as a subprocess (so a crash in
training code can't take down the agent process) and records the run in the
ledger. Everything else here queries the ledger/status — no tool in this
module ever touches the test split; final_test_eval (in final_eval.py) is
the one exception, gated by a lock file.
"""

import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from gotrain import approval, control, ledger, registry, status
from gotrain.config_schema import ConfigValidationError, validate_config, validate_hypothesis
from gotrain.logging_utils import log_event

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATES = {
    "classification": REPO_ROOT / "gotrain" / "templates" / "train_classification.py",
    "segmentation": REPO_ROOT / "gotrain" / "templates" / "train_segmentation.py",
    "fake": REPO_ROOT / "gotrain" / "templates" / "fake_train.py",
}


def launch_training(
    dataset: str,
    task_type: str,
    config: dict,
    parent_id: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    hypothesis: Optional[dict] = None,
) -> dict:
    """Run a training template synchronously and record the result in the ledger.

    `task_type` selects the template ("classification" or "segmentation").
    The template is expected to print a single JSON line to stdout with at
    least a "val_metric" key (and anything else it wants recorded), having
    trained on the train split and evaluated on the validation split only.

    `timeout_seconds`, if given, kills the training subprocess once exceeded
    and raises TimeoutError — this is the per-run wall-clock budget.

    `hypothesis` (see gotrain.config_schema.validate_hypothesis) is required
    whenever `parent_id` is given — every non-baseline run must state what
    changed and why before it's allowed to start.

    Also refuses to run while the engineer control plane
    (gotrain.control) is in a blocking state (paused, manual takeover,
    rollback in progress), and refuses a config that changes a gated
    field (backbone/architecture/encoder) to a value that hasn't been
    engineer-approved yet (gotrain.approval).
    """
    if task_type not in TEMPLATES:
        raise ValueError(f"task_type must be one of {list(TEMPLATES)}, got {task_type!r}")

    # Every check below runs before spawning anything or touching the run
    # budget — a bad config value, a missing/malformed hypothesis, an
    # un-approved gated value, or the control plane being paused must all
    # be rejected here, not discovered after a subprocess has already
    # been counted against max_runs by the agent's limit hook.
    control.raise_if_blocking()
    validate_config(task_type, config)
    validate_hypothesis(hypothesis, has_parent=parent_id is not None)
    approval.check_gate(dataset, task_type, config)

    template_path = TEMPLATES[task_type]
    status.set_action("training", from_=parent_id or "agent", where=dataset)
    log_event("training_started", dataset=dataset, task_type=task_type, config=config)

    import json

    try:
        proc = subprocess.run(
            [sys.executable, str(template_path), "--dataset", dataset, "--config", json.dumps(config)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        log_event(
            "training_timed_out",
            dataset=dataset,
            task_type=task_type,
            timeout_seconds=timeout_seconds,
        )
        status.set_current_run(None)
        raise TimeoutError(
            f"Training for {dataset!r} exceeded its {timeout_seconds}s time budget and was killed."
        )

    # The subprocess has exited (success or failure) — whatever live
    # per-epoch progress it was reporting via status.set_current_run is now
    # stale, win or lose. Clear it before raising on a bad exit code so a
    # dashboard never shows a frozen "epoch 7/10" for a run that's actually
    # over.
    status.set_current_run(None)

    if proc.returncode != 0:
        log_event(
            "training_failed",
            dataset=dataset,
            task_type=task_type,
            stderr=proc.stderr[-4000:],
        )
        raise RuntimeError(f"Training failed for {dataset}: {proc.stderr[-2000:]}")

    result = json.loads(proc.stdout.strip().splitlines()[-1])
    val_metric = result["val_metric"]

    # The ledger's run_id is generated here (rather than inside
    # ledger.add_run) so the same id can be stamped into the registry
    # manifest below, before the ledger record itself exists.
    ledger_run_id = str(uuid.uuid4())

    # Promote the run's config/metrics/checkpoint into the versioned model
    # registry (models/<dataset>/<version>/) when the template reported a
    # run_dir — fake_train.py doesn't produce real artifacts, so there's
    # nothing to register for task_type="fake".
    version_id = None
    run_dir = result.get("run_dir")
    if run_dir is not None:
        version_id = registry.register_run(
            dataset=dataset,
            run_id=ledger_run_id,
            run_dir=run_dir,
            checkpoint_path=result.get("checkpoint_path"),
        )

    record = ledger.add_run(
        dataset=dataset,
        config=config,
        val_metric=val_metric,
        parent_id=parent_id,
        gpu_minutes=result.get("gpu_minutes"),
        decision="pending",
        rationale=result.get("rationale"),
        split_used="validation",
        task_type=task_type,
        version_id=version_id,
        run_id=ledger_run_id,
        extra_metrics=result.get("extra_metrics"),
        hypothesis=hypothesis,
    )

    best = ledger.get_best(dataset)
    if best is not None:
        status.set_best_so_far(best["run_id"], dataset, best["val_metric"])
    status.set_last_event(f"training finished for {dataset}: run {record['run_id']}")
    log_event("training_finished", dataset=dataset, run_id=record["run_id"], val_metric=val_metric)

    return record


def get_run_status(run_id: str) -> dict:
    """Return the ledger record for a run (includes decision, metrics, lineage pointer)."""
    for record in ledger.get_history():
        if record["run_id"] == run_id:
            return record
    raise ValueError(f"run_id {run_id!r} not found")


def get_run_metrics(run_id: str) -> dict:
    """Return just the metric-related fields for a run."""
    record = get_run_status(run_id)
    return {
        "run_id": record["run_id"],
        "dataset": record["dataset"],
        "val_metric": record["val_metric"],
        "delta_vs_parent": record["delta_vs_parent"],
        "split_used": record["split_used"],
        "gpu_minutes": record["gpu_minutes"],
    }


def query_history(dataset: Optional[str] = None) -> list:
    """Return all run records, optionally filtered to one dataset."""
    return ledger.get_history(dataset)


def compare_runs(run_id_a: str, run_id_b: str) -> dict:
    """Compare two runs' configs and validation metrics."""
    a = get_run_status(run_id_a)
    b = get_run_status(run_id_b)

    config_diff = {}
    keys = set(a["config"]) | set(b["config"])
    for key in keys:
        va, vb = a["config"].get(key), b["config"].get(key)
        if va != vb:
            config_diff[key] = {run_id_a: va, run_id_b: vb}

    metric_delta = None
    if isinstance(a["val_metric"], (int, float)) and isinstance(b["val_metric"], (int, float)):
        metric_delta = b["val_metric"] - a["val_metric"]

    return {
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "val_metric_a": a["val_metric"],
        "val_metric_b": b["val_metric"],
        "metric_delta_b_minus_a": metric_delta,
        "config_diff": config_diff,
    }


def record_decision(run_id: str, decision: str, rationale: str) -> dict:
    """Mark a run as kept/discarded/pending, with a rationale."""
    record = ledger.update_run(run_id, decision=decision, rationale=rationale)
    log_event("decision_recorded", run_id=run_id, decision=decision, rationale=rationale)
    return record
