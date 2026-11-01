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

from gotrain import config, control, goaudit, ledger, resource_intelligence, status
from gotrain.config_schema import PreflightValidationError
from gotrain.logging_utils import log_event
from gotrain.tools import adaptation as adaptation_tools
from gotrain.tools import dataiq as dataiq_tools
from gotrain.tools import dataset_support as dataset_support_tools
from gotrain.tools import datasets as datasets_tools
from gotrain.tools import experiment_planner as experiment_planner_tools
from gotrain.tools import geoguard as geoguard_tools
from gotrain.tools import pretrained as pretrained_tools
from gotrain.tools import s3 as s3_tools
from gotrain.tools import stresslab as stresslab_tools
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
def check_dataset_support(dataset: str) -> dict:
    """Check whether GoTrain's training/evaluation code supports `dataset`,
    without downloading anything -- only task_specs.pkl (a small geobench
    metadata file) needs to already exist locally under GEO_BENCH_DIR.

    Returns {"dataset", "supported": bool, "reason": str, "label_kind":
    "single_label"|"multi_label"|"segmentation"|None, "n_classes": int|None,
    "n_bands": int|None}. "supported": False with reason "not found
    locally: ..." means it just hasn't been downloaded yet, not that it
    can't be trained on -- re-check after downloading.

    Args:
        dataset: the GEO-Bench dataset name, e.g. "m-bigearthnet".
    """
    return dataset_support_tools.check_dataset_support(dataset)


@tool
def report_all_datasets() -> dict:
    """Check support status for all 12 GEO-Bench hackathon datasets at
    once, without downloading anything. Use this to see which datasets are
    already trainable and which still need to be downloaded or aren't
    supported, before spending a run budget on any of them.

    Returns {"datasets": [one check_dataset_support result per dataset],
    "n_supported": int, "n_unsupported": int}.
    """
    return dataset_support_tools.report_all_datasets()


@tool
def run_geoguard(dataset: str, task_type: str, max_samples: int = 300) -> dict:
    """Inspect the TRAIN split's data quality and sensor characteristics
    before training: per-band pixel statistics (range, saturation,
    near-constant channels), NaN/Inf counts, class/pixel frequency and
    imbalance, and whether the dataset mixes sensor modalities (e.g.
    m-so2sat's SAR + optical bands). Returns findings plus a recommended
    config (drawn only from the same valid values launch_training accepts)
    with a rationale for each recommendation.

    Call this once, right after get_dataset_info and before your first
    launch_training call, so your baseline config reflects what the data
    actually looks like instead of a generic default.

    Args:
        dataset: the GEO-Bench dataset name, e.g. "m-eurosat".
        task_type: "classification" or "segmentation".
        max_samples: how many TRAIN images to inspect (bounded for speed;
            does not affect the normalization statistics a training run
            itself computes, which always use the full train split).
    """
    return geoguard_tools.run_geoguard(dataset, task_type, max_samples)


@tool
def run_dataiq_structural_scan(dataset: str, task_type: str, max_samples: int = 300) -> dict:
    """Scan a TRAIN sample for corrupted/invalid files and exact
    duplicates (NaN/Inf pixels, constant images, out-of-range labels/
    masks, byte-identical duplicate samples) -- deterministic, model-free
    checks. Objectively corrupted samples are automatically quarantined
    under policy (no engineer approval needed for these specifically) and
    immediately folded into a ready-to-use manifest, returned as
    "quarantine_manifest_id". Call this once, alongside run_geoguard,
    before your first launch_training call.

    Args:
        dataset: the GEO-Bench dataset name.
        task_type: "classification" or "segmentation".
        max_samples: how many TRAIN images to inspect (bounded for speed).
    """
    return dataiq_tools.run_dataiq_structural_scan(dataset, task_type, max_samples)


@tool
def run_dataiq_loss_scan(dataset: str, task_type: str, run_id: str) -> dict:
    """Model-assisted scan for suspicious training examples: loads
    `run_id`'s saved checkpoint and computes per-example training loss /
    predicted class / confidence (classification) or per-image mean pixel
    loss / foreground IoU (segmentation). Flags samples with a high loss
    AND a high-confidence wrong prediction as PENDING data-quality
    proposals (never auto-approved -- call list_data_proposals to see
    them, and they require engineer review before any manifest can use
    them). This is a signal, not proof: a genuinely hard example can look
    identical to a mislabeled one by this metric. Call this after a
    baseline model has trained (needs a real checkpoint to load).

    Args:
        dataset: the GEO-Bench dataset name.
        task_type: "classification" or "segmentation".
        run_id: the run whose checkpoint to scan against.
    """
    return dataiq_tools.run_dataiq_loss_scan(dataset, task_type, run_id)


@tool
def run_stresslab(dataset: str, task_type: str, run_id: str) -> dict:
    """Robustness check: loads `run_id`'s saved checkpoint and re-evaluates
    it on the VALIDATION split under a battery of input perturbations --
    sensor noise, a dropped/dead channel, a lighting shift, and simple
    geometric flips -- scaled to several severities. Reports each
    perturbation's effect on the validation metric and flags any where
    the model is "fragile" or "severely_fragile" (a large relative drop
    vs. the clean baseline), not just the clean accuracy/IoU you already
    have from the run itself. A model with strong clean validation
    accuracy that is fragile here is a real finding worth weighing before
    choosing it as your final model -- mention it in `record_decision` if
    it changes your choice. Never touches the test split. Call this on a
    candidate you're seriously considering submitting, not on every run
    (it costs tool-call budget but not run budget).

    Args:
        dataset: the GEO-Bench dataset name.
        task_type: "classification" or "segmentation".
        run_id: the run whose checkpoint to stress-test.
    """
    return stresslab_tools.run_stresslab(dataset, task_type, run_id)


@tool
def list_data_proposals(dataset: Optional[str] = None, status_filter: Optional[str] = None) -> list:
    """List DataIQ data-quality proposals (from run_dataiq_structural_scan
    or run_dataiq_loss_scan), optionally filtered by dataset and/or status
    ("pending", "approved", "rejected"). Call this to check whether an
    engineer has decided on a proposal before trying to build a manifest
    from it with create_data_manifest.

    Args:
        dataset: restrict to one dataset, or omit for all datasets.
        status_filter: restrict to one status, or omit for all.
    """
    return dataiq_tools.list_data_proposals(dataset, status_filter)


@tool
def create_data_manifest(dataset: str, excluded_sample_ids: list, reason: str, base_manifest_id: str = "base") -> dict:
    """Create a new DataIQ training-data manifest excluding
    `excluded_sample_ids` from `dataset`'s TRAIN split -- never deletes or
    modifies any file, only records which indices a training run should
    skip. Every id must already have an APPROVED EXCLUDE_FROM_TRAINING_
    CANDIDATE or QUARANTINE_INVALID_SAMPLE decision (check with
    list_data_proposals first) -- raises an error naming exactly which
    id isn't approved otherwise, at no cost to your run budget. Pass the
    returned manifest's "manifest_id" as config["manifest_id"] in your
    next launch_training call to run the controlled comparison (Experiment
    B) against your baseline (Experiment A).

    Args:
        dataset: the GEO-Bench dataset name.
        excluded_sample_ids: list of TRAIN-split sample indices to exclude.
        reason: why this manifest was created.
        base_manifest_id: an existing manifest to layer on top of
            (defaults to "base" -- the original, unmodified train split).
    """
    return dataiq_tools.create_data_manifest(dataset, excluded_sample_ids, reason, base_manifest_id)


@tool
def list_compatible_models(task_type: str, dataset: Optional[str] = None) -> list:
    """List candidate pretrained backbones/encoders for `task_type` from
    the curated model registry (gotrain.pretrained), each with its
    architecture, pretraining source/domain, expected input channels, and
    (if `dataset` is given) any cached compatibility-verification result
    for that dataset. This is metadata only -- it never downloads
    anything.

    Args:
        task_type: "classification" or "segmentation".
        dataset: if given, include cached verification status for this
            dataset.
    """
    return pretrained_tools.list_compatible_models(task_type, dataset)


@tool
def estimate_model_resources(model_id: str) -> dict:
    """Get a rough, explicitly-labeled ESTIMATE (never a measurement) of
    a catalog model's parameter count and approximate training GPU
    memory, from gotrain.pretrained.CATALOG.

    Args:
        model_id: a model_id from list_compatible_models's results.
    """
    return pretrained_tools.estimate_model_resources(model_id)


@tool
def validate_model_compatibility(model_id: str, dataset: str, task_type: str, n_bands: int, n_classes: int) -> dict:
    """Actually construct `model_id` for `dataset`'s real band/class
    count and run one forward pass -- the ONLY way a catalog entry's
    compatibility is ever confirmed (never assumed from its metadata
    alone). Raises/returns compatible=false for catalog entries with no
    loader implemented in this environment (the aspirational geospatial-
    foundation-model candidates) rather than attempting an unverified
    download.

    Args:
        model_id: a model_id from list_compatible_models's results.
        dataset: the GEO-Bench dataset to validate against.
        task_type: "classification" or "segmentation".
        n_bands: the dataset's actual band count (see get_dataset_info).
        n_classes: the dataset's actual class count (see get_dataset_info).
    """
    return pretrained_tools.validate_model_compatibility(model_id, dataset, task_type, n_bands, n_classes)


@tool
def get_training_diagnostics(dataset: str, run_id: str) -> dict:
    """Deterministic diagnosis of one completed run, from its ALREADY-
    RECORDED ledger entry and metrics.json -- no model inference is
    re-run. Returns labeled diagnoses (e.g. "possible_overfitting",
    "possible_underfitting", "class_imbalance_hurting_minority_classes",
    "diminishing_returns", "change_did_not_help") plus the evidence
    (actual numbers) behind each one. A diagnosis is a hypothesis worth
    testing, not a proven cause.

    Args:
        dataset: the GEO-Bench dataset name.
        run_id: the run to diagnose.
    """
    return experiment_planner_tools.get_training_diagnostics(dataset, run_id)


@tool
def get_resource_status(dataset: str) -> dict:
    """Your current run/tool-call/data-experiment budget, read from live
    counters -- not something to track or guess yourself. Returns
    runs_used/runs_remaining, tool_calls_used/tool_calls_remaining,
    data_experiments_used/data_experiments_remaining, the per-run
    wall-clock budget (run_minutes), and real measured usage so far
    (total_gpu_minutes_used across completed runs, from the ledger).
    Call this before deciding what to try next, especially once you
    suspect you're running low -- it's free (reading status, not a tool
    call against anything metered) and gives you the exact number instead
    of a guess. Pass `runs_remaining` into `recommend_experiments`'
    `remaining_run_budget` argument rather than estimating it yourself.

    Args:
        dataset: the dataset to report budget status for.
    """
    live = status.get_status().get("resource_budget") or {}
    usage = resource_intelligence.summarize_resource_usage(dataset)
    return {**live, "measured_usage": usage}


@tool
def estimate_run_cost(dataset: str, config: dict) -> dict:
    """Predicts wall-clock GPU-minutes for a candidate config BEFORE you
    spend a run on it, using THIS dataset's own measured history (never
    a generic guess) -- and judges whether that estimate fits within
    your per-run wall-clock budget. A config estimated to exceed that
    budget risks being killed by the timeout partway through, wasting an
    entire run slot on no usable checkpoint at all. Call this before
    `launch_training` whenever you're proposing something notably
    heavier than your baseline (more epochs, a bigger backbone, a larger
    image_size) -- not for every run, since early on there's no history
    yet to estimate from anyway (has_data will say so honestly).

    Args:
        dataset: the GEO-Bench dataset name.
        config: the exact config dict you're about to pass to
            `launch_training` (same keys: backbone/architecture, epochs,
            etc.).
    """
    cost = resource_intelligence.estimate_run_cost(dataset, config)
    live = status.get_status().get("resource_budget") or {}
    fit = resource_intelligence.check_fits_in_run_budget(cost, live.get("run_minutes"))
    return {**cost, "fit": fit}


@tool
def recommend_experiments(dataset: str, run_id: str, remaining_run_budget: Optional[int] = None) -> dict:
    """Evidence-based experiment planner: diagnoses `run_id` (see
    get_training_diagnostics) and proposes ranked, single-variable
    candidate experiments drawn only from already-allowlisted config
    values -- cheapest and lowest-risk first, excluding anything you've
    already tried and anything that doesn't apply to this task_type. Use
    this instead of picking your next change arbitrarily.

    Args:
        dataset: the GEO-Bench dataset name.
        run_id: the run to plan the next experiment from.
        remaining_run_budget: how many launch_training calls you have
            left -- when tight, costlier candidates are ranked lower.
    """
    return experiment_planner_tools.recommend_experiments(dataset, run_id, remaining_run_budget)


@tool
def launch_training(
    dataset: str,
    task_type: str,
    config: dict,
    parent_id: Optional[str] = None,
    hypothesis: Optional[dict] = None,
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
        hypothesis: REQUIRED whenever parent_id is given (rejected
            otherwise, at no cost to your run budget). A dict with:
              - "change": one short sentence naming exactly what you
                changed vs. the parent run's config.
              - "why": one short sentence on the mechanism you expect to
                cause an effect — not "to see what happens".
              - "expected_direction" (optional but recommended): one of
                "increase", "decrease", "no_change" for the validation
                metric.
              - "evidence" (optional): what prior result motivated this
                change.
              - "expected_magnitude" (optional): "small", "moderate", or
                "large".
            The returned record's "hypothesis_outcome" field
            ("supported" / "not_supported" / "inconclusive") is computed
            from the ACTUAL delta_vs_parent against your
            "expected_direction" — it is never self-reported, so cite it
            in record_decision's rationale rather than asserting your own
            read of whether the hypothesis held.

    Returns:
        The ledger record for this run: run_id, val_metric,
        delta_vs_parent, config_diff, hypothesis, hypothesis_outcome, and
        more.
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
def record_adaptation(dataset: str, rejected_proposal: str, new_plan: str, rationale: str) -> dict:
    """Call this immediately whenever a proposal of yours gets rejected or
    blocked -- an approval-gated backbone/architecture/encoder value that
    came back pending/rejected, or a data-quality exclusion with no
    approved decision yet. State clearly what was rejected, what you're
    trying instead, and why, BEFORE you actually try the alternative. This
    is not optional bookkeeping: it is what makes "autonomous but
    accountable" visible rather than just claimed -- without it, a
    rejection just silently turns into a different launch_training call
    and nobody watching can tell the two are connected.

    Do not call this for an ordinary single-variable experiment that
    simply didn't improve on its parent -- that's what `record_decision`
    is for. This is specifically for the moment a proposal was blocked by
    governance (approval gate, unapproved manifest), not for a run that
    completed but scored worse.

    Args:
        dataset: the dataset this adaptation applies to.
        rejected_proposal: one sentence naming exactly what was
            rejected/blocked (e.g. "backbone=resnet50, rejected for GPU
            budget limits").
        new_plan: one sentence naming what you're doing instead.
        rationale: why this alternative is a reasonable substitute, not
            just "trying something else."
    """
    return adaptation_tools.record_adaptation(dataset, rejected_proposal, new_plan, rationale)


@tool
def generate_goaudit_report(dataset: str) -> dict:
    """Generate and persist a GoAudit engineering report for `dataset`:
    a snapshot of progress so far, every kept/discarded decision and its
    rationale, any adaptations recorded after a rejected/blocked
    proposal, pending/rejected engineer approvals, currently active
    failure alerts, and a statistical-confidence note on whether recent
    validation-metric deltas can be told apart from ordinary run-to-run
    noise yet. Call this periodically -- e.g. after every few runs, or
    right before you stop -- so an engineer reviewing your work later
    has a readable account of it instead of needing to reconstruct one
    from raw logs. Purely read-only over your own history: it never
    evaluates anything and never touches the test split. Each call is
    saved as a new, separate report (never overwriting a previous one).

    Args:
        dataset: the dataset to report on.
    """
    report = goaudit.generate_report(dataset)
    path = goaudit.write_report(dataset, report)
    status.set_last_event(f"GoAudit report generated for {dataset}")
    return {"report": report, "path": str(path)}


class MissionRequiresNewExperimentError(ValueError):
    """Raised by submit_final_model when this mission hasn't launched a
    single new training run yet.

    Historical runs from previous missions (query_history, compare_runs)
    remain fully visible as baselines/reference knowledge -- an agent is
    expected to look at them before deciding what to try. What this
    blocks is going straight from "I looked at history" to "I submit
    that old run" with zero new work: a fresh dashboard mission exists
    specifically to run new experiments, so it must run at least one
    before it's allowed to call this done. Diminishing-returns early
    stopping is legitimate -- but only after run #1 of THIS mission, not
    instead of it.
    """


def _submit_final_model(dataset: str, run_id: str, hooks: "LimitsAndLoggingHooks") -> dict:
    """The real implementation behind the submit_final_model @tool,
    parameterized by the session's LimitsAndLoggingHooks instance so it
    can see how many NEW runs this mission has actually launched.
    Bound into a per-session closure in build_agent, same pattern as
    _launch_training below -- see that function's comment for why.
    """
    if hooks.run_count < 1:
        raise MissionRequiresNewExperimentError(
            f"This mission has launched 0 new training runs for {dataset!r} so far "
            f"(run budget: {hooks.max_runs}). query_history may show strong runs from "
            "earlier missions -- those remain valid reference knowledge, but this mission "
            "must call launch_training at least once before submit_final_model is allowed. "
            "If an earlier run's config still looks best, launch_training with that exact "
            "config (as a new run, with it as parent_id) to confirm it reproduces, then "
            "decide from there."
        )

    record = training_tools.get_run_status(run_id)
    if record["dataset"] != dataset:
        raise ValueError(
            f"run_id {run_id!r} belongs to dataset {record['dataset']!r}, not {dataset!r}"
        )
    updated = ledger.update_run(run_id, decision="kept", submitted_final=True)
    status.write_status(stage="FINALIZE", action=f"submitted {run_id} as final model", where=dataset)
    log_event("final_model_submitted", dataset=dataset, run_id=run_id)
    control.transition(control.COMPLETED, reason=f"final model submitted: {run_id}", requested_by="agent")

    # Best-effort: the Model Passport documents what was submitted, but a
    # failure to generate it must never make submit_final_model itself
    # fail (the submission above has already succeeded and is what
    # matters) -- same philosophy as report_progress in the training
    # templates.
    try:
        from gotrain import model_passport

        passport = model_passport.generate_passport(dataset, run_id)
        model_passport.write_passport(dataset, passport)
        log_event("model_passport_generated", dataset=dataset, run_id=run_id)
    except Exception as e:
        log_event("model_passport_generation_failed", dataset=dataset, run_id=run_id, error=str(e))

    _auto_run_stresslab(dataset=dataset, run_id=run_id, record=record)

    return updated


def _auto_run_stresslab(dataset: str, run_id: str, record: dict) -> None:
    """Run StressLab automatically on the model the agent just submitted
    as final -- this is the ONLY place StressLab is triggered without an
    explicit agent tool call. Previously StressLab only ran if the agent
    chose to call the run_stresslab tool, which the system prompt
    describes as optional ("call it on a candidate you're seriously
    weighing... not on every run") -- under tool-call budget pressure the
    agent would skip it, leaving every submitted model's robustness
    unevaluated. This makes it unconditional for the one run that
    actually matters (the submitted model), without spending it on every
    experiment: it is called exactly once per submit_final_model, after
    the submission itself has already succeeded.

    Never raises -- a StressLab failure (missing checkpoint, unsupported
    task_type, a crash inside the torch forward pass) must not undo an
    otherwise-successful submission. Every outcome (ran, skipped as
    unavailable, or failed) is logged and persisted so the dashboard can
    show something more specific than silence:
      - task_type not in ("classification", "segmentation"): logs
        "stresslab_unavailable" with a reason and returns -- covers
        task_type="fake" and any future task type StressLab doesn't
        support yet.
      - no registered checkpoint (version_id is empty -- set by
        registry.register_run only when the training template actually
        reported a run_dir, see gotrain/tools/training.py): same
        "stresslab_unavailable" path, since run_stresslab would just
        raise FileNotFoundError on a checkpoint that was never written.
      - any other exception (torch/geobench import failure, OOM, a bug in
        the perturbation battery itself): logged as "stresslab_failed"
        with the error, not swallowed silently and not re-raised.
    """
    task_type = record.get("task_type")
    if task_type not in ("classification", "segmentation"):
        log_event(
            "stresslab_unavailable", dataset=dataset, run_id=run_id,
            reason=f"StressLab has no evaluation path for task_type={task_type!r}",
        )
        return

    if not record.get("version_id"):
        log_event(
            "stresslab_unavailable", dataset=dataset, run_id=run_id,
            reason="no registered checkpoint for this run (version_id is empty) -- "
                   "the training template may not have reported a run_dir",
        )
        return

    try:
        stresslab_tools.run_stresslab(dataset=dataset, task_type=task_type, run_id=run_id)
    except Exception as e:
        log_event("stresslab_failed", dataset=dataset, run_id=run_id, error=str(e))


@tool
def submit_final_model(dataset: str, run_id: str) -> dict:
    """Submit your final chosen model for this dataset. This only RECORDS
    your choice in the ledger — it does not evaluate anything and does not
    touch the test split. Call this once, when you are done iterating, with
    the run_id of your best validation run. A human will run the one-time
    test evaluation separately afterward.

    This mission must have launched at least one new training run before
    this is allowed, even if query_history shows a strong run from an
    earlier mission — that history is reference knowledge, not a
    substitute for this mission actually training something.

    Args:
        dataset: the dataset this submission is for.
        run_id: the run_id of the model you are submitting as final.
    """
    # This module-level stub is never actually registered on a live agent
    # (build_agent swaps in a per-session closure bound to its
    # LimitsAndLoggingHooks, the same way it does for launch_training) --
    # it exists so ALL_TOOLS / tests can reference submit_final_model by
    # name. Calling it directly without that binding is a programming
    # error, not a reachable agent path.
    raise RuntimeError(
        "submit_final_model must be called through the per-session binding "
        "build_agent() installs; the unbound @tool stub has no mission to check."
    )


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
    check_dataset_support,
    report_all_datasets,
    run_geoguard,
    run_dataiq_structural_scan,
    run_dataiq_loss_scan,
    run_stresslab,
    list_data_proposals,
    create_data_manifest,
    list_compatible_models,
    estimate_model_resources,
    validate_model_compatibility,
    get_training_diagnostics,
    get_resource_status,
    estimate_run_cost,
    recommend_experiments,
    launch_training,
    get_run_status,
    get_run_metrics,
    query_history,
    compare_runs,
    record_decision,
    record_adaptation,
    generate_goaudit_report,
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

    def __init__(
        self,
        dataset: str,
        max_runs: int,
        run_minutes: float,
        max_tool_calls: int,
        max_data_experiments: Optional[int] = config.DEFAULT_MAX_DATA_EXPERIMENTS,
    ):
        self.dataset = dataset
        self.max_runs = max_runs
        self.run_seconds = run_minutes * 60
        self.max_tool_calls = max_tool_calls
        self.max_data_experiments = max_data_experiments
        self.tool_call_count = 0
        self.run_count = 0
        self.data_experiment_count = 0

    @staticmethod
    def _uses_data_experiment(tool_input: dict) -> bool:
        manifest_id = (tool_input.get("config") or {}).get("manifest_id")
        return manifest_id not in (None, "base")

    def _publish_resource_budget(self) -> None:
        """Mirrors this hook's live, in-memory counters into
        logs/status.json so they're visible from OUTSIDE this process --
        gotrain.agent's get_resource_status tool and the dashboard both
        read it back from there, since neither has (or should have)
        direct access to this hook instance. Called on every tool call,
        before and after, so the published snapshot is never more than
        one call stale.
        """
        status.set_resource_budget({
            "dataset": self.dataset,
            "run_minutes": self.run_seconds / 60,
            "runs_used": self.run_count, "max_runs": self.max_runs,
            "runs_remaining": max(self.max_runs - self.run_count, 0),
            "tool_calls_used": self.tool_call_count, "max_tool_calls": self.max_tool_calls,
            "tool_calls_remaining": max(self.max_tool_calls - self.tool_call_count, 0),
            "data_experiments_used": self.data_experiment_count,
            "max_data_experiments": self.max_data_experiments,
            "data_experiments_remaining": (
                max(self.max_data_experiments - self.data_experiment_count, 0)
                if self.max_data_experiments is not None else None
            ),
        })

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._on_before)
        registry.add_callback(AfterToolCallEvent, self._on_after)

    def _on_before(self, event: BeforeToolCallEvent) -> None:
        tool_name = event.tool_use["name"]

        # Engineer control plane (gotrain.control) takes priority over
        # every other check, including the tool-call/run budgets below: a
        # paused/manually-controlled/rollback-in-progress agent gets
        # EVERY tool call cancelled, not just launch_training, and the
        # cancellation doesn't cost tool-call or run budget (it's not the
        # agent's doing). The first blocked call while PAUSE_REQUESTED
        # acknowledges the pause (-> PAUSED_FOR_ENGINEER), which is how
        # the dashboard knows the agent has actually stopped, not just
        # that a pause was requested.
        control_state = control.get_state()
        if control_state in control.BLOCKING_STATES:
            if control_state == control.PAUSE_REQUESTED:
                control.acknowledge_pause()
            event.cancel_tool = (
                f"GoTrain is currently {control_state} (engineer-controlled) — no "
                "tool calls are permitted right now. Do not retry or keep polling: "
                "stop and wait. You will be told when to resume."
            )
            return

        self.tool_call_count += 1
        self._publish_resource_budget()

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

        if (
            tool_name == "launch_training"
            and self.max_data_experiments is not None
            and self._uses_data_experiment(event.tool_use.get("input", {}))
            and self.data_experiment_count >= self.max_data_experiments
        ):
            event.cancel_tool = (
                f"Data-correction experiment budget exhausted ({self.max_data_experiments} used "
                f"for {self.dataset}). Repeatedly testing sample exclusions against the same "
                "validation set risks overfitting to validation, not genuine generalization "
                "improvement. Do not propose another exclusion-based run -- proceed with your "
                "best already-validated configuration (which may or may not be a data-corrected "
                "one; check query_history)."
            )
            return

        if tool_name == "launch_training":
            # Run counting happens in _on_after, once we know whether the
            # call actually attempted training or was rejected pre-flight
            # for a bad config value or hypothesis (PreflightValidationError)
            # — a rejected launch must not cost the agent one of its max_runs.
            event.tool_use["input"]["timeout_seconds"] = self.run_seconds

    def _on_after(self, event: AfterToolCallEvent) -> None:
        tool_name = event.tool_use["name"]
        tool_input = event.tool_use.get("input", {})

        if tool_name == "launch_training" and not isinstance(event.exception, PreflightValidationError):
            self.run_count += 1
            if self._uses_data_experiment(tool_input):
                self.data_experiment_count += 1
            self._publish_resource_budget()

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
                tool_name == "launch_training" and not isinstance(event.exception, PreflightValidationError)
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
    max_data_experiments: Optional[int] = config.DEFAULT_MAX_DATA_EXPERIMENTS,
) -> Agent:
    """Build a Strands Agent wired to the GoTrain tools for one dataset.

    Hard limits (max runs, per-run time budget, max tool calls, and a
    separate max_data_experiments for DataIQ sample-exclusion runs) are
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
        max_data_experiments=max_data_experiments,
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
        hypothesis: Optional[dict] = None,
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
            hypothesis: required whenever parent_id is given — see the
                launch_training @tool docstring above for its shape.
        """
        return training_tools.launch_training(
            dataset=dataset,
            task_type=task_type,
            config=config,
            parent_id=parent_id,
            hypothesis=hypothesis,
            timeout_seconds=timeout_seconds,
        )

    # submit_final_model's real implementation needs to see this session's
    # hooks.run_count (how many NEW runs this mission has launched) before
    # allowing submission -- bound here the same way _launch_training above
    # is bound to this session's timeout, rather than trusting the model
    # to only submit after actually training something.
    @tool(name="submit_final_model")
    def _submit_final_model_tool(dataset: str, run_id: str) -> dict:
        """Submit your final chosen model for this dataset. This only RECORDS
        your choice in the ledger — it does not evaluate anything and does not
        touch the test split. Call this once, when you are done iterating, with
        the run_id of your best validation run. A human will run the one-time
        test evaluation separately afterward.

        This mission must have launched at least one new training run before
        this is allowed, even if query_history shows a strong run from an
        earlier mission — that history is reference knowledge, not a
        substitute for this mission actually training something.

        Args:
            dataset: the dataset this submission is for.
            run_id: the run_id of the model you are submitting as final.
        """
        return _submit_final_model(dataset=dataset, run_id=run_id, hooks=hooks)

    tool_list = [
        t for t in ALL_TOOLS
        if t.tool_name not in ("launch_training", "submit_final_model")
    ] + [_launch_training, _submit_final_model_tool]

    status.write_status(stage="EXPLORE", action="agent initialized", where=dataset, iteration=0)
    log_event("agent_initialized", dataset=dataset, max_runs=max_runs, run_minutes=run_minutes, max_tool_calls=max_tool_calls)

    return Agent(
        model=model,
        tools=tool_list,
        system_prompt=load_system_prompt(),
        hooks=[hooks],
    )
