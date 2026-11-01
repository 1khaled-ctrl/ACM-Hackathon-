"""Evidence-based experiment planning -- replaces ad hoc/random tuning
with a deterministic diagnose -> propose -> rank pipeline, drawing only
on data the project has ALREADY recorded (the ledger, a run's own
metrics.json) rather than re-running anything.

get_training_diagnostics(dataset, run_id) reads a run's ledger record +
saved metrics.json and derives labeled, evidence-backed diagnoses
(possible_overfitting, possible_underfitting, class_imbalance_hurting_
minority_classes, change_did_not_help, diminishing_returns). Every
diagnosis is a HEURISTIC over real recorded numbers, never a re-inference
over the model -- it's explicitly labeled as such, and the brief's own
caution applies: a heuristic match is a hypothesis to test, not a proven
cause.

recommend_experiments(dataset, run_id, ...) turns diagnoses into ranked
candidate single-variable changes drawn only from gotrain.config_schema's
allowlisted values (never a free-form suggestion), filtered against prior
experiment history so a previously-tried direction is never re-suggested,
and ranked cheap-and-low-risk-first via rank_experiments().
"""

import json
from pathlib import Path
from typing import Optional

from gotrain import ledger
from gotrain.config_schema import VALID_VALUES_BY_TASK_TYPE

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"

# Heuristic thresholds -- deliberately conservative, documented here so
# they're one place to tune, not scattered magic numbers.
OVERFITTING_TRAIN_LOSS_THRESHOLD = 0.15
OVERFITTING_VAL_METRIC_CEILING = {"classification": 0.90, "segmentation": 0.75}
UNDERFITTING_TRAIN_LOSS_FLOOR = 0.8
UNDERFITTING_VAL_METRIC_CEILING = {"classification": 0.5, "segmentation": 0.3}
PLATEAU_DELTA_THRESHOLD = 0.003  # matches the system prompt's stopping-condition "<0.3 points"
MINORITY_CLASS_GAP_THRESHOLD = 0.25  # best-vs-worst per-class metric gap considered "hurting minority classes"

COST_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}
RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}

# Candidate single-variable experiments per diagnosis. Each entry names
# exactly one field to change plus the direction/value, a plain-language
# test statement, and an expected cost/risk -- "classification_only" /
# "segmentation_only" restrict a candidate to the task type it actually
# applies to (e.g. "loss" is not a classification config key).
DIAGNOSIS_TO_CANDIDATES = {
    "possible_overfitting": [
        {"change_field": "weight_decay", "direction": "increase", "cost": "low", "risk": "low",
         "test": "does stronger L2 regularization reduce the train/val gap without hurting val_metric?"},
        {"change_field": "augmentation", "direction": "increase", "cost": "medium", "risk": "low",
         "test": "does more aggressive augmentation reduce overfitting by increasing effective data diversity?"},
        {"change_field": "label_smoothing", "direction": "increase", "cost": "low", "risk": "low",
         "task_type_only": "classification",
         "test": "does label smoothing reduce overconfident, overfit predictions?"},
    ],
    "possible_underfitting": [
        {"change_field": "epochs", "direction": "increase", "cost": "medium", "risk": "low",
         "test": "has the model simply not been trained long enough to converge?"},
        {"change_field": "lr", "direction": "increase", "cost": "low", "risk": "medium",
         "test": "is the learning rate too small for meaningful progress within the epoch budget?"},
        {"change_field": "backbone", "direction": "increase_capacity", "cost": "medium", "risk": "medium",
         "task_type_only": "classification",
         "test": "is the current backbone too small to fit this task at all?"},
    ],
    "class_imbalance_hurting_minority_classes": [
        {"change_field": "class_weights", "direction": "set_capped_inverse_frequency", "cost": "low", "risk": "low",
         "test": "does class weighting improve minority-class metrics without materially hurting the overall metric?"},
        {"change_field": "loss", "direction": "set:ce+dice", "cost": "low", "risk": "low",
         "task_type_only": "segmentation",
         "test": "does a Dice-augmented loss improve foreground/minority-class performance?"},
    ],
    "change_did_not_help": [
        {"change_field": "revert_to_parent", "direction": "revert", "cost": "none", "risk": "none",
         "test": "confirm the parent run remains the best validated checkpoint; do not repeat this exact change."},
    ],
    "diminishing_returns": [
        {"change_field": "stop_and_submit", "direction": "submit_final_model", "cost": "none", "risk": "none",
         "test": "further single-variable tweaks are unlikely to beat the remaining budget's opportunity cost."},
    ],
}


def _run_metrics_path(dataset: str, run_id: str) -> Path:
    return RUNS_DIR / dataset / run_id / "metrics.json"


def _read_run_metrics(dataset: str, run_id: str) -> Optional[dict]:
    path = _run_metrics_path(dataset, run_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_run_metrics_file(dataset: str, run_id: str) -> Optional[dict]:
    """Public read-only accessor for a run's saved metrics.json (the same
    file _read_run_metrics reads internally) -- exposed for callers
    outside this module (the FastAPI bridge's lineage endpoint) that need
    fields this file has but the ledger record doesn't, e.g.
    warm_started_from. None if the run has no metrics.json (never ran
    locally, or was cleaned up) -- never raises for a missing file.
    """
    return _read_run_metrics(dataset, run_id)


def diagnose_run_record(run_record: dict, metrics: Optional[dict], task_type: str) -> dict:
    """Pure core: given an already-loaded ledger record and (optionally)
    its run_dir metrics.json, return {"diagnoses": [...], "evidence": {...}}.
    Never reads a file itself -- callers (get_training_diagnostics, or a
    test) supply already-loaded data, so this is fully unit-testable
    without any run directory on disk.
    """
    diagnoses = []
    evidence = {}

    val_metric = run_record.get("val_metric")
    train_loss = (metrics or {}).get("best", {}).get("train_loss") if metrics else None
    ceiling = OVERFITTING_VAL_METRIC_CEILING.get(task_type, 0.9)
    underfitting_ceiling = UNDERFITTING_VAL_METRIC_CEILING.get(task_type, 0.5)

    if train_loss is not None and val_metric is not None:
        evidence["train_loss"] = train_loss
        evidence["val_metric"] = val_metric
        if train_loss < OVERFITTING_TRAIN_LOSS_THRESHOLD and val_metric < ceiling:
            diagnoses.append("possible_overfitting")
        elif train_loss > UNDERFITTING_TRAIN_LOSS_FLOOR and val_metric < underfitting_ceiling:
            diagnoses.append("possible_underfitting")

    delta = run_record.get("delta_vs_parent")
    if isinstance(delta, (int, float)):
        evidence["delta_vs_parent"] = delta
        if abs(delta) < PLATEAU_DELTA_THRESHOLD:
            diagnoses.append("diminishing_returns")

    if run_record.get("hypothesis_outcome") == "not_supported":
        evidence["hypothesis_outcome"] = "not_supported"
        evidence["hypothesis"] = run_record.get("hypothesis")
        diagnoses.append("change_did_not_help")

    per_class = None
    if metrics:
        per_class = (metrics.get("best") or {}).get("val_per_class_accuracy") or (metrics.get("best") or {}).get("val_per_class_iou")
    if per_class:
        numeric = [v for v in per_class if isinstance(v, (int, float))]
        if len(numeric) >= 2:
            gap = max(numeric) - min(numeric)
            evidence["per_class_metric_gap"] = gap
            if gap >= MINORITY_CLASS_GAP_THRESHOLD:
                diagnoses.append("class_imbalance_hurting_minority_classes")

    return {"diagnoses": diagnoses, "evidence": evidence}


def get_training_diagnostics(dataset: str, run_id: str) -> dict:
    """Read a run's ledger record + saved metrics.json (if its run
    directory still exists -- it may not, for very old runs cleaned up
    manually; metrics.json is read-only and never required) and return
    diagnose_run_record's result, plus the raw record/metrics for
    reference.
    """
    run_record = next((r for r in ledger.get_history(dataset) if r["run_id"] == run_id), None)
    if run_record is None:
        raise ValueError(f"run_id {run_id!r} not found for dataset {dataset!r}")

    task_type = run_record.get("task_type") or "classification"
    metrics = _read_run_metrics(dataset, run_id)
    diagnosis = diagnose_run_record(run_record, metrics, task_type)

    return {
        "dataset": dataset,
        "run_id": run_id,
        "task_type": task_type,
        "diagnoses": diagnosis["diagnoses"],
        "evidence": diagnosis["evidence"],
        "val_metric": run_record.get("val_metric"),
        "config": run_record.get("config"),
    }


def _already_tried_directions(dataset: str, field: str) -> set:
    """Every (direction-ish) value this dataset's history has already
    changed `field` to/from, via config_diff -- used so
    recommend_experiments never re-suggests an identical change.
    """
    tried = set()
    for record in ledger.get_history(dataset):
        diff = record.get("config_diff") or {}
        if field in diff:
            tried.add(diff[field].get("to"))
    return tried


def recommend_experiments(dataset: str, run_id: str, remaining_run_budget: Optional[int] = None) -> dict:
    """The main evidence-based planner entry point: diagnose `run_id`,
    propose candidate single-variable experiments for every diagnosis
    found, drop any whose exact field doesn't apply to this task_type,
    rank by cost/risk (cheapest+safest first, see rank_experiments), and
    return everything needed to decide: diagnosis, evidence, ranked
    candidates.
    """
    diagnostics = get_training_diagnostics(dataset, run_id)
    task_type = diagnostics["task_type"]
    valid_fields = set(VALID_VALUES_BY_TASK_TYPE.get(task_type, {})) | {
        "lr", "epochs", "weight_decay", "augmentation", "label_smoothing", "class_weights",
        "backbone", "loss", "revert_to_parent", "stop_and_submit",
    }

    candidates = []
    for diagnosis in diagnostics["diagnoses"]:
        for candidate in DIAGNOSIS_TO_CANDIDATES.get(diagnosis, []):
            restriction = candidate.get("task_type_only")
            if restriction is not None and restriction != task_type:
                continue
            if candidate["change_field"] not in valid_fields:
                continue
            tried = _already_tried_directions(dataset, candidate["change_field"])
            if candidate["direction"] in tried:
                continue
            row = dict(candidate)
            row["diagnosis"] = diagnosis
            candidates.append(row)

    ranked = rank_experiments(candidates, remaining_run_budget=remaining_run_budget)
    return {
        "dataset": dataset,
        "run_id": run_id,
        "diagnosis": diagnostics["diagnoses"],
        "evidence": diagnostics["evidence"],
        "candidate_experiments": ranked,
    }


def rank_experiments(candidates: list, remaining_run_budget: Optional[int] = None) -> list:
    """Sort candidates cheapest-and-lowest-risk first. When
    `remaining_run_budget` is tight (<=2 runs left), anything costed
    "medium" or "high" is pushed further down -- there may not be budget
    left to recover from a disappointing result.
    """
    def key(c):
        cost_rank = COST_ORDER.get(c.get("cost", "medium"), 2)
        risk_rank = RISK_ORDER.get(c.get("risk", "medium"), 2)
        low_budget_penalty = 10 if (remaining_run_budget is not None and remaining_run_budget <= 2 and cost_rank >= 2) else 0
        return (low_budget_penalty, cost_rank, risk_rank)

    return sorted(candidates, key=key)
