"""Run ledger: one record per training run, with lineage and decisions.

Written to logs/ledger.jsonl — one JSON line per run. This is distinct from
logs/events.jsonl (gotrain/logging_utils.py), which holds fine-grained events;
the ledger holds one row per run meant for comparison and lineage tracking
(run_id, parent_id, config diff vs parent, validation metric, decision).

Runs are evaluated and compared on the validation split only. The test split
is not referenced here at all — see gotrain/tools/final_test_eval for the
one-time final evaluation.
"""

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from gotrain.logging_utils import LOGS_DIR

LEDGER_PATH = LOGS_DIR / "ledger.jsonl"

VALID_DECISIONS = {"kept", "discarded", "pending"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_all() -> list:
    if not LEDGER_PATH.exists():
        return []
    records = []
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_all(records: list) -> None:
    """Rewrite the whole ledger atomically (temp file + os.replace)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=LOGS_DIR, prefix=".ledger_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        os.replace(tmp_path, LEDGER_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


HYPOTHESIS_OUTCOME_EPSILON = 0.001  # below this, a delta is noise, not signal


def _evaluate_hypothesis(hypothesis: Optional[dict], delta_vs_parent: Optional[float]) -> Optional[str]:
    """Deterministically grade a hypothesis against what actually happened
    -- "supported" / "not_supported" / "inconclusive" -- never left to the
    model to self-report (see prompts/system_prompt.md: "do not present
    hypotheses as confirmed explanations"). Returns None when there's
    nothing to grade against (no hypothesis, no parent, or no
    expected_direction stated).

    A delta smaller than HYPOTHESIS_OUTCOME_EPSILON is "inconclusive" for
    increase/decrease hypotheses (too small to read as real signal vs.
    noise) but can still "support" a no_change hypothesis, since a small
    delta is exactly what that hypothesis predicted.
    """
    if not hypothesis or delta_vs_parent is None:
        return None
    direction = hypothesis.get("expected_direction")
    if direction is None:
        return None

    is_small = abs(delta_vs_parent) < HYPOTHESIS_OUTCOME_EPSILON
    if direction == "no_change":
        return "supported" if is_small else "not_supported"
    if is_small:
        return "inconclusive"
    if direction == "increase":
        return "supported" if delta_vs_parent > 0 else "not_supported"
    if direction == "decrease":
        return "supported" if delta_vs_parent < 0 else "not_supported"
    return None


def _diff_config(config: dict, parent_config: Optional[dict]) -> dict:
    if not parent_config:
        return {}
    diff = {}
    keys = set(config) | set(parent_config)
    for key in keys:
        old = parent_config.get(key)
        new = config.get(key)
        if old != new:
            diff[key] = {"from": old, "to": new}
    return diff


def add_run(
    dataset: str,
    config: dict,
    val_metric: Any,
    parent_id: Optional[str] = None,
    gpu_minutes: Optional[float] = None,
    decision: str = "pending",
    rationale: Optional[str] = None,
    split_used: str = "validation",
    run_id: Optional[str] = None,
    task_type: Optional[str] = None,
    version_id: Optional[str] = None,
    extra_metrics: Optional[dict] = None,
    hypothesis: Optional[dict] = None,
) -> dict:
    """Append a new run record to the ledger and return it.

    `config_diff` is computed automatically against the parent run's config,
    when a parent is given. `delta_vs_parent` is computed against the
    parent's val_metric when both are numeric. `version_id` is the model
    registry version (models/<dataset>/<version>/) this run was promoted
    to, if any. `extra_metrics` is an opaque passthrough for anything a
    template reports beyond val_metric (e.g. segmentation's
    foreground_iou) that a dashboard may want to show but that isn't part
    of the core comparison logic here. `hypothesis` (see
    gotrain.config_schema.validate_hypothesis) is stored as given, plus a
    `hypothesis_outcome` computed here from the ACTUAL delta_vs_parent --
    the caller's stated expectation is graded by code, not trusted.
    """
    if decision not in VALID_DECISIONS:
        raise ValueError(f"decision must be one of {VALID_DECISIONS}, got {decision!r}")
    if split_used != "validation":
        raise ValueError(
            f"add_run must only be called with split_used='validation', got {split_used!r}. "
            "The test split is evaluated exactly once, via the final evaluation tool."
        )

    records = _read_all()
    parent = None
    if parent_id is not None:
        parent = next((r for r in records if r["run_id"] == parent_id), None)
        if parent is None:
            raise ValueError(f"parent_id {parent_id!r} not found in ledger")

    delta_vs_parent = None
    if parent is not None:
        parent_metric = parent.get("val_metric")
        if isinstance(parent_metric, (int, float)) and isinstance(val_metric, (int, float)):
            delta_vs_parent = val_metric - parent_metric

    record = {
        "run_id": run_id or str(uuid.uuid4()),
        "parent_id": parent_id,
        "dataset": dataset,
        "task_type": task_type,
        "version_id": version_id,
        "config": config,
        "config_diff": _diff_config(config, parent["config"] if parent else None),
        "val_metric": val_metric,
        "delta_vs_parent": delta_vs_parent,
        "gpu_minutes": gpu_minutes,
        "decision": decision,
        "rationale": rationale,
        "split_used": split_used,
        "extra_metrics": extra_metrics or {},
        "hypothesis": hypothesis,
        "hypothesis_outcome": _evaluate_hypothesis(hypothesis, delta_vs_parent),
        "timestamp": _now(),
    }
    records.append(record)
    _write_all(records)
    return record


def update_run(run_id: str, **fields: Any) -> dict:
    """Update fields on an existing run record (e.g. decision, rationale)."""
    if "decision" in fields and fields["decision"] not in VALID_DECISIONS:
        raise ValueError(
            f"decision must be one of {VALID_DECISIONS}, got {fields['decision']!r}"
        )
    records = _read_all()
    for record in records:
        if record["run_id"] == run_id:
            record.update(fields)
            record["updated_at"] = _now()
            _write_all(records)
            return record
    raise ValueError(f"run_id {run_id!r} not found in ledger")


def get_history(dataset: Optional[str] = None) -> list:
    """Return all run records, optionally filtered to one dataset, oldest first."""
    records = _read_all()
    if dataset is not None:
        records = [r for r in records if r["dataset"] == dataset]
    return records


def get_run(run_id: str) -> Optional[dict]:
    """Return the single run record for `run_id`, or None if not found."""
    return next((r for r in _read_all() if r["run_id"] == run_id), None)


def get_best(dataset: str) -> Optional[dict]:
    """Return the run with the highest val_metric for `dataset` (None if no runs)."""
    records = [
        r for r in get_history(dataset)
        if isinstance(r.get("val_metric"), (int, float))
    ]
    if not records:
        return None
    return max(records, key=lambda r: r["val_metric"])


def get_lineage(run_id: str) -> list:
    """Return the chain of runs from the root ancestor down to `run_id` (inclusive)."""
    records = {r["run_id"]: r for r in _read_all()}
    if run_id not in records:
        raise ValueError(f"run_id {run_id!r} not found in ledger")

    chain = []
    current = run_id
    seen = set()
    while current is not None:
        if current in seen:
            raise ValueError(f"cycle detected in ledger lineage at run_id {current!r}")
        seen.add(current)
        record = records.get(current)
        if record is None:
            break
        chain.append(record)
        current = record.get("parent_id")
    chain.reverse()
    return chain
