"""Logging helpers shared by the agent and the training templates.

Two things get written to logs/:
  - an event stream (logs/events.jsonl) — one line per event (dataset
    downloaded, training run started/finished, evaluation performed, etc.)
  - status.json (logs/status.json) — the current state of the run, so the
    agent (or a human) can check "what happened last" without replaying the
    whole event stream.

Run records (one row per training run, with lineage/decision/metrics) live
in logs/ledger.jsonl, written by gotrain/ledger.py — not by this module.

Every evaluation MUST be logged through log_evaluation() with its split name
("train" / "validation" / "test"), so it's always visible which split was
used. The test split is only ever expected to appear here once, at the very
end of a task.

LOGS_DIR defaults to <repo>/logs but can be overridden with the
GOTRAIN_LOGS_DIR environment variable (same pattern as GEO_BENCH_DIR) --
e.g. to point a local dashboard.py at a synced copy of a teammate's logs/
from a different machine instead of this machine's own training runs.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR = Path(os.environ.get("GOTRAIN_LOGS_DIR", DEFAULT_LOGS_DIR))
EVENTS_PATH = LOGS_DIR / "events.jsonl"
STATUS_PATH = LOGS_DIR / "status.json"

VALID_SPLITS = {"train", "validation", "test"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(event_type: str, **fields: Any) -> dict:
    """Append one event to logs/events.jsonl and return the record written."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": _now(), "event": event_type, **fields}
    with open(EVENTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def log_evaluation(
    task: str,
    split: str,
    metrics: dict,
    model: Optional[str] = None,
    **extra: Any,
) -> dict:
    """Log an evaluation result, tagged with the split it was computed on.

    Raises if `split` isn't one of train/validation/test, so a typo can't
    silently mislabel a test-split peek as something else.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    return log_event(
        "evaluation",
        task=task,
        split=split,
        model=model,
        metrics=metrics,
        **extra,
    )


def read_events(event: Optional[str] = None, dataset: Optional[str] = None) -> list:
    """Read logs/events.jsonl, optionally filtered to one event type and/or
    one dataset, oldest first. Used by gotrain.alerts and gotrain.goaudit
    to read back the same event stream log_event writes -- there was
    previously no reader for this file at all.
    """
    if not EVENTS_PATH.exists():
        return []
    records = []
    with open(EVENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if event is not None and record.get("event") != event:
                continue
            if dataset is not None and record.get("dataset") != dataset:
                continue
            records.append(record)
    return records


def read_status() -> dict:
    if not STATUS_PATH.exists():
        return {}
    with open(STATUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def write_status(**fields: Any) -> dict:
    """Merge `fields` into logs/status.json and write it back out atomically.

    Writes to a temp file in the same directory and os.replace()s it over
    status.json, so a reader never sees a half-written file and a crash
    mid-write can't corrupt the existing status.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    status = read_status()
    status.update(fields)
    status["updated_at"] = _now()

    fd, tmp_path = tempfile.mkstemp(dir=LOGS_DIR, prefix=".status_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        os.replace(tmp_path, STATUS_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return status
