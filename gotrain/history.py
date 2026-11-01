"""Mission history: one durable record per dashboard-launched session.

A "session" is one press of Start in the NEXUS dashboard -- one
run_gotrain.py subprocess, with its own budgets, its own start/finish
boundary, and whatever experiments the agent chose to run inside it.
ledger.jsonl records individual TRAINING RUNS and events.jsonl records
individual EVENTS, but neither knows where one mission ended and the
next began. This module adds that missing grouping layer.

Storage: logs/history/<session_id>.json, one small JSON file per
session (same append-only, human-readable, no-database spirit as the
rest of logs/).

Deliberately NOT duplicated here: per-run configs, metrics, decisions,
hypotheses, epoch curves. Those already live in ledger.jsonl /
live_metrics.jsonl / events.jsonl, which remain the single source of
truth. A session file stores only what nothing else knows -- the
requested budgets, the time boundary, and how the mission ended -- and
load_session() derives the rest by filtering those logs to the
session's [started_at, finished_at] window. Copying the metrics in here
instead would mean two records of the same fact that can silently
disagree (e.g. after a ledger correction).

Test-split safety: this module reads ledger.jsonl, which
gotrain/ledger.py already refuses to write with split_used !=
"validation". It never touches the test split, never reads
logs/test_lock_*.json, and never imports gotrain.tools.final_eval.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from gotrain.logging_utils import EVENTS_PATH, LIVE_METRICS_PATH, LOGS_DIR

# approval_requested/approval_decided live in their own file (see
# gotrain/approval.py), not events.jsonl -- merged into a session's event
# stream at read time so "History must record engineer
# pause/resume/approval/reject/rollback/manual-control events" holds
# without approval.py writing to two places.
APPROVALS_LOG_PATH = LOGS_DIR / "approvals.jsonl"

HISTORY_DIR = LOGS_DIR / "history"

# Session status values. RUNNING is written at launch; the rest are
# terminal and written when the session is closed out.
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
STOPPED = "STOPPED"
FAILED = "FAILED"
UNKNOWN = "UNKNOWN"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def new_session_id() -> str:
    return "session_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _session_path(session_id: str) -> Path:
    # Guard against a session_id with path separators ever reaching the
    # filesystem -- these ids are generated here, but this function is
    # also reachable from an HTTP path parameter via the dashboard API.
    safe = Path(session_id).name
    return HISTORY_DIR / f"{safe}.json"


def _write_atomic(path: Path, data: dict) -> dict:
    """Same tempfile + os.replace pattern as status.json/control.json, so
    a dashboard reading mid-write never sees a truncated file.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=HISTORY_DIR, prefix=".session_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return data


def start_session(
    dataset: str,
    max_runs: int,
    run_minutes: float,
    max_tool_calls: Optional[int] = None,
    pid: Optional[int] = None,
    session_id: Optional[str] = None,
) -> dict:
    """Open a new session record. Called by the dashboard API the moment
    a mission is launched -- before the subprocess has done anything, so
    an agent that dies on startup still leaves a visible record of
    having been attempted rather than vanishing without trace.
    """
    session_id = session_id or new_session_id()
    record = {
        "session_id": session_id,
        "dataset": dataset,
        "runs_requested": max_runs,
        "minutes_per_run": run_minutes,
        "max_tool_calls": max_tool_calls,
        "pid": pid,
        "started_at": _now(),
        "finished_at": None,
        "status": RUNNING,
    }
    _write_atomic(_session_path(session_id), record)
    return record


def finish_session(session_id: str, status: str = COMPLETED, **extra: Any) -> dict:
    """Close out a session: stamp finished_at, duration, and final
    status. A no-op returning the existing record if the session file is
    missing (nothing to close), so a stray call can't create a
    half-formed record.
    """
    path = _session_path(session_id)
    if not path.exists():
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("finished_at"):
        return record  # already closed; never re-stamp a finished session

    record["finished_at"] = _now()
    record["status"] = status
    started = _parse(record.get("started_at"))
    finished = _parse(record["finished_at"])
    if started and finished:
        record["duration_seconds"] = round((finished - started).total_seconds(), 3)
    record.update(extra)
    _write_atomic(path, record)
    return record


def get_open_session() -> Optional[dict]:
    """The most recent session still marked RUNNING, if any. Used to
    attribute a finish/stop to the right session without the caller
    having to track the id.
    """
    for record in list_sessions():
        if record.get("status") == RUNNING:
            return record
    return None


def get_current_session() -> Optional[dict]:
    """The session the live dashboard should attribute its Decision
    Timeline / Run Comparison / Event Feed panels to: the RUNNING one if
    there is one, else the single most recently started session overall
    (so those panels don't go blank the instant a mission finishes --
    they keep showing its final state until a new mission starts).
    Returns a bare _derive()d summary; callers needing runs/events/
    live_metrics should pass its session_id to load_session().
    """
    sessions = list_sessions()
    if not sessions:
        return None
    for record in sessions:
        if record.get("status") == RUNNING:
            return record
    return sessions[0]  # already sorted newest-first by list_sessions()


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a writer may be mid-line
    return rows


def _in_window(ts: Optional[str], start: Optional[datetime], end: Optional[datetime]) -> bool:
    moment = _parse(ts)
    if moment is None or start is None:
        return False
    if moment < start:
        return False
    # An open session has no end yet -- everything after start belongs to it.
    return end is None or moment <= end


def _windowed_events(start: Optional[datetime], end: Optional[datetime]) -> list:
    """events.jsonl + approvals.jsonl, merged and time-sorted, filtered to
    one session's window. One merge point so the live snapshot and
    History Details can never see two different event streams for the
    same mission. Deduplicated on (timestamp, event, run_id/key) -- two
    processes racing to write the same fact (e.g. a reconciler and the
    agent both logging the same control_transition) must not show twice.
    """
    merged = _read_jsonl(EVENTS_PATH) + _read_jsonl(APPROVALS_LOG_PATH)
    windowed = [e for e in merged if _in_window(e.get("timestamp"), start, end)]
    windowed.sort(key=lambda e: e.get("timestamp") or "")

    seen = set()
    deduped = []
    for e in windowed:
        key = (e.get("timestamp"), e.get("event"), e.get("run_id") or e.get("key"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
    return deduped


def _derive(record: dict) -> dict:
    """Attach the metrics this session produced, derived from the ledger
    rather than stored here. See the module docstring for why.
    """
    start = _parse(record.get("started_at"))
    end = _parse(record.get("finished_at"))

    try:
        from gotrain import ledger

        all_runs = ledger.get_history(record.get("dataset"))
    except Exception:
        all_runs = []

    runs = [r for r in all_runs if _in_window(r.get("timestamp"), start, end)]
    scored = [r for r in runs if isinstance(r.get("val_metric"), (int, float))]
    best = max(scored, key=lambda r: r["val_metric"]) if scored else None

    record["runs_completed"] = len(runs)
    record["run_ids"] = [r.get("run_id") for r in runs]
    record["best_val_metric"] = best.get("val_metric") if best else None
    record["best_run_id"] = best.get("run_id") if best else None
    if best:
        config = best.get("config") or {}
        record["best_model"] = config.get("backbone") or config.get("encoder") or config.get("architecture")
    else:
        record["best_model"] = None

    events = _windowed_events(start, end)
    record["event_count"] = len(events)
    submitted = [e for e in events if e.get("event") == "final_model_submitted"]
    record["final_model_run_id"] = submitted[-1].get("run_id") if submitted else None
    record["timed_out_runs"] = sum(1 for e in events if e.get("event") == "training_timed_out")
    # "Engineer intervention" = anything a human did to this mission from
    # the dashboard: pause/resume/manual-control (control_transition),
    # approve/reject (approval_decided -- approval_requested is the
    # agent's own proposal, not an engineer action), and rollback
    # (rollback_succeeded/rollback_failed, always engineer-initiated per
    # gotrain/rollback.py).
    record["interventions"] = [
        e for e in events
        if (e.get("event") == "control_transition" and e.get("requested_by") == "engineer")
        or e.get("event") == "approval_decided"
        or e.get("event") in ("rollback_succeeded", "rollback_failed")
    ]
    return record


def list_sessions(limit: Optional[int] = None) -> list:
    """All sessions, newest first, each with derived summary metrics."""
    if not HISTORY_DIR.is_dir():
        return []
    records = []
    for path in HISTORY_DIR.glob("session_*.json"):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue  # a corrupt/half-written file must not break the list
    records.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    if limit is not None:
        records = records[:limit]
    return [_derive(r) for r in records]


def load_session(session_id: str) -> Optional[dict]:
    """One session with its full detail: summary, the ledger runs it
    produced (configs, hypotheses, decisions, lineage), its per-epoch
    curves, and its event stream.
    """
    path = _session_path(session_id)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    record = _derive(record)
    start = _parse(record.get("started_at"))
    end = _parse(record.get("finished_at"))

    try:
        from gotrain import ledger

        all_runs = ledger.get_history(record.get("dataset"))
    except Exception:
        all_runs = []

    record["runs"] = [r for r in all_runs if _in_window(r.get("timestamp"), start, end)]
    record["events"] = _windowed_events(start, end)
    record["live_metrics"] = [
        m for m in _read_jsonl(LIVE_METRICS_PATH) if _in_window(m.get("timestamp"), start, end)
    ]
    return record
