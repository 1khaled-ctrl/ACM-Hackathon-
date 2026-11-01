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
    """Pre-existing bug fixed here: millisecond-truncated timestamps alone
    collide under rapid successive calls (seen in practice once this
    module's own test suite grew dense enough to trigger it -- two
    start_session() calls landing in the same millisecond silently
    overwrote one session's file). A short random suffix makes collision
    astronomically unlikely without losing the timestamp prefix
    list_sessions() et al. read for display -- sort order itself always
    comes from the stored started_at field, never from the id string.
    """
    import secrets

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"session_{stamp}_{secrets.token_hex(3)}"


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


def derive_stop_reason(dataset: Optional[str], started_at: Optional[str], ended_at: Optional[str] = None) -> Optional[str]:
    """A short, human-readable reason a STOPPED session ended with no
    completed runs or fewer than requested -- previously the dashboard
    could only show "STOPPED, 0/3" with no indication of WHY, even though
    gotrain.tools.training.launch_training already logs exactly what went
    wrong (training_timed_out / training_failed) to events.jsonl every
    time; that information just never made it into the session record.

    Looks at this session's own event window (same merged/deduped stream
    load_session()/list_sessions() already use) for the most specific
    signal available, in priority order:
      1. The LAST launch_training attempt's outcome (timeout vs. crash) --
         the most direct, almost always the actual cause when runs_completed
         is 0 or short of what was requested.
      2. Tool-call budget exhaustion (the hook's own cancellation message,
         logged nowhere currently -- inferred from the session having hit
         its max_tool_calls with no final_model_submitted event).
      3. None when there's no training-related event at all to point to
         (e.g. the agent process died before ever calling launch_training,
         or the session is not actually STOPPED) -- callers fall back to a
         generic label rather than inventing a reason with no evidence.

    `ended_at` defaults to now (ISO string or None) -- a STOPPED session is
    always closed out live, at the moment its process is found dead, so
    there is no better "end" to bound the window with than the present.

    Returns None, never raises -- a courtesy annotation, same spirit as
    _training_health, never allowed to break session closing itself.
    """
    if not dataset:
        return None
    try:
        start = _parse(started_at)
        end = _parse(ended_at) if ended_at else datetime.now(timezone.utc)
        events = _windowed_events(start, end)
    except Exception:
        return None

    training_events = [
        e for e in events
        if e.get("event") in ("training_timed_out", "training_failed", "training_finished")
        and e.get("dataset") == dataset
    ]
    if training_events:
        last = training_events[-1]
        if last["event"] == "training_timed_out":
            timeout = last.get("timeout_seconds")
            suffix = f" after {timeout:.0f}s" if isinstance(timeout, (int, float)) else ""
            return f"training timed out{suffix} with no completed run"
        if last["event"] == "training_failed":
            stderr_tail = (last.get("stderr") or "").strip().splitlines()
            last_line = stderr_tail[-1][:160] if stderr_tail else None
            return f"training crashed: {last_line}" if last_line else "training crashed (see train.log)"
        # last event is training_finished: runs did complete, so a
        # shortfall vs. the requested count means something stopped the
        # MISSION after that, not training itself (budget exhaustion,
        # engineer stop) -- fall through to the generic reasons below.

    submitted = any(e.get("event") == "final_model_submitted" for e in events)
    if not submitted:
        tool_calls = sum(1 for e in events if e.get("event") == "tool_call")
        if tool_calls:
            return "stopped before submitting a final model (tool-call or run budget exhausted)"

    return None


# Training Health card states. UNSTABLE is deliberately not produced by
# _training_health below -- gotrain/experiment_planner.py's
# diagnose_run_record has no loss-spike/NaN/divergence diagnosis to map
# from, and the brief is explicit: map existing diagnoses, don't invent a
# new heuristic. The state stays defined (so the UI has a real place to
# render it) for whenever a diagnosis that warrants it exists upstream.
HEALTH_HEALTHY = "HEALTHY"
HEALTH_OVERFITTING = "OVERFITTING"
HEALTH_UNDERFITTING = "UNDERFITTING"
HEALTH_UNSTABLE = "UNSTABLE"
HEALTH_UNKNOWN = "UNKNOWN"

# gotrain/experiment_planner.py diagnose_run_record()'s real vocabulary
# (the only diagnoses that module ever produces) mapped to one evidence
# phrase each -- plain-English renderings of the SAME evidence numbers
# diagnose_run_record already computed (train_loss, val_metric), not new
# measurements. "possible_" is dropped from the label; the uncertainty is
# still real (these are heuristics over recorded numbers, not a
# re-inference over the model -- see experiment_planner.py's own
# docstring) but the card has no room for that caveat, so the evidence
# line names the literal numbers instead of overclaiming confidence.
_DIAGNOSIS_EVIDENCE = {
    "possible_overfitting": lambda ev: (
        f"train loss {ev['train_loss']:.3f} but val metric only {ev['val_metric']:.3f}"
        if "train_loss" in ev and "val_metric" in ev else "low train loss, validation lagging"
    ),
    "possible_underfitting": lambda ev: (
        f"train loss still {ev['train_loss']:.3f}, val metric {ev['val_metric']:.3f}"
        if "train_loss" in ev and "val_metric" in ev else "train loss has not dropped enough yet"
    ),
}


def _training_health(dataset: Optional[str], runs: list) -> dict:
    """{"state": one of HEALTH_*, "evidence": [<=2 short strings]} for the
    most recently completed run in this session's window -- reuses
    gotrain.experiment_planner.get_training_diagnostics exactly as an
    agent tool call would, rather than re-deriving anything from scratch.
    UNKNOWN (not an error) whenever there is no completed run yet to
    diagnose, or the lookup itself fails for any reason -- diagnostics
    are a courtesy overlay on top of the real run data, never allowed to
    break session summary building.
    """
    if not runs or not dataset:
        return {"state": HEALTH_UNKNOWN, "evidence": []}

    latest = max(runs, key=lambda r: r.get("timestamp") or "")
    run_id = latest.get("run_id")
    if not run_id:
        return {"state": HEALTH_UNKNOWN, "evidence": []}

    try:
        from gotrain.experiment_planner import get_training_diagnostics

        diagnostics = get_training_diagnostics(dataset, run_id)
    except Exception:
        return {"state": HEALTH_UNKNOWN, "evidence": []}

    diagnoses = diagnostics.get("diagnoses") or []
    evidence_data = diagnostics.get("evidence") or {}

    if "possible_overfitting" in diagnoses:
        state = HEALTH_OVERFITTING
    elif "possible_underfitting" in diagnoses:
        state = HEALTH_UNDERFITTING
    else:
        state = HEALTH_HEALTHY

    evidence = []
    for name in ("possible_overfitting", "possible_underfitting"):
        if name in diagnoses:
            evidence.append(_DIAGNOSIS_EVIDENCE[name](evidence_data))
    if not evidence and "diminishing_returns" in diagnoses:
        evidence.append("validation gains have stalled run over run")
    if len(evidence) < 2 and "class_imbalance_hurting_minority_classes" in diagnoses:
        gap = evidence_data.get("per_class_metric_gap")
        evidence.append(f"widening per-class gap ({gap:.2f})" if isinstance(gap, (int, float)) else "widening per-class gap")

    return {"state": state, "evidence": evidence[:2], "run_id": run_id}


# Statistical Improvement Verification states, mapped from
# gotrain/statistics.py::is_improvement_significant's existing tri-state
# `significant` field (True / False / None) -- no new verdict logic here,
# just naming the three cases that module already distinguishes.
STAT_VERIFIED = "VERIFIED"
STAT_NOT_ABOVE_NOISE = "NOT_ABOVE_NOISE"
STAT_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


def _attach_statistical_verdicts(dataset: Optional[str], runs: list) -> list:
    """Attach `statistical_verdict` (VERIFIED/NOT_ABOVE_NOISE/
    INSUFFICIENT_EVIDENCE + reason + noise_floor) to a copy of each run,
    reusing gotrain.statistics.is_improvement_significant exactly against
    one shared estimate_noise_floor call for the whole list -- never a
    re-derivation, and never one noise-floor scan of the dataset's full
    ledger per run. Returns `runs` with a None verdict on every entry if
    the dataset is missing or the statistics call fails; a courtesy
    field, never allowed to break run comparison itself.
    """
    if not dataset:
        return [{**r, "statistical_verdict": None} for r in runs]
    try:
        from gotrain import statistics

        noise_floor = statistics.estimate_noise_floor(dataset)
    except Exception:
        return [{**r, "statistical_verdict": None} for r in runs]

    out = []
    for run in runs:
        try:
            verdict = statistics.is_improvement_significant(run.get("delta_vs_parent"), noise_floor)
        except Exception:
            out.append({**run, "statistical_verdict": None})
            continue
        if verdict["significant"] is True:
            state = STAT_VERIFIED
        elif verdict["significant"] is False:
            state = STAT_NOT_ABOVE_NOISE
        else:
            state = STAT_INSUFFICIENT_EVIDENCE
        out.append({
            **run,
            "statistical_verdict": {
                "state": state,
                "reason": verdict.get("reason"),
                "noise_floor": noise_floor.get("noise_floor"),
            },
        })
    return out


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

    record["training_health"] = _training_health(record.get("dataset"), runs)

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

    runs = [r for r in all_runs if _in_window(r.get("timestamp"), start, end)]
    record["runs"] = _attach_statistical_verdicts(record.get("dataset"), runs)
    record["events"] = _windowed_events(start, end)
    record["live_metrics"] = [
        m for m in _read_jsonl(LIVE_METRICS_PATH) if _in_window(m.get("timestamp"), start, end)
    ]
    return record
