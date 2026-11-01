"""GoAudit: periodically-generated, human-readable AI engineering reports
-- the explainability/traceability layer tying together data this project
already logs (the run ledger, approvals, adaptation records, DataIQ/
calibration signals, the control plane) into one document an engineer can
read instead of reconstructing the story by hand from several logs.

generate_report builds the structured dict; format_report renders it as
readable text. write_report persists a timestamped snapshot under
logs/goaudit/<dataset>/, so multiple reports over a session form an audit
trail (never overwriting a previous report) -- this is deliberately NOT a
single "latest status" file, which is what logs/status.json already is.

Pure Python/JSON reads over existing logs -- no torch needed.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from gotrain import alerts, approval, ledger, resource_intelligence, statistics
from gotrain.logging_utils import LOGS_DIR, read_events

GOAUDIT_DIR = LOGS_DIR / "goaudit"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress_section(history: list) -> dict:
    scored = [r for r in history if isinstance(r.get("val_metric"), (int, float))]
    best = max(scored, key=lambda r: r["val_metric"]) if scored else None

    # "Improving" / "plateaued" read off the same single-variable-tweak
    # pacing the stopping condition in prompts/system_prompt.md uses: the
    # last 2 deltas vs. parent, if both are present and numeric.
    recent_deltas = [r["delta_vs_parent"] for r in history[-2:] if isinstance(r.get("delta_vs_parent"), (int, float))]
    trend = "insufficient_data"
    if len(recent_deltas) == 2:
        trend = "plateaued" if all(abs(d) < 0.3 for d in recent_deltas) else "improving"

    return {
        "n_runs": len(history),
        "n_scored_runs": len(scored),
        "best_run_id": best["run_id"] if best else None,
        "best_val_metric": best["val_metric"] if best else None,
        "trend": trend,
    }


def _resource_usage_section(dataset: str) -> dict:
    return resource_intelligence.summarize_resource_usage(dataset)


def _decisions_section(history: list) -> list:
    return [
        {
            "run_id": r["run_id"],
            "decision": r.get("decision"),
            "val_metric": r.get("val_metric"),
            "delta_vs_parent": r.get("delta_vs_parent"),
            "rationale": r.get("rationale"),
            "hypothesis_outcome": r.get("hypothesis_outcome"),
        }
        for r in history
    ]


def _adaptations_section(dataset: str) -> list:
    events = read_events(event="adaptation_recorded", dataset=dataset)
    return [
        {
            "rejected_proposal": e.get("rejected_proposal"),
            "new_plan": e.get("new_plan"),
            "rationale": e.get("rationale"),
            "recorded_at": e.get("recorded_at"),
        }
        for e in events
    ]


def _approvals_section(dataset: str) -> list:
    return [
        {
            "field": p["field"], "value": p["value"], "status": p["status"],
            "decision_reason": p.get("decision_reason"),
        }
        for p in approval.list_proposals(kind="config_gate") if p["dataset"] == dataset
    ]


def _recommendations(progress: dict, approvals: list, failure_alerts: list, noise_floor: dict) -> list:
    recommendations = []
    pending = [p for p in approvals if p["status"] == approval.PENDING]
    if pending:
        recommendations.append(
            f"{len(pending)} gated config value(s) are awaiting engineer approval "
            f"({', '.join(f'{p['field']}={p['value']!r}' for p in pending)}) -- "
            "a decision here would unblock exploration the agent cannot currently make."
        )
    if progress["trend"] == "plateaued":
        recommendations.append(
            "Validation metric has plateaued over the last 2 runs (both deltas < 0.3) -- "
            "the agent should be nearing its stopping condition rather than continuing "
            "single-variable tweaks."
        )
    if any(a["severity"] == alerts.CRITICAL for a in failure_alerts):
        recommendations.append(
            "One or more CRITICAL alerts are active for this dataset -- see the failures "
            "section before trusting any further autonomous progress."
        )
    if not noise_floor.get("has_data"):
        recommendations.append(
            "No repeated-config runs exist yet for this dataset, so reported deltas between "
            "configs cannot be distinguished from ordinary run-to-run noise with any statistical "
            "confidence -- treat ranking by val_metric alone as directional, not conclusive."
        )
    if not recommendations:
        recommendations.append("No specific action items -- nothing here needs engineer attention right now.")
    return recommendations


def generate_report(dataset: str) -> dict:
    """Build one GoAudit report for `dataset` from its current ledger/
    approvals/adaptation/alerts state. Read-only -- never mutates
    anything it reads.
    """
    history = ledger.get_history(dataset)
    progress = _progress_section(history)
    approvals = _approvals_section(dataset)
    failure_alerts = alerts.check_alerts(dataset)
    noise_floor = statistics.estimate_noise_floor(dataset)

    return {
        "dataset": dataset,
        "generated_at": _now(),
        "progress": progress,
        "resource_usage": _resource_usage_section(dataset),
        "decisions": _decisions_section(history),
        "adaptations": _adaptations_section(dataset),
        "approvals": approvals,
        "failures": failure_alerts,
        "statistical_confidence": noise_floor,
        "recommendations": _recommendations(progress, approvals, failure_alerts, noise_floor),
    }


def format_report(report: dict) -> str:
    """Render a generate_report() dict as readable plain text."""
    lines = [
        f"GoAudit report -- {report['dataset']}",
        f"Generated: {report['generated_at']}",
        "",
        "## Progress",
        f"  Runs so far: {report['progress']['n_runs']} ({report['progress']['n_scored_runs']} scored)",
        f"  Best run: {report['progress']['best_run_id']} (val_metric={report['progress']['best_val_metric']})",
        f"  Trend: {report['progress']['trend']}",
        "",
        "## Resource Usage",
        f"  GPU-minutes used: {report['resource_usage']['total_gpu_minutes_used']:.1f} "
        f"across {report['resource_usage']['n_runs_with_recorded_cost']}/{report['resource_usage']['n_runs']} run(s) with recorded cost "
        f"(avg {report['resource_usage']['avg_gpu_minutes_per_run']:.1f} min/run)"
        if report["resource_usage"]["avg_gpu_minutes_per_run"] is not None
        else "  GPU-minutes used: no runs with recorded cost yet",
        "",
        "## Decisions",
    ]
    if report["decisions"]:
        for d in report["decisions"]:
            lines.append(
                f"  - {d['run_id']}: {d['decision']} (val_metric={d['val_metric']}, "
                f"delta_vs_parent={d['delta_vs_parent']}) -- {d['rationale'] or 'no rationale recorded'}"
            )
    else:
        lines.append("  (no runs yet)")

    lines += ["", "## Adaptations after rejected/blocked proposals"]
    if report["adaptations"]:
        for a in report["adaptations"]:
            lines.append(f"  - rejected: {a['rejected_proposal']} -> instead: {a['new_plan']} ({a['rationale']})")
    else:
        lines.append("  (none recorded)")

    lines += ["", "## Engineer approvals"]
    if report["approvals"]:
        for p in report["approvals"]:
            lines.append(f"  - {p['field']}={p['value']!r}: {p['status']} ({p['decision_reason'] or 'no reason given'})")
    else:
        lines.append("  (no gated proposals yet)")

    lines += ["", "## Failures / active alerts"]
    if report["failures"]:
        for a in report["failures"]:
            lines.append(f"  - [{a['severity'].upper()}] {a['message']}")
    else:
        lines.append("  (none active)")

    lines += ["", "## Statistical confidence", f"  {report['statistical_confidence']['note']}"]

    lines += ["", "## Recommendations"]
    for r in report["recommendations"]:
        lines.append(f"  - {r}")

    return "\n".join(lines)


def _report_dir(dataset: str) -> Path:
    return GOAUDIT_DIR / dataset


def write_report(dataset: str, report: dict) -> Path:
    """Persist `report` as a new timestamped snapshot -- never overwrites
    a previous report, so repeated calls build an audit trail.
    """
    report_dir = _report_dir(dataset)
    report_dir.mkdir(parents=True, exist_ok=True)
    safe_timestamp = report["generated_at"].replace(":", "-")
    path = report_dir / f"{safe_timestamp}.json"

    fd, tmp_path = tempfile.mkstemp(dir=report_dir, prefix=".goaudit_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path


def list_reports(dataset: str) -> list:
    """Timestamps (oldest first) of every GoAudit report ever generated for `dataset`."""
    report_dir = _report_dir(dataset)
    if not report_dir.is_dir():
        return []
    return sorted(p.stem for p in report_dir.glob("*.json"))


def read_report(dataset: str, generated_at: str) -> Optional[dict]:
    """Read back one previously-written report by its (colon-safe) timestamp stem."""
    path = _report_dir(dataset) / f"{generated_at}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
