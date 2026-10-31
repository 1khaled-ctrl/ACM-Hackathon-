"""Persists/reads one GeoGuard report per dataset, next to the rest of
GoTrain's logs: logs/geoguard/<dataset>.json. Same atomic-write pattern as
gotrain/logging_utils.py (tempfile + os.replace) so a reader (dashboard or
the agent) never sees a half-written report.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from gotrain.logging_utils import LOGS_DIR

GEOGUARD_DIR = LOGS_DIR / "geoguard"


def _report_path(dataset: str) -> Path:
    return GEOGUARD_DIR / f"{dataset}.json"


def write_report(dataset: str, report: dict) -> Path:
    GEOGUARD_DIR.mkdir(parents=True, exist_ok=True)
    path = _report_path(dataset)
    fd, tmp_path = tempfile.mkstemp(dir=GEOGUARD_DIR, prefix=".geoguard_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path


def read_report(dataset: str) -> Optional[dict]:
    path = _report_path(dataset)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_summary(report: dict) -> str:
    """Short human-readable rendering of a report -- used by the CLI tool
    result (what the agent/a human reads) and reusable by the dashboard.
    """
    lines = [f"GeoGuard report for {report['dataset']} ({report['task_type']}):"]

    findings = report.get("findings") or []
    if findings:
        lines.append("Findings:")
        lines.extend(f"  - {f}" for f in findings)
    else:
        lines.append("Findings: none notable.")

    config = report.get("recommended_config") or {}
    if config:
        lines.append(f"Recommended config additions: {json.dumps(config)}")
        lines.append("Rationale:")
        lines.extend(f"  - {r}" for r in report.get("rationale", []))
    else:
        lines.append("Recommended config additions: none -- defaults look appropriate.")

    return "\n".join(lines)
