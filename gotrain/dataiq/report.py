"""Persists/reads one DataIQ structural-scan report per dataset, next to
GeoGuard's own report (gotrain/geoguard/report.py) -- same atomic
tempfile+os.replace pattern as the rest of gotrain's logs.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from gotrain.logging_utils import LOGS_DIR

DATAIQ_REPORTS_DIR = LOGS_DIR / "dataiq" / "structural_scan"


def _report_path(dataset: str) -> Path:
    return DATAIQ_REPORTS_DIR / f"{dataset}.json"


def write_report(dataset: str, report: dict) -> Path:
    DATAIQ_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _report_path(dataset)
    fd, tmp_path = tempfile.mkstemp(dir=DATAIQ_REPORTS_DIR, prefix=".dataiq_", suffix=".tmp")
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
