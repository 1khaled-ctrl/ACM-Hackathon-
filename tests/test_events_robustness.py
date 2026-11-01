"""logs/events.jsonl robustness: a process killed mid-write (OOM kill,
timeout SIGKILL) can leave a truncated last line. history.py and the
dashboard's reader already skipped such lines, but logging_utils.read_events
-- used by GoAudit, failure alerts and promotion -- raised JSONDecodeError
on it, so one torn line broke the generate_goaudit_report tool.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import alerts, logging_utils  # noqa: E402


@pytest.fixture
def events_file(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    monkeypatch.setattr(logging_utils, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", path)
    return path


def test_truncated_and_garbage_lines_are_skipped(events_file):
    logging_utils.log_event("training_failed", dataset="m-cashew-plant", returncode=-9)
    with open(events_file, "a", encoding="utf-8") as f:
        f.write('{"event": "training_timed_out", "dataset": "m-cashew-pl\n')  # torn write
        f.write("not json at all\n")
        f.write("[1, 2, 3]\n")  # valid JSON, not an event object
    logging_utils.log_event("training_timed_out", dataset="m-cashew-plant", timeout_seconds=300)

    events = logging_utils.read_events(dataset="m-cashew-plant")
    assert [e["event"] for e in events] == ["training_failed", "training_timed_out"]


def test_alerts_still_work_with_a_torn_line(events_file):
    logging_utils.log_event("training_failed", dataset="m-SA-crop-type", returncode=1)
    with open(events_file, "a", encoding="utf-8") as f:
        f.write('{"event": "training_fai')
    fired = alerts.check_alerts("m-SA-crop-type")
    assert isinstance(fired, list)
