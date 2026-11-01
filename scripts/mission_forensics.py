#!/usr/bin/env python
"""Print the evidence behind each mission's outcome -- run on the machine
whose logs/ and runs/ you want to inspect (e.g. SageMaker).

    python scripts/mission_forensics.py                 # every session
    python scripts/mission_forensics.py m-SA-crop-type  # one dataset

Per session: status, stored stop reason, runs/timeouts attributed to it,
every training_failed / training_timed_out event in its window (exit code +
the most informative stderr line), and the tail of every runs/<dataset>/*/
train.log written during the session -- a killed run never reaches the
ledger, but its train.log shows how far it got. Reads logs only; never
touches the test split. ASCII output.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gotrain import history  # noqa: E402

TAIL_LINES = 12


def _ascii(text) -> str:
    return str(text).encode("ascii", "replace").decode()


def _ts(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def main() -> int:
    only = set(sys.argv[1:])
    sessions = [s for s in history.list_sessions() if not only or s.get("dataset") in only]
    if not sessions:
        print("No mission sessions found under", history.HISTORY_DIR)
        return 0
    for s in reversed(sessions):  # oldest first
        start, end = history._parse(s.get("started_at")), history._parse(s.get("finished_at"))
        print("=" * 78)
        print(_ascii(f"{s.get('dataset')}  {s.get('status')}  {s.get('runs_completed', 0)}/{s.get('runs_requested')} runs  "
                     f"timed_out={s.get('timed_out_runs', 0)}  session={s.get('session_id')}"))
        print(_ascii(f"  started {s.get('started_at')}  finished {s.get('finished_at')}"))
        print(_ascii(f"  stored stop_reason: {s.get('stop_reason')}"))
        print(_ascii(f"  re-derived reason : {history.derive_stop_reason(s.get('dataset'), s.get('started_at'), s.get('finished_at'))}"))
        print(_ascii(f"  final submitted   : {s.get('final_model_run_id')}   best run: {s.get('best_run_id')} "
                     f"val={s.get('best_val_metric')}"))
        for e in history._windowed_events(start, end):
            if e.get("event") in ("training_failed", "training_timed_out", "stresslab_failed", "stresslab_unavailable",
                                  "final_model_submitted"):
                detail = {k: e.get(k) for k in ("returncode", "reason", "timeout_seconds", "error", "run_id") if e.get(k) is not None}
                print(_ascii(f"  - {e.get('timestamp')} {e['event']} {json.dumps(detail)}"))
                if e.get("stderr"):
                    print(_ascii(f"      stderr: {history._most_informative_stderr_line(e['stderr'])}"))
        runs_dir = REPO_ROOT / "runs" / str(s.get("dataset"))
        if runs_dir.is_dir() and start is not None:
            for log in sorted(runs_dir.glob("*/train.log"), key=os.path.getmtime):
                written = _ts(log)
                if written < start or (end is not None and written > end):
                    continue
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
                print(_ascii(f"  train.log {log.parent.name} (last write {written.isoformat()}):"))
                for line in lines[-TAIL_LINES:]:
                    print(_ascii(f"      {line[:200]}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
