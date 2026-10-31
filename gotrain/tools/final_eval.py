"""The ONLY place in this codebase allowed to touch the test split.

final_test_eval runs a trained model's evaluation on the test split for one
dataset, exactly once. A lock file (logs/test_lock_<dataset>.json) is
written atomically on first use; any subsequent call for the same dataset
is refused rather than silently re-running — re-evaluating on test defeats
the point of holding it out, so this has to fail loudly, not retry quietly.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gotrain.logging_utils import LOGS_DIR, log_evaluation

TEST_LOCK_TEMPLATE = "test_lock_{dataset}.json"


class TestAlreadyEvaluatedError(RuntimeError):
    pass


def _lock_path(dataset: str) -> Path:
    return LOGS_DIR / TEST_LOCK_TEMPLATE.format(dataset=dataset)


def _acquire_lock(dataset: str, run_id: str) -> None:
    """Atomically create the lock file; raises if it already exists.

    Uses O_CREAT|O_EXCL so a concurrent second call can't race past the
    exists-check between check and write.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path(dataset)
    payload = {
        "dataset": dataset,
        "run_id": run_id,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        raise TestAlreadyEvaluatedError(
            f"Test split for {dataset!r} was already evaluated "
            f"(run_id={existing.get('run_id')!r} at {existing.get('evaluated_at')!r}). "
            "The test split may only be evaluated once per dataset."
        ) from None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def final_test_eval(dataset: str, run_id: str, eval_fn) -> dict:
    """Run `eval_fn` (a no-arg callable returning a metrics dict) against the
    test split for `dataset`, tied to `run_id` (the chosen final model),
    exactly once.

    Raises TestAlreadyEvaluatedError if this dataset's test split has
    already been evaluated. `eval_fn` is the caller's responsibility to
    build — it must load the test split itself; this function only
    enforces the one-time lock and logs the result.
    """
    _acquire_lock(dataset, run_id)
    metrics = eval_fn()
    record = log_evaluation(task=dataset, split="test", metrics=metrics, model=run_id)
    return record
