"""Controlled model promotion: activates a dataset's agent-submitted model
as the "active" production checkpoint, but ONLY after both (1) the
one-time test-split evaluation has actually been run for it
(gotrain.tools.final_eval.final_test_eval, via run_final_eval.py) and (2)
the same integrity+inference verification chain gotrain.rollback already
uses passes for its checkpoint.

This is deliberately a distinct, named operation from gotrain.rollback's
generic "activate any registered version" -- rollback can point the
active pointer at ANY version an engineer picks (including rolling
forward to the newest one), with no requirement that it was ever
test-evaluated. promote_submitted_model is the narrower, gated path: it
refuses to activate a model that hasn't cleared the one-time test split
yet, so "promotion" always means "a model that has been fully validated,"
not just "a model an engineer clicked."

Like gotrain.rollback, this is an engineer-only operation -- no agent
tool ever calls it.
"""

from typing import Optional

from gotrain import ledger, rollback
from gotrain.logging_utils import log_event, read_events


class NoSubmittedModelError(RuntimeError):
    """No run has been submitted as final for this dataset yet."""


class ModelNotTestEvaluatedError(RuntimeError):
    """The submitted run has not been through the one-time test-split
    evaluation yet (run_final_eval.py) -- promotion requires that
    validation step to have already happened; it is not a shortcut
    around it.
    """


def _find_test_result(run_id: str) -> Optional[dict]:
    for event in read_events(event="evaluation"):
        if event.get("split") == "test" and event.get("model") == run_id:
            return event
    return None


def _find_submitted_run(dataset: str) -> dict:
    submitted = [r for r in ledger.get_history(dataset) if r.get("submitted_final")]
    if not submitted:
        raise NoSubmittedModelError(
            f"No run has been submitted as final for {dataset!r} yet -- nothing to promote."
        )
    return sorted(submitted, key=lambda r: r["timestamp"])[-1]


def promote_submitted_model(dataset: str, task_type: str, requested_by: str = "engineer") -> dict:
    """Find the run submitted as final for `dataset`, require it has a
    test-split result on record, then activate its registry version via
    gotrain.rollback's own verified-activation machinery (integrity +
    inference checks). Raises NoSubmittedModelError /
    ModelNotTestEvaluatedError / rollback.RollbackVerificationError /
    rollback.RollbackConflictError without touching the active-version
    pointer on any failure -- same "never flip the pointer and hope"
    guarantee rollback_to_version already provides.
    """
    run_record = _find_submitted_run(dataset)
    run_id = run_record["run_id"]

    version_id = run_record.get("version_id")
    if not version_id:
        raise ModelNotTestEvaluatedError(
            f"submitted run {run_id!r} for {dataset!r} has no registered checkpoint version "
            "to promote (version_id is empty)."
        )

    test_result = _find_test_result(run_id)
    if test_result is None:
        raise ModelNotTestEvaluatedError(
            f"submitted run {run_id!r} for {dataset!r} has not been evaluated on the test "
            "split yet -- run run_final_eval.py before promoting it to active."
        )

    result = rollback.execute_rollback(dataset, version_id, task_type, requested_by=requested_by)
    log_event(
        "model_promoted", dataset=dataset, run_id=run_id, version=version_id,
        test_metrics=test_result.get("metrics"), requested_by=requested_by,
    )
    return result
