"""Data-quality decision records (DataIQ).

Shares its review queue, audit trail, and decide()/list_proposals()
machinery with gotrain.approval -- the SAME dashboard mechanism an
engineer already uses to approve/reject config changes -- per the design
brief's explicit instruction to extend approvals to data modifications
rather than build a parallel governance system. Every record created
here carries "kind": "data_action" so gotrain.approval.list_proposals and
the dashboard can tell it apart from a config-gate proposal (an older
record with no "kind" key is always treated as a config gate, for
backward compatibility with entries written before this module existed).

Decisions (approve/reject) are made by calling gotrain.approval.decide()
directly with one of this module's keys -- there is nothing
data-specific about deciding, only about proposing.
"""

from datetime import datetime, timezone
from typing import Optional

from gotrain import approval

ACTIONS = [
    "KEEP",
    "FLAG_FOR_REVIEW",
    "EXCLUDE_FROM_TRAINING_CANDIDATE",
    "CORRECT_LABEL_CANDIDATE",
    "QUARANTINE_INVALID_SAMPLE",
]

# Only an objectively corrupted/unreadable sample (gotrain.dataiq.
# diagnostics severity "high" structural findings -- NaN/Inf pixels, a
# constant image, an out-of-range label/mask) can be auto-approved under
# policy. A suspected mislabel or a "this example looks hard" exclusion
# is NEVER auto-approved -- those always require a human, since disagreeing
# with the model is not proof the label (or the sample) is wrong.
AUTO_APPROVED_ACTIONS = {"QUARANTINE_INVALID_SAMPLE"}

NO_APPROVAL_NEEDED_ACTIONS = {"KEEP", "FLAG_FOR_REVIEW"}  # informational only; nothing changes yet


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _key(dataset: str, sample_id: int, action: str) -> str:
    return f"{dataset}:data:{sample_id}:{action}"


def propose_action(
    dataset: str,
    sample_id: int,
    action: str,
    evidence: dict,
    severity: str,
    reason: str,
    confidence: Optional[float] = None,
) -> dict:
    """Record a proposed data-quality action for `sample_id`.

    KEEP/FLAG_FOR_REVIEW are informational -- recorded for the audit
    trail, but nothing about training changes as a result (no approval
    gate). QUARANTINE_INVALID_SAMPLE is auto-approved under policy
    (objectively corrupted samples need no human review to exclude).
    EXCLUDE_FROM_TRAINING_CANDIDATE and CORRECT_LABEL_CANDIDATE are
    always PENDING -- exactly the "ambiguous label corrections and
    potentially legitimate hard-sample exclusions" the design brief
    says must never be automatic.

    Re-proposing an already-decided (dataset, sample_id, action) returns
    the EXISTING record unchanged -- same non-reset-on-resubmission
    guarantee as gotrain.approval.request_approval.
    """
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}, got {action!r}")

    key = _key(dataset, sample_id, action)
    state = approval._read_state()
    if key in state:
        return state[key]

    if action in AUTO_APPROVED_ACTIONS:
        status, decided_at, decided_by, decision_reason = (
            approval.APPROVED, _now(), "policy", "auto-approved: objectively corrupted/invalid sample",
        )
    elif action in NO_APPROVAL_NEEDED_ACTIONS:
        status, decided_at, decided_by, decision_reason = approval.APPROVED, _now(), "system", "informational only"
    else:
        status, decided_at, decided_by, decision_reason = approval.PENDING, None, None, None

    record = {
        "key": key,
        "kind": "data_action",
        "dataset": dataset,
        "sample_id": sample_id,
        "field": "sample_action",  # shape-compatible with approval's generic record
        "value": action,
        "action": action,
        "evidence": evidence,
        "severity": severity,
        "confidence": confidence,
        "reason": reason,
        "status": status,
        "requested_at": _now(),
        "decided_at": decided_at,
        "decided_by": decided_by,
        "decision_reason": decision_reason,
    }
    state[key] = record
    approval._write_state(state)
    approval._append_audit("data_action_proposed", **record)
    return record


def get_approved_action(dataset: str, sample_id: int) -> Optional[str]:
    """Return the APPROVED exclusion-eligible action for `sample_id`
    (EXCLUDE_FROM_TRAINING_CANDIDATE or QUARANTINE_INVALID_SAMPLE -- the
    only two actions gotrain.dataiq.manifest.create_manifest cares
    about), or None if neither is approved.
    """
    state = approval._read_state()
    for action in ("EXCLUDE_FROM_TRAINING_CANDIDATE", "QUARANTINE_INVALID_SAMPLE"):
        record = state.get(_key(dataset, sample_id, action))
        if record is not None and record["status"] == approval.APPROVED:
            return action
    return None


def list_data_proposals(dataset: Optional[str] = None, status: Optional[str] = None) -> list:
    proposals = approval.list_proposals(status=status, kind="data_action")
    if dataset is not None:
        proposals = [p for p in proposals if p["dataset"] == dataset]
    return proposals
