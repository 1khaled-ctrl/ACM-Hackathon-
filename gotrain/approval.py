"""Code-enforced approval gate for resource/risk-sensitive config choices
(architecture/backbone/encoder swaps -- the kind of change that costs real
GPU budget or wall-clock time if it turns out to be wrong).

Worked example from the design brief: GoTrain proposes switching from
resnet18 to resnet50; the engineer rejects it for GPU limits. The system
must record the rejection, prevent execution, and refuse to let the agent
bypass it by resubmitting under a different rationale. That last part is
true by construction here, not by convention: gate state is keyed by
(dataset, field, value) -- never by a run_id, call description, or
rationale text -- so no amount of rephrasing changes the key being looked
up. See test_approval.py::test_rejected_value_cannot_be_bypassed_by_resubmission.

Persisted to logs/approvals_state.json (current status per key, mutable)
and an append-only logs/approvals.jsonl (every request/decision, for
audit). Same atomic tempfile+os.replace write pattern as the rest of
gotrain's logs.

No agent tool calls request_approval or decide() directly -- requests are
created as a side effect of check_gate (called from
gotrain/tools/training.py, before any subprocess runs), and decisions are
only ever made from the dashboard, by a human.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from gotrain.config_schema import PreflightValidationError
from gotrain.logging_utils import LOGS_DIR

APPROVALS_STATE_PATH = LOGS_DIR / "approvals_state.json"
APPROVALS_LOG_PATH = LOGS_DIR / "approvals.jsonl"

# Config fields treated as risk/resource-sensitive enough to require
# engineer sign-off before a NEW value is used. What's not listed here
# (lr, batch_size, loss, augmentation, optimizer, ...) remains fully
# autonomous -- gating every field would defeat the point of an autonomous
# agent; gating none would defeat the point of engineer oversight.
GATED_FIELDS_BY_TASK_TYPE = {
    "classification": ["backbone"],
    "segmentation": ["architecture", "encoder"],
}

# The training templates' own DEFAULTS (see gotrain/templates/train_*.py)
# are pre-approved -- they're the conservative, known-safe starting point
# every run falls back to. Approval is only required to *move away* from
# the default, matching the design brief's framing ("GoTrain proposes
# SWITCHING from ResNet-18 to ResNet-50" -- the gate is on the switch).
PRE_APPROVED_DEFAULTS = {
    "classification": {"backbone": "resnet18"},
    "segmentation": {"architecture": "unet", "encoder": "resnet34"},
}

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"


class ApprovalRequiredError(PreflightValidationError):
    """Raised when a gated config value has no decided approval yet
    (first time seen -> a pending proposal is created on the spot),
    is still pending, or was rejected. A PreflightValidationError like
    ConfigValidationError/HypothesisValidationError: rejected before any
    subprocess is spawned, so it never counts against --max-runs.
    """


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gate_key(dataset: str, field: str, value) -> str:
    return f"{dataset}:{field}:{value}"


def _read_state() -> dict:
    if not APPROVALS_STATE_PATH.exists():
        return {}
    return json.loads(APPROVALS_STATE_PATH.read_text(encoding="utf-8"))


def _write_state(state: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=LOGS_DIR, prefix=".approvals_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, APPROVALS_STATE_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _append_audit(event: str, **fields) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": _now(), "event": event, **fields}
    with open(APPROVALS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def list_proposals(status: Optional[str] = None, kind: Optional[str] = None) -> list:
    """List review-queue entries. `kind` filters to "config_gate" (this
    module's own backbone/architecture/encoder proposals -- the default
    when a record predates the "kind" field, since every entry created
    before gotrain.dataiq existed was a config gate) or "data_action"
    (gotrain.dataiq.decision's sample-level proposals, sharing this same
    queue/audit trail per the design brief's instruction to extend
    approvals to data modifications rather than building a parallel
    system). Omit `kind` to see everything.
    """
    proposals = list(_read_state().values())
    if kind is not None:
        proposals = [p for p in proposals if p.get("kind", "config_gate") == kind]
    if status is not None:
        proposals = [p for p in proposals if p["status"] == status]
    return sorted(proposals, key=lambda p: p["requested_at"])


def request_approval(dataset: str, task_type: str, field: str, value, reason: str) -> dict:
    """Create a new pending proposal for (dataset, field, value) if one
    doesn't already exist. Returns the EXISTING record unchanged if one
    does -- re-requesting an already-decided value must never reset a
    rejection back to pending.
    """
    key = _gate_key(dataset, field, value)
    state = _read_state()
    if key in state:
        return state[key]
    record = {
        "key": key, "kind": "config_gate", "dataset": dataset, "task_type": task_type, "field": field,
        "value": value, "status": PENDING, "reason": reason,
        "requested_at": _now(), "decided_at": None, "decided_by": None, "decision_reason": None,
    }
    state[key] = record
    _write_state(state)
    _append_audit("approval_requested", **record)
    return record


def decide(key: str, approve: bool, decided_by: str = "engineer", decision_reason: Optional[str] = None) -> dict:
    """Approve or reject a pending proposal. Only ever called from the
    dashboard (a human action) -- never from gotrain/tools/training.py or
    gotrain/agent.py. Re-deciding an already-decided proposal is allowed
    (an engineer can change their mind later).
    """
    state = _read_state()
    if key not in state:
        raise ValueError(f"no approval proposal found for key {key!r}")
    record = dict(state[key])
    record["status"] = APPROVED if approve else REJECTED
    record["decided_at"] = _now()
    record["decided_by"] = decided_by
    record["decision_reason"] = decision_reason
    state[key] = record
    _write_state(state)
    _append_audit("approval_decided", **record)
    return record


def check_gate(dataset: str, task_type: str, config: dict) -> None:
    """Raise ApprovalRequiredError if `config` uses a gated field's value
    that isn't approved yet (and isn't the pre-approved default). Called
    from gotrain/tools/training.py::launch_training before any training
    subprocess is spawned -- the same pre-flight slot as
    validate_config/validate_hypothesis.
    """
    gated_fields = GATED_FIELDS_BY_TASK_TYPE.get(task_type, [])
    defaults = PRE_APPROVED_DEFAULTS.get(task_type, {})

    for field in gated_fields:
        if field not in config or config[field] is None:
            continue
        value = config[field]
        if value == defaults.get(field):
            continue

        key = _gate_key(dataset, field, value)
        record = _read_state().get(key)

        if record is None:
            request_approval(
                dataset, task_type, field, value,
                reason=f"first use of {field}={value!r} for {dataset}",
            )
            raise ApprovalRequiredError(
                f"{field}={value!r} requires engineer approval before it can run "
                f"(first time this value has been proposed for {dataset}). A "
                "pending proposal has been recorded in the dashboard -- try a "
                "different, already-approved or ungated config in the meantime."
            )
        if record["status"] == PENDING:
            raise ApprovalRequiredError(
                f"{field}={value!r} is still pending engineer approval for "
                f"{dataset}. Do not keep retrying this exact value -- try "
                "something else while you wait."
            )
        if record["status"] == REJECTED:
            raise ApprovalRequiredError(
                f"{field}={value!r} was REJECTED by the engineer for {dataset} "
                f"({record.get('decision_reason') or 'no reason given'}). This "
                "will not be approved by resubmitting the same value under a "
                "different config or rationale -- propose a genuinely different "
                "value, or fall back to the default."
            )
        # record["status"] == APPROVED -> fall through, allowed.
