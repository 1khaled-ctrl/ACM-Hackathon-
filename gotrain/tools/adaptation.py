"""Adaptation records: when a proposal gets rejected (an approval-gated
config value, an unapproved data-quality exclusion), the agent is
expected to narrate what it's trying instead and why -- not just silently
fall back. This makes that narration a first-class, logged artifact
instead of something only implicit in the next tool call's arguments.

Reuses logs/events.jsonl (gotrain.logging_utils.log_event) rather than
introducing a new store -- an adaptation is conceptually just another
event in the same stream the dashboard's activity feed already reads,
tagged distinctly ("adaptation_recorded") so it can be told apart from a
plain tool call.
"""

from datetime import datetime, timezone

from gotrain import status
from gotrain.logging_utils import log_event


def record_adaptation(dataset: str, rejected_proposal: str, new_plan: str, rationale: str) -> dict:
    """Log that a proposal was rejected/blocked and state what's being
    tried instead and why. Called by the agent itself -- never inferred
    or generated on its behalf -- so the record reflects the agent's own
    stated reasoning, not a system-generated guess at one.
    """
    record = {
        "dataset": dataset,
        "rejected_proposal": rejected_proposal,
        "new_plan": new_plan,
        "rationale": rationale,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    log_event("adaptation_recorded", **record)
    status.set_last_event(f"Adapted after rejection on {dataset}: {new_plan}")
    return record
