"""High-level status helpers built on top of logging_utils.write_status/read_status.

status.json is the single "what's happening right now" snapshot the agent
(or a dashboard) reads, as opposed to events.jsonl (full history) or
ledger.jsonl (per-run records). Fields tracked here:

  stage         - current phase, e.g. "exploring", "training", "evaluating"
  action        - the specific thing being done right now
  from          - where this step was triggered from (e.g. a run_id, "agent")
  where         - dataset / task the action applies to
  iteration     - which iteration of the train/eval loop this is
  best_so_far   - {"run_id", "dataset", "val_metric"} for the best run so far
  last_event    - short description of the most recently logged event
"""

from typing import Any, Optional

from gotrain.logging_utils import read_status, write_status


def set_stage(stage: str, action: Optional[str] = None) -> dict:
    fields = {"stage": stage}
    if action is not None:
        fields["action"] = action
    return write_status(**fields)


def set_action(
    action: str,
    from_: Optional[str] = None,
    where: Optional[str] = None,
    iteration: Optional[int] = None,
) -> dict:
    """Record the action currently being taken.

    `from_` identifies what triggered the action (e.g. a parent run_id or
    "agent"/"user"); `where` is the dataset/task it applies to.
    """
    fields: dict[str, Any] = {"action": action}
    if from_ is not None:
        fields["from"] = from_
    if where is not None:
        fields["where"] = where
    if iteration is not None:
        fields["iteration"] = iteration
    return write_status(**fields)


def set_best_so_far(run_id: str, dataset: str, val_metric: Any) -> dict:
    return write_status(
        best_so_far={"run_id": run_id, "dataset": dataset, "val_metric": val_metric}
    )


def set_last_event(description: str) -> dict:
    return write_status(last_event=description)


def get_status() -> dict:
    return read_status()
