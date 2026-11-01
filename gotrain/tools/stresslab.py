"""Agent-facing wrapper around gotrain.stresslab: logs the call and
current status the same way every other tool in this package does (see
gotrain/tools/dataiq.py for the identical pattern).
"""

from typing import Optional

from gotrain import status
from gotrain.logging_utils import log_event
from gotrain.stresslab import runner as stresslab_runner


def run_stresslab(dataset: str, task_type: str, run_id: str, max_samples: Optional[int] = None) -> dict:
    """Robustness battery against `run_id`'s saved checkpoint. Needs
    torch + the real downloaded dataset -- see
    gotrain/stresslab/runner.py's module docstring for the full caveat.
    """
    status.set_action("evaluating robustness under perturbations (StressLab)", where=dataset)
    log_event("stresslab_started", dataset=dataset, task_type=task_type, run_id=run_id)

    if task_type == "classification":
        summary = stresslab_runner.run_stresslab_classification(dataset, run_id, max_samples=max_samples)
    elif task_type == "segmentation":
        summary = stresslab_runner.run_stresslab_segmentation(dataset, run_id, max_samples=max_samples)
    else:
        raise ValueError(f"task_type must be 'classification' or 'segmentation', got {task_type!r}")

    status.set_last_event(f"StressLab for {dataset}/{run_id}: {summary['overall_verdict']}")
    return summary
