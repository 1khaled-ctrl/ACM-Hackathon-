"""Agent-facing wrappers around gotrain.experiment_planner (evidence-based
experiment planning). See gotrain/tools/geoguard.py for the identical
logging pattern.
"""

from typing import Optional

from gotrain import experiment_planner
from gotrain.logging_utils import log_event


def get_training_diagnostics(dataset: str, run_id: str) -> dict:
    return experiment_planner.get_training_diagnostics(dataset, run_id)


def recommend_experiments(dataset: str, run_id: str, remaining_run_budget: Optional[int] = None) -> dict:
    result = experiment_planner.recommend_experiments(dataset, run_id, remaining_run_budget=remaining_run_budget)
    log_event(
        "experiments_recommended", dataset=dataset, run_id=run_id,
        diagnosis=result["diagnosis"], n_candidates=len(result["candidate_experiments"]),
    )
    return result
