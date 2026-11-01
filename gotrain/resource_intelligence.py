"""Resource intelligence: lets the agent reason about its own run/tool-call
budget and the real wall-clock cost of a candidate config BEFORE spending
a run on it, instead of only finding out reactively -- a budget-exhausted
tool rejection, or a timeout-killed run with no usable checkpoint at all.

Two separate concerns, both built from data this project already records:
  - estimate_run_cost / check_fits_in_run_budget: how long will THIS
    config likely take, based on this dataset's own measured
    (gpu_minutes / epochs) history -- never a generic made-up number;
    explicitly says so when there's no history yet to estimate from.
  - summarize_resource_usage: how much of this dataset's resources
    (GPU-minutes, runs) has actually been spent so far.

Pure ledger reads -- no torch needed, fully testable in this environment.

The live run/tool-call/data-experiment COUNTERS themselves (as opposed to
max_runs/max_tool_calls, which are just configuration) live only inside
gotrain.agent.LimitsAndLoggingHooks, in-memory, for the duration of one
agent process -- this module has no way to see those directly. gotrain.agent
mirrors them into logs/status.json's "resource_budget" field on every tool
call (gotrain.status.set_resource_budget) specifically so
gotrain.tools.resource_intelligence's get_resource_status tool (and the
dashboard) can read them back from outside that process.
"""

from typing import Optional

from gotrain import ledger

# A judgment call, like calibration.py's ECE thresholds: below this
# fraction of the per-run wall-clock budget, a config is comfortably
# safe; above 1.0 it's expected to be killed by the timeout before
# finishing, wasting the run slot on no usable checkpoint.
SAFE_BUDGET_FRACTION = 0.8


def estimate_run_cost(dataset: str, config: dict, default_epochs: int = 2) -> dict:
    """Predicts wall-clock GPU-minutes for `config` from this dataset's
    OWN measured (gpu_minutes / epochs) history -- preferring runs that
    used the same backbone/architecture (the single biggest cost driver)
    when there's enough history to do that, falling back to every run's
    average otherwise. Returns has_data=False (never a fabricated
    number) when this dataset has no runs with both fields recorded yet.
    """
    history = ledger.get_history(dataset)
    candidate_backbone = config.get("backbone") or config.get("architecture")

    def _samples(require_same_backbone: bool) -> list:
        samples = []
        for record in history:
            gpu_minutes = record.get("gpu_minutes")
            run_config = record.get("config") or {}
            epochs = run_config.get("epochs")
            if not (isinstance(gpu_minutes, (int, float)) and isinstance(epochs, (int, float)) and epochs > 0):
                continue
            if require_same_backbone:
                run_backbone = run_config.get("backbone") or run_config.get("architecture")
                if run_backbone != candidate_backbone:
                    continue
            samples.append(gpu_minutes / epochs)
        return samples

    samples = _samples(require_same_backbone=True) if candidate_backbone else []
    matched_backbone = bool(samples)
    if not samples:
        samples = _samples(require_same_backbone=False)

    if not samples:
        return {
            "has_data": False, "estimated_minutes": None, "minutes_per_epoch": None,
            "n_historical_runs": 0, "matched_backbone": False,
            "note": "No prior run for this dataset has both gpu_minutes and epochs recorded yet "
                    "-- cannot estimate cost. The first run of any dataset is always cost-blind.",
        }

    minutes_per_epoch = sum(samples) / len(samples)
    epochs = config.get("epochs", default_epochs)
    estimated_minutes = minutes_per_epoch * epochs
    basis = "the same backbone/architecture" if matched_backbone else "all backbones (none yet match this one)"
    return {
        "has_data": True,
        "estimated_minutes": estimated_minutes,
        "minutes_per_epoch": minutes_per_epoch,
        "n_historical_runs": len(samples),
        "matched_backbone": matched_backbone,
        "note": f"Estimated from {len(samples)} prior run(s) using {basis} for this dataset, "
                f"averaging {minutes_per_epoch:.2f} GPU-minute(s)/epoch.",
    }


def check_fits_in_run_budget(cost_estimate: dict, run_minutes_budget: Optional[float]) -> dict:
    """Judges an estimate_run_cost() result against the per-run
    wall-clock budget (the `run_minutes` every launch_training call is
    capped at via timeout_seconds) -- a config estimated to exceed that
    budget risks being killed mid-training, wasting an entire run slot
    on no usable checkpoint at all.
    """
    if not cost_estimate.get("has_data"):
        return {"fits": None, "reason": "no cost estimate available yet to judge fit against."}
    if not run_minutes_budget:
        return {"fits": None, "reason": "no per-run wall-clock budget given to judge fit against."}

    estimated = cost_estimate["estimated_minutes"]
    ratio = estimated / run_minutes_budget
    if ratio <= SAFE_BUDGET_FRACTION:
        fits, note = True, (
            f"Estimated {estimated:.1f} min fits comfortably within the "
            f"{run_minutes_budget:.0f}-minute per-run budget."
        )
    elif ratio <= 1.0:
        fits, note = True, (
            f"Estimated {estimated:.1f} min is close to the {run_minutes_budget:.0f}-minute "
            "per-run budget -- tight, but should finish."
        )
    else:
        fits, note = False, (
            f"Estimated {estimated:.1f} min EXCEEDS the {run_minutes_budget:.0f}-minute per-run "
            "budget -- this config risks being killed by the timeout before finishing, wasting "
            "a run slot on no usable checkpoint."
        )
    return {"fits": fits, "ratio": ratio, "note": note}


def summarize_resource_usage(dataset: str) -> dict:
    """Real, measured resource usage so far for `dataset`: total
    GPU-minutes actually spent (sum of every completed run's recorded
    gpu_minutes) and how many runs that covers -- distinct from
    estimate_run_cost's forward-looking PREDICTION for a not-yet-run
    config.
    """
    history = ledger.get_history(dataset)
    gpu_minutes_values = [r["gpu_minutes"] for r in history if isinstance(r.get("gpu_minutes"), (int, float))]
    total_gpu_minutes = sum(gpu_minutes_values)
    return {
        "n_runs": len(history),
        "n_runs_with_recorded_cost": len(gpu_minutes_values),
        "total_gpu_minutes_used": total_gpu_minutes,
        "avg_gpu_minutes_per_run": (total_gpu_minutes / len(gpu_minutes_values)) if gpu_minutes_values else None,
    }
