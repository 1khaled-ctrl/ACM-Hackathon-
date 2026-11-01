"""Distinguishes a genuine validation-metric improvement from ordinary
run-to-run noise (different random seed, different dataloader shuffle
order, etc.) -- rather than trusting any positive delta_vs_parent at face
value.

GoTrain has no repeated-seed runs by default (every run explores a new
config to spend its limited run budget efficiently), so there is usually
no direct evidence of how much a validation metric moves on IDENTICAL
configs alone. estimate_noise_floor looks for whatever repeated-config
runs DO exist in the ledger (two runs that only differ by `seed`) and
derives a noise floor from them; is_improvement_significant refuses to
call a verdict when there isn't enough evidence, rather than fabricating
a false sense of statistical confidence from a single run.

This is deliberately a heuristic built from this project's own actual
run history, not a textbook significance test (that would need multiple
repeated runs per config, which costs run budget this project doesn't
spend by default) -- see its "note"/"reason" fields, which always say
exactly what evidence (or lack of it) the verdict is based on.
"""

import json
from typing import Optional

from gotrain import ledger


def estimate_noise_floor(dataset: str) -> dict:
    """Groups this dataset's ledger runs by config (excluding `seed`) and,
    for any group with 2+ runs, computes how much val_metric varied
    despite using the identical non-seed config. The noise floor is the
    largest such spread observed -- a conservative (not optimistic)
    estimate, since claiming a delta is "real" when it's actually smaller
    than observed seed noise would be the worse mistake to make here.
    """
    groups: dict = {}
    for record in ledger.get_history(dataset):
        val_metric = record.get("val_metric")
        if not isinstance(val_metric, (int, float)):
            continue
        config = record.get("config") or {}
        key = json.dumps(
            {k: v for k, v in config.items() if k != "seed"}, sort_keys=True, default=str
        )
        groups.setdefault(key, []).append(val_metric)

    repeated = {key: values for key, values in groups.items() if len(values) >= 2}
    if not repeated:
        return {
            "has_data": False,
            "noise_floor": None,
            "n_repeated_configs": 0,
            "note": (
                "No two runs for this dataset share an identical config (aside "
                "from seed) yet, so there is no direct evidence of how much "
                "val_metric moves from randomness alone. Treat any delta "
                "between DIFFERENT configs as suggestive, not statistically "
                "confirmed, until repeated-seed evidence exists."
            ),
        }

    spreads = [max(values) - min(values) for values in repeated.values()]
    noise_floor = max(spreads)
    return {
        "has_data": True,
        "noise_floor": noise_floor,
        "n_repeated_configs": len(repeated),
        "spreads": spreads,
        "note": (
            f"Estimated from {len(repeated)} repeated-config group(s) in this "
            f"dataset's run history; the largest observed same-config spread "
            f"was {noise_floor:.4f}. A delta smaller than this cannot be "
            "distinguished from seed noise alone."
        ),
    }


def is_improvement_significant(delta: Optional[float], noise_floor_result: dict) -> dict:
    """Judges one delta_vs_parent against a noise_floor_result from
    estimate_noise_floor. Returns significant=None (not True or False)
    when there isn't enough evidence to judge at all -- a missing verdict
    is more honest than guessing.
    """
    if delta is None:
        return {"significant": None, "reason": "no delta_vs_parent to evaluate (no parent run, or non-numeric metric)."}
    if not noise_floor_result.get("has_data"):
        return {
            "significant": None,
            "reason": (
                "no repeated-config evidence yet for this dataset -- cannot "
                "distinguish this delta from seed noise with any confidence; "
                f"delta={delta:+.4f} should be treated as suggestive only."
            ),
        }

    noise_floor = noise_floor_result["noise_floor"]
    significant = abs(delta) > noise_floor
    return {
        "significant": significant,
        "delta": delta,
        "noise_floor": noise_floor,
        "reason": (
            f"delta magnitude {abs(delta):.4f} "
            f"{'exceeds' if significant else 'does not exceed'} the largest "
            f"observed same-config spread ({noise_floor:.4f}) for this dataset."
        ),
    }
