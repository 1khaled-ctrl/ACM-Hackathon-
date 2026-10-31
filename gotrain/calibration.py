"""Model-confidence calibration (uncertainty evaluation).

Softmax confidence is a common but known-poorly-calibrated proxy for true
uncertainty -- a model can report 95% confidence and be wrong far more
than 5% of the time, especially on an out-of-distribution or minority-
class sample. DataIQ's loss-scan (gotrain/dataiq/loss_scan.py) already
uses raw softmax confidence as a per-sample signal; this module adds the
missing check: is that confidence actually trustworthy for THIS model, on
THIS dataset? Expected Calibration Error (Guo et al. 2017) answers that
directly from data loss_scan already gathers -- no new forward passes, no
retraining.

Deliberately NOT included here: temperature scaling or any other
*recalibration* of confidence values. gotrain/dataiq/decision.py's
HIGH_CONFIDENCE_WRONG_THRESHOLD already depends on softmax confidence
meaning what it currently means; rescaling it without re-validating every
threshold that reads it is exactly the kind of coupling this module
stays out of. This reports whether a model's confidence can be trusted;
it does not try to fix it when it can't.
"""

from typing import Optional

import numpy as np

DEFAULT_N_BINS = 10
WELL_CALIBRATED_ECE = 0.05
MODERATELY_MISCALIBRATED_ECE = 0.15


def compute_calibration(confidences, correct, n_bins: int = DEFAULT_N_BINS) -> dict:
    """Expected Calibration Error (ECE) + a per-bin reliability
    breakdown, from parallel arrays: `confidences[i]` is the model's
    reported confidence (0-1) for sample i, `correct[i]` is whether its
    prediction was actually right (bool or 0/1).

    ECE = sum over confidence bins of (bin_weight * |bin_avg_confidence -
    bin_accuracy|). A well-calibrated model's confidence tracks its real
    accuracy in every bin, so ECE sits near 0; a model that's
    consistently overconfident (reports 90%+ confidence while actually
    right half the time) has a high ECE driven by that gap.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidences.shape != correct.shape:
        raise ValueError(
            f"confidences and correct must be the same shape, got {confidences.shape} vs {correct.shape}"
        )
    n = len(confidences)
    if n == 0:
        return {
            "ece": None, "n_samples": 0, "n_bins": n_bins, "mean_confidence": None,
            "mean_accuracy": None, "bias": None, "bins": [], "verdict": "no samples to calibrate",
        }

    mean_confidence = float(confidences.mean())
    mean_accuracy = float(correct.mean())
    bias = mean_confidence - mean_accuracy  # positive = overconfident, negative = underconfident

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences <= hi if i == n_bins - 1 else confidences < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"range": [float(lo), float(hi)], "count": 0, "avg_confidence": None, "accuracy": None, "gap": None})
            continue
        bin_avg_confidence = float(confidences[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        gap = abs(bin_avg_confidence - bin_accuracy)
        ece += (count / n) * gap
        bins.append({
            "range": [float(lo), float(hi)], "count": count,
            "avg_confidence": bin_avg_confidence, "accuracy": bin_accuracy, "gap": gap,
        })

    return {
        "ece": float(ece),
        "n_samples": n,
        "n_bins": n_bins,
        "mean_confidence": mean_confidence,
        "mean_accuracy": mean_accuracy,
        "bias": bias,
        "bins": bins,
        "verdict": _verdict(ece, bias),
    }


def _verdict(ece: float, bias: float) -> str:
    """A plain-language read of a raw ECE/bias pair -- the thresholds
    (5% / 15%) are a common rule of thumb for this metric, not a
    universal standard, and are named as constants above for exactly
    that reason: they're a judgment call, not a law of nature.
    """
    if ece < WELL_CALIBRATED_ECE:
        return "well-calibrated"
    direction = "overconfident" if bias > 0 else "underconfident"
    severity = "moderately" if ece < MODERATELY_MISCALIBRATED_ECE else "severely"
    return f"{severity} {direction} -- confidence should not be trusted at face value"
