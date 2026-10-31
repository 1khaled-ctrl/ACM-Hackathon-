"""Tests for gotrain/calibration.py -- Expected Calibration Error (ECE)
over synthetic (confidence, correct) pairs. Pure numpy, no torch needed,
so these are real, not approximations of a torch-dependent code path.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import calibration


def test_perfectly_calibrated_model_has_near_zero_ece():
    # Confidence exactly tracks accuracy in each decile -- the textbook
    # "well-calibrated" case.
    rng = np.random.default_rng(0)
    confidences = rng.uniform(0, 1, size=5000)
    correct = rng.uniform(0, 1, size=5000) < confidences  # P(correct) == confidence, by construction
    result = calibration.compute_calibration(confidences, correct, n_bins=10)
    assert result["ece"] < 0.03
    assert result["verdict"] == "well-calibrated"


def test_severely_overconfident_model_is_flagged():
    # Always 95% confident, only right half the time.
    confidences = np.full(1000, 0.95)
    correct = np.zeros(1000)
    correct[:500] = 1
    result = calibration.compute_calibration(confidences, correct, n_bins=10)
    assert result["ece"] > 0.3
    assert result["bias"] > 0  # overconfident: confidence > accuracy
    assert "overconfident" in result["verdict"]
    assert "severely" in result["verdict"]


def test_underconfident_model_is_flagged_with_opposite_bias():
    # Always 40% confident, right 90% of the time.
    confidences = np.full(1000, 0.40)
    correct = np.ones(1000)
    correct[:100] = 0
    result = calibration.compute_calibration(confidences, correct, n_bins=10)
    assert result["bias"] < 0  # underconfident: confidence < accuracy
    assert "underconfident" in result["verdict"]


def test_mean_confidence_and_accuracy_reported():
    confidences = np.array([0.9, 0.9, 0.9, 0.9])
    correct = np.array([1, 1, 0, 0])
    result = calibration.compute_calibration(confidences, correct, n_bins=10)
    assert result["mean_confidence"] == pytest.approx(0.9)
    assert result["mean_accuracy"] == pytest.approx(0.5)
    assert result["bias"] == pytest.approx(0.4)


def test_bins_report_count_and_gap():
    confidences = np.array([0.05, 0.05, 0.95, 0.95])
    correct = np.array([0, 0, 1, 0])  # low-confidence bin perfectly calibrated, high-confidence bin overconfident
    result = calibration.compute_calibration(confidences, correct, n_bins=10)
    low_bin = result["bins"][0]
    high_bin = result["bins"][9]
    assert low_bin["count"] == 2
    assert low_bin["accuracy"] == pytest.approx(0.0)
    assert low_bin["gap"] == pytest.approx(0.05)
    assert high_bin["count"] == 2
    assert high_bin["accuracy"] == pytest.approx(0.5)
    assert high_bin["gap"] == pytest.approx(0.45)


def test_empty_bins_report_none_not_zero():
    # A bin with no samples must not silently report 0 accuracy -- that
    # would misleadingly look like "always wrong" instead of "no data."
    confidences = np.array([0.95, 0.95])
    correct = np.array([1, 1])
    result = calibration.compute_calibration(confidences, correct, n_bins=10)
    empty_bin = result["bins"][0]  # the [0.0, 0.1) bin, nothing falls in it
    assert empty_bin["count"] == 0
    assert empty_bin["accuracy"] is None
    assert empty_bin["avg_confidence"] is None


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        calibration.compute_calibration([0.9, 0.8], [1, 0, 1], n_bins=5)


def test_handles_empty_input_gracefully():
    result = calibration.compute_calibration([], [], n_bins=10)
    assert result["ece"] is None
    assert result["n_samples"] == 0
    assert result["verdict"] == "no samples to calibrate"


def test_highest_confidence_bin_includes_exactly_1_0():
    # Confidence of exactly 1.0 must land in the last bin, not get
    # excluded by a half-open [lo, hi) boundary.
    confidences = np.array([1.0])
    correct = np.array([1])
    result = calibration.compute_calibration(confidences, correct, n_bins=10)
    assert result["bins"][-1]["count"] == 1
