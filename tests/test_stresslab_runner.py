"""Tests for the torch-free parts of gotrain/stresslab/runner.py:
classify_fragility, run_perturbation_battery (with a fake evaluate
function standing in for the real torch forward pass), and the
record/read report round trip. run_stresslab_classification/segmentation
themselves need torch + geobench + real data and are NOT exercised here
-- see the module docstring.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import ledger, logging_utils
from gotrain.stresslab import perturbations, runner


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_utils, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(runner, "STRESSLAB_DIR", tmp_path / "stresslab")
    return tmp_path


def _synthetic_images(n=4, h=4, w=4, c=3):
    rng = np.random.default_rng(0)
    return rng.normal(100.0, 20.0, size=(n, h, w, c)).astype(np.float32)


# ---------------------------------------------------------------------------
# classify_fragility
# ---------------------------------------------------------------------------


def test_classify_fragility_robust_below_threshold():
    assert runner.classify_fragility(0.05) == "robust"


def test_classify_fragility_fragile_between_thresholds():
    assert runner.classify_fragility(0.20) == "fragile"


def test_classify_fragility_severely_fragile_above_threshold():
    assert runner.classify_fragility(0.50) == "severely_fragile"


def test_classify_fragility_unknown_when_no_relative_drop():
    assert runner.classify_fragility(None) == "unknown"


def test_classify_fragility_boundary_values_are_exclusive_on_the_low_side():
    assert runner.classify_fragility(runner.FRAGILE_RELATIVE_DROP) == "fragile"
    assert runner.classify_fragility(runner.SEVERELY_FRAGILE_RELATIVE_DROP) == "severely_fragile"


# ---------------------------------------------------------------------------
# run_perturbation_battery -- orchestration only, fake evaluate()
# ---------------------------------------------------------------------------


def test_run_perturbation_battery_reports_robust_model_when_metric_never_drops():
    images = _synthetic_images()
    result = runner.run_perturbation_battery(lambda imgs: 0.90, images)

    assert result["clean_metric"] == 0.90
    assert result["n_fragile"] == 0
    assert result["overall_verdict"] == "robust"
    assert result["n_perturbations_tested"] == len(result["results"])


def test_run_perturbation_battery_flags_fragility_under_a_specific_perturbation():
    images = _synthetic_images()

    def evaluate(imgs):
        # Identical to the clean array -> clean metric; anything perturbed
        # (band_dropout zeroes values, so it won't match exactly) -> worse.
        return 0.90 if np.array_equal(imgs, images) else 0.50

    result = runner.run_perturbation_battery(evaluate, images)
    assert result["clean_metric"] == 0.90
    assert result["n_fragile"] > 0
    assert "fragile under" in result["overall_verdict"]

    band_dropout_results = [r for r in result["results"] if r["perturbation"] == "band_dropout"]
    assert len(band_dropout_results) == len(perturbations.SEVERITIES_FOR["band_dropout"])
    for r in band_dropout_results:
        assert r["fragility"] in ("fragile", "severely_fragile")


def test_run_perturbation_battery_handles_zero_clean_metric_without_crashing():
    images = _synthetic_images()
    result = runner.run_perturbation_battery(lambda imgs: 0.0, images)
    assert result["clean_metric"] == 0.0
    for r in result["results"]:
        assert r["relative_drop"] is None
        assert r["fragility"] == "unknown"


def test_run_perturbation_battery_covers_every_registered_perturbation():
    images = _synthetic_images()
    result = runner.run_perturbation_battery(lambda imgs: 0.8, images)
    tested_names = {r["perturbation"] for r in result["results"]}
    assert tested_names == set(perturbations.PERTURBATIONS)


# ---------------------------------------------------------------------------
# _record_and_log / read_report round trip
# ---------------------------------------------------------------------------


def test_record_and_log_writes_report_and_logs_event(isolated_logs):
    battery_result = runner.run_perturbation_battery(lambda imgs: 0.9, _synthetic_images())
    summary = runner._record_and_log("m-eurosat", "run-1", "classification", battery_result)

    assert summary["dataset"] == "m-eurosat"
    assert summary["run_id"] == "run-1"

    read_back = runner.read_report("m-eurosat", "run-1")
    assert read_back["summary"] == summary
    assert len(read_back["results"]) == battery_result["n_perturbations_tested"]

    events = logging_utils.read_events(event="stresslab_finished", dataset="m-eurosat")
    assert len(events) == 1
    assert events[0]["run_id"] == "run-1"


def test_read_report_returns_none_when_nothing_recorded(isolated_logs):
    assert runner.read_report("m-eurosat", "never-run") is None


# ---------------------------------------------------------------------------
# _get_run_record -- the one pure-Python pre-flight check inside the
# torch-dependent run_stresslab_* functions, worth covering directly
# ---------------------------------------------------------------------------


def test_get_run_record_raises_for_unknown_run_id(isolated_logs, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", isolated_logs / "ledger.jsonl")
    with pytest.raises(ValueError):
        runner._get_run_record("m-eurosat", "nonexistent")


def test_get_run_record_raises_for_cross_dataset_run_id(isolated_logs, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", isolated_logs / "ledger.jsonl")
    ledger.add_run(dataset="m-so2sat", config={}, val_metric=0.5, run_id="run-1")
    with pytest.raises(ValueError):
        runner._get_run_record("m-eurosat", "run-1")
