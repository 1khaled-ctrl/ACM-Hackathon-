"""Tests for gotrain/resource_intelligence.py. Pure Python/ledger I/O,
no torch needed.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import ledger, resource_intelligence


@pytest.fixture
def isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    return tmp_path


# ---------------------------------------------------------------------------
# estimate_run_cost
# ---------------------------------------------------------------------------


def test_estimate_run_cost_has_no_data_for_a_fresh_dataset(isolated_ledger):
    result = resource_intelligence.estimate_run_cost("m-eurosat", {"backbone": "resnet18", "epochs": 2})
    assert result["has_data"] is False
    assert result["estimated_minutes"] is None


def test_estimate_run_cost_scales_with_epochs_from_history(isolated_ledger):
    ledger.add_run(
        dataset="m-eurosat", config={"backbone": "resnet18", "epochs": 2}, val_metric=0.8,
        gpu_minutes=10.0,
    )
    result = resource_intelligence.estimate_run_cost("m-eurosat", {"backbone": "resnet18", "epochs": 4})
    assert result["has_data"] is True
    assert result["minutes_per_epoch"] == pytest.approx(5.0)
    assert result["estimated_minutes"] == pytest.approx(20.0)
    assert result["matched_backbone"] is True


def test_estimate_run_cost_prefers_same_backbone_history(isolated_ledger):
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet50", "epochs": 2}, val_metric=0.8, gpu_minutes=20.0)
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "epochs": 2}, val_metric=0.8, gpu_minutes=4.0)

    result = resource_intelligence.estimate_run_cost("m-eurosat", {"backbone": "resnet18", "epochs": 2})
    assert result["matched_backbone"] is True
    assert result["minutes_per_epoch"] == pytest.approx(2.0)  # only the resnet18 run, not resnet50's


def test_estimate_run_cost_falls_back_to_all_backbones_when_none_match(isolated_ledger):
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet50", "epochs": 2}, val_metric=0.8, gpu_minutes=20.0)

    result = resource_intelligence.estimate_run_cost("m-eurosat", {"backbone": "convnext_tiny", "epochs": 2})
    assert result["has_data"] is True
    assert result["matched_backbone"] is False
    assert result["minutes_per_epoch"] == pytest.approx(10.0)


def test_estimate_run_cost_ignores_runs_missing_gpu_minutes_or_epochs(isolated_ledger):
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18"}, val_metric=0.8, gpu_minutes=10.0)  # no epochs in config
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "epochs": 2}, val_metric=0.8)  # no gpu_minutes

    result = resource_intelligence.estimate_run_cost("m-eurosat", {"backbone": "resnet18", "epochs": 2})
    assert result["has_data"] is False


def test_estimate_run_cost_uses_default_epochs_when_config_omits_it(isolated_ledger):
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "epochs": 2}, val_metric=0.8, gpu_minutes=10.0)
    result = resource_intelligence.estimate_run_cost("m-eurosat", {"backbone": "resnet18"}, default_epochs=3)
    assert result["estimated_minutes"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# check_fits_in_run_budget
# ---------------------------------------------------------------------------


def test_check_fits_returns_none_verdict_without_cost_data():
    result = resource_intelligence.check_fits_in_run_budget({"has_data": False}, run_minutes_budget=30)
    assert result["fits"] is None


def test_check_fits_returns_none_verdict_without_budget():
    cost = {"has_data": True, "estimated_minutes": 10.0}
    result = resource_intelligence.check_fits_in_run_budget(cost, run_minutes_budget=None)
    assert result["fits"] is None


def test_check_fits_true_when_comfortably_within_budget():
    cost = {"has_data": True, "estimated_minutes": 10.0}
    result = resource_intelligence.check_fits_in_run_budget(cost, run_minutes_budget=30)
    assert result["fits"] is True
    assert result["ratio"] == pytest.approx(10.0 / 30)


def test_check_fits_true_but_tight_near_budget_edge():
    cost = {"has_data": True, "estimated_minutes": 27.0}
    result = resource_intelligence.check_fits_in_run_budget(cost, run_minutes_budget=30)
    assert result["fits"] is True
    assert "tight" in result["note"]


def test_check_fits_false_when_exceeding_budget():
    cost = {"has_data": True, "estimated_minutes": 45.0}
    result = resource_intelligence.check_fits_in_run_budget(cost, run_minutes_budget=30)
    assert result["fits"] is False
    assert "EXCEEDS" in result["note"]


# ---------------------------------------------------------------------------
# summarize_resource_usage
# ---------------------------------------------------------------------------


def test_summarize_resource_usage_on_empty_dataset(isolated_ledger):
    result = resource_intelligence.summarize_resource_usage("m-eurosat")
    assert result["n_runs"] == 0
    assert result["total_gpu_minutes_used"] == 0
    assert result["avg_gpu_minutes_per_run"] is None


def test_summarize_resource_usage_sums_real_gpu_minutes(isolated_ledger):
    ledger.add_run(dataset="m-eurosat", config={}, val_metric=0.8, gpu_minutes=5.0)
    ledger.add_run(dataset="m-eurosat", config={}, val_metric=0.85, gpu_minutes=7.0)
    ledger.add_run(dataset="m-eurosat", config={}, val_metric=0.82)  # no gpu_minutes recorded

    result = resource_intelligence.summarize_resource_usage("m-eurosat")
    assert result["n_runs"] == 3
    assert result["n_runs_with_recorded_cost"] == 2
    assert result["total_gpu_minutes_used"] == pytest.approx(12.0)
    assert result["avg_gpu_minutes_per_run"] == pytest.approx(6.0)


def test_summarize_resource_usage_ignores_other_datasets(isolated_ledger):
    ledger.add_run(dataset="m-eurosat", config={}, val_metric=0.8, gpu_minutes=5.0)
    ledger.add_run(dataset="m-so2sat", config={}, val_metric=0.5, gpu_minutes=100.0)

    result = resource_intelligence.summarize_resource_usage("m-eurosat")
    assert result["total_gpu_minutes_used"] == pytest.approx(5.0)
