"""Tests for gotrain/statistics.py. Pure Python/ledger I/O, no torch needed."""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import ledger, statistics


@pytest.fixture
def isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    return tmp_path


def test_estimate_noise_floor_reports_no_data_when_nothing_repeats(isolated_ledger):
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "seed": 0}, val_metric=0.80)
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet50", "seed": 0}, val_metric=0.82)

    result = statistics.estimate_noise_floor("m-eurosat")
    assert result["has_data"] is False
    assert result["noise_floor"] is None
    assert result["n_repeated_configs"] == 0


def test_estimate_noise_floor_computes_spread_from_seed_only_repeats(isolated_ledger):
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "seed": 0}, val_metric=0.80)
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "seed": 1}, val_metric=0.815)
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet50", "seed": 0}, val_metric=0.90)

    result = statistics.estimate_noise_floor("m-eurosat")
    assert result["has_data"] is True
    assert result["n_repeated_configs"] == 1
    assert result["noise_floor"] == pytest.approx(0.015)


def test_estimate_noise_floor_ignores_other_datasets(isolated_ledger):
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "seed": 0}, val_metric=0.80)
    ledger.add_run(dataset="m-eurosat", config={"backbone": "resnet18", "seed": 1}, val_metric=0.90)
    ledger.add_run(dataset="m-so2sat", config={"backbone": "resnet18", "seed": 0}, val_metric=0.50)

    result = statistics.estimate_noise_floor("m-so2sat")
    assert result["has_data"] is False


def test_is_improvement_significant_returns_none_verdict_without_delta():
    result = statistics.is_improvement_significant(None, {"has_data": False})
    assert result["significant"] is None


def test_is_improvement_significant_returns_none_verdict_without_noise_data():
    result = statistics.is_improvement_significant(0.05, {"has_data": False})
    assert result["significant"] is None
    assert "no repeated-config evidence" in result["reason"]


def test_is_improvement_significant_true_when_delta_exceeds_noise_floor():
    noise = {"has_data": True, "noise_floor": 0.01}
    result = statistics.is_improvement_significant(0.05, noise)
    assert result["significant"] is True


def test_is_improvement_significant_false_when_delta_within_noise_floor():
    noise = {"has_data": True, "noise_floor": 0.03}
    result = statistics.is_improvement_significant(0.01, noise)
    assert result["significant"] is False
