"""Tests for gotrain/data_loading.py and the memory pre-flight in
gotrain/tools/training.py -- the shared fixes for the m-NeonTree,
m-SA-crop-type, m-cashew-plant and m-bigearthnet failure paths found in the
pre-submission audit.

Findings these pin down (measured on real GEO-Bench samples fetched as a
train/valid-only subset; no test-split sample was downloaded):
  - m-NeonTree bands have different native sizes (RGB 400x400, Canopy
    Height 40x40x1, Neon hyperspectral 40x40x369). pack_to_3d without
    resample raised (masked by a formatting bug in geobench's own error
    message), so every all-bands NeonTree run, GeoGuard and DataIQ call
    crashed. And len(band_names)=5 is not the channel count (373).
  - All-bands NeonTree is infeasible: ~40 s and ~239 MB per sample once
    resampled. RGB + Canopy Height is 4 channels, ~3 MB/sample.
  - Eager loading held two copies of a split (list + np.stack) and the
    stats cast the whole train array to float64 -- ~4x the float32 size at
    peak for m-SA-crop-type / m-cashew-plant / m-bigearthnet.
"""

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import data_loading  # noqa: E402
from gotrain.tools import training  # noqa: E402


def test_band_stats_matches_numpy_population_stats():
    rng = np.random.default_rng(0)
    images = rng.normal(1500, 400, size=(37, 9, 11, 13)).astype(np.float32)
    mean, std = data_loading.band_stats(images, chunk_values=1000)  # force many chunks
    flat = images.reshape(-1, 13).astype(np.float64)
    np.testing.assert_allclose(mean, flat.mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(std, flat.std(axis=0), rtol=1e-7)


def test_band_stats_guards_constant_band_and_int16_input():
    images = np.full((4, 8, 8, 2), 7, dtype=np.int16)
    images[..., 1] = np.arange(4 * 8 * 8).reshape(4, 8, 8)
    mean, std = data_loading.band_stats(images)
    assert std[0] == 1.0
    assert mean[0] == 7.0
    assert std[1] > 1.0


def test_pack_sample_resamples_and_narrows_float64():
    calls = {}

    class Sample:
        def pack_to_3d(self, band_names, resample=False):
            calls["resample"] = resample
            return np.zeros((4, 4, 2), dtype=np.float64), band_names

    arr = data_loading.pack_sample(Sample(), ["a", "b"])
    assert calls["resample"] is True
    assert arr.dtype == np.float32


def test_projection_fails_fast_with_a_concrete_sample_cap():
    started = time.time() - 6.4  # 64 samples took 6.4 s -> 100 ms/sample
    with pytest.raises(data_loading.DataLoadingBudgetExceeded) as exc:
        data_loading.check_loading_budget("train", 64, 20000, started, loading_deadline=started + 150)
    message = str(exc.value)
    assert "max_train_samples <=" in message
    assert "100 ms/sample" in message


def test_early_projection_only_fires_on_a_gross_overrun():
    started = time.time() - 0.8  # 8 samples at 100 ms -> 1.2 s left needed for 20 total
    data_loading.check_loading_budget("train", 8, 20, started, loading_deadline=time.time() + 1.0)
    with pytest.raises(data_loading.DataLoadingBudgetExceeded):
        data_loading.check_loading_budget("train", 8, 200, started, loading_deadline=time.time() + 1.0)


def test_no_deadline_never_raises():
    data_loading.check_loading_budget("train", 64, 10 ** 9, time.time() - 1000, None)


# ---------------------------------------------------------------------------
# Memory pre-flight (gotrain.tools.training.check_memory_budget)
# ---------------------------------------------------------------------------


def _estimate(per_sample_mb, n_train, n_val, band_channels=None, band_shapes=None):
    per = int(per_sample_mb * 1e6)
    return {
        "per_sample_bytes": per, "n_train": n_train, "n_val": n_val,
        "total_bytes": per * (n_train + n_val), "channels": 373, "height": 400, "width": 400,
        "band_channels": band_channels or {"Red": 1}, "band_shapes": band_shapes or {"Red": [400, 400]},
    }


def _patch(monkeypatch, estimate, available_gb):
    import psutil

    from gotrain import dataset_spec

    monkeypatch.setattr(dataset_spec, "estimate_training_memory", lambda *a, **k: estimate)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=int(available_gb * 1e9)))


def test_memory_preflight_refuses_neontree_all_bands_and_names_the_bands(monkeypatch):
    estimate = _estimate(
        239, 270, 94,
        band_channels={"Red": 1, "Green": 1, "Blue": 1, "Canopy Height Model": 1, "Neon": 369},
        band_shapes={"Red": [400, 400], "Neon": [40, 40]},
    )
    _patch(monkeypatch, estimate, available_gb=14)
    with pytest.raises(training.ResourceBudgetError) as exc:
        training.check_memory_budget("m-NeonTree", "segmentation", {})
    message = str(exc.value)
    assert "Neon" in message and "bands" in message
    assert isinstance(exc.value, training.PreflightValidationError)  # never costs a run


def test_memory_preflight_suggests_a_train_cap_that_fits(monkeypatch):
    _patch(monkeypatch, _estimate(0.7, 20000, 1000), available_gb=10)
    with pytest.raises(training.ResourceBudgetError) as exc:
        training.check_memory_budget("m-bigearthnet", "classification", {})
    cap = int(str(exc.value).split("max_train_samples <= ")[1].split(".")[0])
    assert 0 < cap < 20000
    assert (cap + 1000) * 0.7e6 <= 10e9 * training.MEMORY_BUDGET_FRACTION


def test_memory_preflight_allows_a_config_that_fits(monkeypatch):
    _patch(monkeypatch, _estimate(0.7, 500, 100), available_gb=10)
    training.check_memory_budget("m-bigearthnet", "classification", {"max_train_samples": 500})


def test_memory_preflight_skips_fake_and_unknown_estimates(monkeypatch):
    _patch(monkeypatch, None, available_gb=0.001)
    training.check_memory_budget("m-eurosat", "classification", {})
    training.check_memory_budget("m-eurosat", "fake", {})
