"""Tests for GeoGuard's pure logic (diagnostics stats, the decision engine,
and QA checks) using synthetic numpy arrays -- no GEO-Bench data on disk
and no GPU required, since none of these functions touch geobench except
inside load_train_sample (not exercised here; that path mirrors the
already-tested build_dataset_arrays in the training templates and needs
real downloaded data to run).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain.config_schema import ConfigValidationError
from gotrain.geoguard import decision, diagnostics, qa, report


# ---------------------------------------------------------------------------
# diagnostics.classify_bands
# ---------------------------------------------------------------------------


def test_classify_bands_splits_so2sat_sar_and_optical():
    band_names = [
        "VH.Real", "VH.Imaginary", "VV.Real", "VV.Imaginary",
        "VH.LEE Filtered", "VV.LEE Filtered", "VH.LEE Filtered.Real", "VV.LEE Filtered.Imaginary",
        "02 - Blue", "03 - Green", "04 - Red", "08 - NIR",
    ]
    result = diagnostics.classify_bands(band_names)
    assert len(result["sar_bands"]) == 8
    assert len(result["optical_bands"]) == 4
    assert result["is_multimodal"] is True


def test_classify_bands_rgb_only_is_not_multimodal():
    result = diagnostics.classify_bands(["Red", "Green", "Blue"])
    assert result["sar_bands"] == []
    assert result["is_multimodal"] is False


# ---------------------------------------------------------------------------
# diagnostics.compute_pixel_diagnostics
# ---------------------------------------------------------------------------


def test_compute_pixel_diagnostics_basic_stats():
    rng = np.random.default_rng(0)
    images = rng.normal(loc=100.0, scale=10.0, size=(20, 8, 8, 2)).astype(np.float32)
    result = diagnostics.compute_pixel_diagnostics(images, ["band_a", "band_b"])

    assert result["n_samples"] == 20
    assert result["n_channels"] == 2
    assert result["nan_count"] == 0
    assert result["inf_count"] == 0
    for band in result["per_band"]:
        assert abs(band["mean"] - 100.0) < 2.0
        assert abs(band["std"] - 10.0) < 2.0
        assert band["near_constant"] is False


def test_compute_pixel_diagnostics_detects_saturation():
    images = np.random.uniform(0, 100, size=(100, 4, 4, 1)).astype(np.float32)
    # Force >0.5% of pixels to sit exactly at a clipped max value.
    images[:10, :, :, 0] = 255.0
    result = diagnostics.compute_pixel_diagnostics(images, ["band_a"])
    assert result["per_band"][0]["possible_saturation"] is True


def test_compute_pixel_diagnostics_excludes_nan_from_stats_but_counts_them():
    images = np.ones((5, 2, 2, 1), dtype=np.float32) * 10.0
    images[0, 0, 0, 0] = np.nan
    images[1, 0, 0, 0] = np.inf
    result = diagnostics.compute_pixel_diagnostics(images, ["band_a"])
    assert result["nan_count"] == 1
    assert result["inf_count"] == 1
    assert result["per_band"][0]["mean"] == pytest.approx(10.0)


def test_compute_pixel_diagnostics_detects_near_constant_band():
    images = np.full((10, 2, 2, 1), 5.0, dtype=np.float32)
    result = diagnostics.compute_pixel_diagnostics(images, ["constant_band"])
    assert result["per_band"][0]["near_constant"] is True


def test_compute_pixel_diagnostics_rejects_wrong_band_count():
    images = np.zeros((5, 2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        diagnostics.compute_pixel_diagnostics(images, ["only_one_band"])


# ---------------------------------------------------------------------------
# diagnostics.compute_class_diagnostics
# ---------------------------------------------------------------------------


def test_compute_class_diagnostics_classification_rare_classes():
    # 100 samples: class 0 x90, class 1 x9, class 2 x1 -> class 2 is rare.
    labels = np.array([0] * 90 + [1] * 9 + [2] * 1)
    result = diagnostics.compute_class_diagnostics(labels, ["a", "b", "c"], is_pixelwise=False)
    assert result["rare_classes"] == ["c"]
    assert result["unit"] == "samples"
    assert result["background_dominant"] is False  # only applies to binary pixelwise


def test_compute_class_diagnostics_segmentation_background_dominant():
    # 95% background (class 0), 5% foreground (class 1).
    masks = np.zeros((1, 100, 100), dtype=np.int64)
    masks[0, :5, :] = 1
    result = diagnostics.compute_class_diagnostics(masks, ["background", "foreground"], is_pixelwise=True)
    assert result["background_dominant"] is True
    assert result["unit"] == "pixels"


def test_compute_class_diagnostics_handles_absent_class():
    labels = np.array([0, 0, 1, 1])
    result = diagnostics.compute_class_diagnostics(labels, ["a", "b", "c"], is_pixelwise=False)
    assert result["classes_absent_from_sample"] == ["c"]


# ---------------------------------------------------------------------------
# decision.recommend
# ---------------------------------------------------------------------------


def _pixel_diag(n_samples=2000, saturated=False, nan_count=0, inf_count=0):
    band = {
        "band": "b0", "min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.1,
        "pct_at_max": 0.01 if saturated else 0.0, "pct_at_min": 0.0,
        "possible_saturation": saturated, "near_constant": False,
    }
    return {"n_samples": n_samples, "height": 8, "width": 8, "n_channels": 1,
            "nan_count": nan_count, "inf_count": inf_count, "per_band": [band]}


def _class_diag(rare=None, background_dominant=False, unit="samples", frequencies=None):
    return {
        "n_classes": 2, "class_names": ["a", "b"], "counts": [1, 1],
        "frequencies": frequencies or [0.5, 0.5], "rare_classes": rare or [],
        "classes_absent_from_sample": [], "imbalance_ratio": 1.0,
        "background_dominant": background_dominant, "unit": unit,
    }


def test_recommend_always_returns_a_validatable_config():
    result = decision.recommend(
        dataset="m-pv4ger-seg", task_type="segmentation",
        pixel_diagnostics=_pixel_diag(n_samples=5000), class_diagnostics=_class_diag(),
        band_classification=diagnostics.classify_bands(["Red", "Green", "Blue"]),
        dataset_notes={"sensor": "RGB", "warnings": []},
    )
    # validate_config inside recommend() would have raised if this were invalid.
    assert "recommended_config" in result


def test_recommend_m_eurosat_forces_bands_null():
    result = decision.recommend(
        dataset="m-eurosat", task_type="classification",
        pixel_diagnostics=_pixel_diag(n_samples=2000), class_diagnostics=_class_diag(),
        band_classification=diagnostics.classify_bands(["01 - Coastal aerosol"]),
        dataset_notes=diagnostics.DATASET_NOTES["m-eurosat"],
    )
    assert result["recommended_config"]["bands"] is None
    assert any("band name" in f.lower() or "bands" in f.lower() for f in result["findings"])


def test_recommend_small_dataset_triggers_strong_augmentation():
    result = decision.recommend(
        dataset="m-nz-cattle", task_type="segmentation",
        pixel_diagnostics=_pixel_diag(n_samples=524), class_diagnostics=_class_diag(unit="pixels"),
        band_classification=diagnostics.classify_bands(["Red", "Green", "Blue"]),
        dataset_notes={"sensor": "RGB", "warnings": []},
    )
    assert result["recommended_config"]["augmentation"] == "strong"


def test_recommend_background_dominant_segmentation_recommends_dice():
    result = decision.recommend(
        dataset="m-nz-cattle", task_type="segmentation",
        pixel_diagnostics=_pixel_diag(n_samples=5000),
        class_diagnostics=_class_diag(background_dominant=True, unit="pixels", frequencies=[0.98, 0.02]),
        band_classification=diagnostics.classify_bands(["Red", "Green", "Blue"]),
        dataset_notes={"sensor": "RGB", "warnings": []},
    )
    assert result["recommended_config"]["loss"] == "ce+dice"


def test_recommend_background_dominant_classification_does_not_set_loss():
    # "loss" is not a classification config key -- must never be proposed there.
    result = decision.recommend(
        dataset="m-so2sat", task_type="classification",
        pixel_diagnostics=_pixel_diag(n_samples=20000),
        class_diagnostics=_class_diag(background_dominant=True, unit="samples"),
        band_classification=diagnostics.classify_bands(["VH.Real", "02 - Blue"]),
        dataset_notes={"sensor": "SAR+optical", "warnings": []},
    )
    assert "loss" not in result["recommended_config"]


def test_recommend_rare_classes_sets_normalized_class_weights():
    freqs = [0.9, 0.09, 0.01]
    result = decision.recommend(
        dataset="m-so2sat", task_type="classification",
        pixel_diagnostics=_pixel_diag(n_samples=20000),
        class_diagnostics=_class_diag(rare=["c"], unit="samples", frequencies=freqs),
        band_classification=diagnostics.classify_bands(["VH.Real", "02 - Blue"]),
        dataset_notes={"sensor": "SAR+optical", "warnings": []},
    )
    weights = result["recommended_config"]["class_weights"]
    assert len(weights) == 3
    assert weights[2] > weights[0]  # rarer class gets a larger weight


def test_recommend_multimodal_dataset_adds_rationale_without_log_transform():
    result = decision.recommend(
        dataset="m-so2sat", task_type="classification",
        pixel_diagnostics=_pixel_diag(n_samples=20000), class_diagnostics=_class_diag(unit="samples"),
        band_classification=diagnostics.classify_bands(["VH.Real", "VV.Real", "02 - Blue"]),
        dataset_notes={"sensor": "SAR+optical", "warnings": []},
    )
    assert any("multi-modality" in f.lower() or "sar" in f.lower() for f in result["findings"])
    assert any("log" in r.lower() or "db" in r.lower() for r in result["rationale"])


# ---------------------------------------------------------------------------
# qa.check_classification_batch / check_segmentation_batch
# ---------------------------------------------------------------------------


def test_qa_classification_batch_passes_on_valid_input():
    images = np.zeros((4, 3, 8, 8), dtype=np.float32)
    labels = np.array([0, 1, 2, 1])
    result = qa.check_classification_batch(images, labels, n_bands=3, n_classes=3)
    assert result["passed"] is True
    assert result["problems"] == []


def test_qa_classification_batch_catches_nan():
    images = np.zeros((2, 3, 4, 4), dtype=np.float32)
    images[0, 0, 0, 0] = np.nan
    labels = np.array([0, 1])
    result = qa.check_classification_batch(images, labels, n_bands=3, n_classes=3)
    assert result["passed"] is False
    assert any("nan" in p.lower() for p in result["problems"])


def test_qa_classification_batch_catches_wrong_channel_count():
    images = np.zeros((2, 5, 4, 4), dtype=np.float32)
    labels = np.array([0, 1])
    result = qa.check_classification_batch(images, labels, n_bands=3, n_classes=3)
    assert result["passed"] is False
    assert any("channels" in p.lower() for p in result["problems"])


def test_qa_classification_batch_catches_out_of_range_label():
    images = np.zeros((2, 3, 4, 4), dtype=np.float32)
    labels = np.array([0, 5])
    result = qa.check_classification_batch(images, labels, n_bands=3, n_classes=3)
    assert result["passed"] is False
    assert any("out of range" in p.lower() for p in result["problems"])


def test_qa_segmentation_batch_passes_on_valid_input():
    images = np.zeros((2, 3, 16, 16), dtype=np.float32)
    masks = np.zeros((2, 16, 16), dtype=np.int64)
    result = qa.check_segmentation_batch(images, masks, n_bands=3, n_classes=2)
    assert result["passed"] is True


def test_qa_segmentation_batch_catches_shape_mismatch():
    images = np.zeros((2, 3, 16, 16), dtype=np.float32)
    masks = np.zeros((2, 8, 8), dtype=np.int64)
    result = qa.check_segmentation_batch(images, masks, n_bands=3, n_classes=2)
    assert result["passed"] is False
    assert any("mismatch" in p.lower() for p in result["problems"])


def test_qa_segmentation_batch_catches_out_of_range_mask_value():
    images = np.zeros((1, 3, 4, 4), dtype=np.float32)
    masks = np.full((1, 4, 4), 9, dtype=np.int64)
    result = qa.check_segmentation_batch(images, masks, n_bands=3, n_classes=2)
    assert result["passed"] is False
    assert any("out of range" in p.lower() for p in result["problems"])


# ---------------------------------------------------------------------------
# report.write_report / read_report round-trip
# ---------------------------------------------------------------------------


def test_report_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "GEOGUARD_DIR", tmp_path / "geoguard")
    sample = {"dataset": "m-eurosat", "task_type": "classification", "findings": ["x"], "recommended_config": {}}

    report.write_report("m-eurosat", sample)
    loaded = report.read_report("m-eurosat")
    assert loaded == sample

    # no leftover .tmp file after a write, same atomicity guarantee as logging_utils.
    assert not list((tmp_path / "geoguard").glob(".geoguard_*.tmp"))


def test_report_read_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "GEOGUARD_DIR", tmp_path / "geoguard")
    assert report.read_report("does-not-exist") is None


def test_format_summary_includes_findings_and_config():
    sample = {
        "dataset": "m-nz-cattle", "task_type": "segmentation",
        "findings": ["small dataset"], "recommended_config": {"augmentation": "strong"},
        "rationale": ["because small"],
    }
    text = report.format_summary(sample)
    assert "m-nz-cattle" in text
    assert "small dataset" in text
    assert "augmentation" in text
