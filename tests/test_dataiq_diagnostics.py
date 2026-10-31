"""Tests for DataIQ's deterministic, model-free structural diagnostics
(gotrain/dataiq/diagnostics.py) -- pure numpy, no torch/geobench I/O.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain.dataiq import diagnostics


# ---------------------------------------------------------------------------
# scan_samples
# ---------------------------------------------------------------------------


def test_scan_samples_clean_classification_batch():
    images = np.random.uniform(0, 1, size=(5, 4, 4, 3)).astype(np.float32)
    labels = np.array([0, 1, 2, 1, 0])
    records = diagnostics.scan_samples(images, labels, "classification", n_classes=3)
    assert len(records) == 5
    assert all(r["severity"] == diagnostics.SEVERITY_NONE for r in records)
    assert all(r["problems"] == [] for r in records)


def test_scan_samples_flags_nan_and_inf():
    images = np.random.uniform(0, 1, size=(3, 4, 4, 1)).astype(np.float32)
    images[0, 0, 0, 0] = np.nan
    images[1, 1, 1, 0] = np.inf
    labels = np.array([0, 1, 0])
    records = diagnostics.scan_samples(images, labels, "classification", n_classes=2)
    assert records[0]["severity"] == diagnostics.SEVERITY_HIGH
    assert "nan" in records[0]["problems"][0].lower()
    assert records[1]["severity"] == diagnostics.SEVERITY_HIGH
    assert records[2]["severity"] == diagnostics.SEVERITY_NONE


def test_scan_samples_flags_constant_image():
    images = np.zeros((2, 4, 4, 1), dtype=np.float32)
    images[0] = 5.0  # constant, nonzero value -- still zero variance
    images[1] = np.random.uniform(0, 1, size=(4, 4, 1)).astype(np.float32)
    labels = np.array([0, 0])
    records = diagnostics.scan_samples(images, labels, "classification", n_classes=1)
    assert records[0]["severity"] == diagnostics.SEVERITY_HIGH
    assert "constant" in records[0]["problems"][0].lower()
    assert records[1]["severity"] == diagnostics.SEVERITY_NONE


def test_scan_samples_flags_out_of_range_classification_label():
    images = np.random.uniform(0, 1, size=(2, 4, 4, 1)).astype(np.float32)
    labels = np.array([0, 7])
    records = diagnostics.scan_samples(images, labels, "classification", n_classes=3)
    assert records[1]["severity"] == diagnostics.SEVERITY_HIGH
    assert "out of range" in records[1]["problems"][0].lower()


def test_scan_samples_flags_segmentation_mask_shape_mismatch():
    images = np.random.uniform(0, 1, size=(1, 8, 8, 3)).astype(np.float32)
    masks = np.zeros((1, 4, 4), dtype=np.int64)  # wrong shape
    records = diagnostics.scan_samples(images, masks, "segmentation", n_classes=2)
    assert records[0]["severity"] == diagnostics.SEVERITY_HIGH
    assert "shape" in records[0]["problems"][0].lower()


def test_scan_samples_flags_segmentation_mask_out_of_range():
    images = np.random.uniform(0, 1, size=(1, 4, 4, 3)).astype(np.float32)
    masks = np.full((1, 4, 4), 9, dtype=np.int64)
    records = diagnostics.scan_samples(images, masks, "segmentation", n_classes=2)
    assert records[0]["severity"] == diagnostics.SEVERITY_HIGH
    assert "out of range" in records[0]["problems"][0].lower()


def test_scan_samples_clean_segmentation_batch():
    images = np.random.uniform(0, 1, size=(2, 4, 4, 3)).astype(np.float32)
    masks = np.zeros((2, 4, 4), dtype=np.int64)
    masks[0, :2, :] = 1
    records = diagnostics.scan_samples(images, masks, "segmentation", n_classes=2)
    assert all(r["severity"] == diagnostics.SEVERITY_NONE for r in records)


def test_scan_samples_rejects_mismatched_counts():
    images = np.zeros((3, 4, 4, 1), dtype=np.float32)
    labels = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError):
        diagnostics.scan_samples(images, labels, "classification", n_classes=1)


# ---------------------------------------------------------------------------
# find_exact_duplicates
# ---------------------------------------------------------------------------


def test_find_exact_duplicates_detects_identical_images():
    base = np.random.uniform(0, 1, size=(4, 4, 3)).astype(np.float32)
    images = np.stack([base, base, np.random.uniform(0, 1, size=(4, 4, 3)).astype(np.float32), base])
    groups = diagnostics.find_exact_duplicates(images)
    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1, 3]


def test_find_exact_duplicates_none_when_all_distinct():
    images = np.random.uniform(0, 1, size=(5, 4, 4, 3)).astype(np.float32)
    assert diagnostics.find_exact_duplicates(images) == []


def test_find_exact_duplicates_does_not_flag_near_but_not_exact_duplicates():
    base = np.random.uniform(0, 1, size=(4, 4, 3)).astype(np.float32)
    near = base.copy()
    near[0, 0, 0] += 0.0001  # tiny perturbation -- NOT an exact duplicate
    images = np.stack([base, near])
    assert diagnostics.find_exact_duplicates(images) == []


# ---------------------------------------------------------------------------
# summarize_scan
# ---------------------------------------------------------------------------


def test_summarize_scan_combines_structural_and_duplicate_findings():
    structural_records = [
        {"sample_id": 0, "severity": diagnostics.SEVERITY_NONE, "problems": []},
        {"sample_id": 1, "severity": diagnostics.SEVERITY_HIGH, "problems": ["1 NaN / 0 Inf pixel value(s)"]},
        {"sample_id": 2, "severity": diagnostics.SEVERITY_NONE, "problems": []},
    ]
    duplicate_groups = [[0, 2]]
    summary = diagnostics.summarize_scan(structural_records, duplicate_groups)
    assert summary["n_samples_scanned"] == 3
    assert summary["n_invalid"] == 1
    assert summary["invalid_sample_ids"] == [1]
    assert summary["n_duplicate_groups"] == 1
    assert sorted(summary["duplicate_sample_ids"]) == [0, 2]
