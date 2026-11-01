"""Tests for the central dataset specification (gotrain/dataset_spec.py)
and the multi-label-aware logic it unblocks across train_classification.py,
GeoGuard, and DataIQ.

classify_label_type / spec_for are tested against REAL geobench.LabelType
instances constructed in-process (geobench is installed, so this needs no
downloaded data on disk) -- this is the actual object every downstream
consumer (training templates, GeoGuard, DataIQ, StressLab, final_eval) gets
back from task_specs.label_type, not a hand-rolled stand-in for it.

check_support()/check_dataset_support() against datasets NOT on disk is
tested too, since that's exactly the "before any download" use case the
task asks for a reporting tool to cover.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

import geobench

from gotrain.dataset_spec import (
    LABEL_KIND_MULTI,
    LABEL_KIND_SEGMENTATION,
    LABEL_KIND_SINGLE,
    DatasetSpec,
    UnsupportedLabelTypeError,
    _class_names_of,
    check_support,
    classify_label_type,
    spec_for,
)


# ---------------------------------------------------------------------------
# classify_label_type -- the core of the registry
# ---------------------------------------------------------------------------


def test_classify_label_type_single_label_classification():
    label_type = geobench.Classification(n_classes=10, class_names=[str(i) for i in range(10)])
    assert classify_label_type(label_type) == LABEL_KIND_SINGLE


def test_classify_label_type_multi_label_classification():
    # m-bigearthnet's real shape: 43 classes, multi-hot targets.
    label_type = geobench.MultiLabelClassification(n_classes=43, class_names=[f"c{i}" for i in range(43)])
    assert classify_label_type(label_type) == LABEL_KIND_MULTI


def test_classify_label_type_segmentation():
    label_type = geobench.SegmentationClasses(
        name="label", spatial_resolution=1.0, n_classes=7, class_names=[f"c{i}" for i in range(7)]
    )
    assert classify_label_type(label_type) == LABEL_KIND_SEGMENTATION


def test_classify_label_type_unsupported_raises_with_reason():
    unknown_label_type = SimpleNamespace(n_classes=5)  # not any real geobench.LabelType subclass
    with pytest.raises(UnsupportedLabelTypeError) as exc_info:
        classify_label_type(unknown_label_type)
    assert "SimpleNamespace" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _class_names_of -- the class_names vs class_name attribute-name mismatch
# between geobench.Classification/SegmentationClasses and
# geobench.MultiLabelClassification (see dataset_spec.py's docstring)
# ---------------------------------------------------------------------------


def test_class_names_of_handles_plural_attribute():
    label_type = geobench.Classification(n_classes=3, class_names=["a", "b", "c"])
    assert _class_names_of(label_type) == ["a", "b", "c"]


def test_class_names_of_handles_multilabel_singular_attribute():
    # geobench.MultiLabelClassification stores names under `class_name`
    # (singular), not `class_names` -- confirmed by reading the installed
    # geobench package's source. A naive `getattr(lt, "class_names", None)`
    # alone would silently return None/[] for every multi-label dataset.
    label_type = geobench.MultiLabelClassification(n_classes=3, class_names=["x", "y", "z"])
    assert _class_names_of(label_type) == ["x", "y", "z"]


def test_class_names_of_returns_none_when_dataset_has_no_names():
    label_type = geobench.Classification(n_classes=3, class_names=None)
    assert _class_names_of(label_type) is None


# ---------------------------------------------------------------------------
# spec_for -- builds the full DatasetSpec from a (possibly already-loaded)
# task_specs object, without touching disk when task_specs is passed in.
# ---------------------------------------------------------------------------


def _fake_task_specs(label_type, band_names):
    bands_info = [SimpleNamespace(name=n) for n in band_names]
    return SimpleNamespace(label_type=label_type, bands_info=bands_info)


def test_spec_for_single_label_classification_eurosat_like():
    label_type = geobench.Classification(n_classes=10, class_names=[f"c{i}" for i in range(10)])
    task_specs = _fake_task_specs(label_type, [f"b{i}" for i in range(13)])
    spec = spec_for("m-eurosat", task_specs=task_specs)

    assert spec.label_kind == LABEL_KIND_SINGLE
    assert spec.n_classes == 10
    assert spec.n_bands == 13
    assert spec.is_multi_label is False
    assert spec.is_segmentation is False
    assert spec.task_type == "classification"
    assert spec.benchmark == "classification_v1.0"


def test_spec_for_multi_label_classification_bigearthnet_like():
    label_type = geobench.MultiLabelClassification(n_classes=43, class_names=[f"c{i}" for i in range(43)])
    task_specs = _fake_task_specs(label_type, [f"b{i}" for i in range(12)])
    spec = spec_for("m-bigearthnet", task_specs=task_specs)

    assert spec.label_kind == LABEL_KIND_MULTI
    assert spec.n_classes == 43
    assert spec.n_bands == 12
    assert spec.is_multi_label is True
    assert spec.is_segmentation is False
    assert spec.task_type == "classification"  # same template/config schema as single-label
    assert spec.benchmark == "classification_v1.0"
    assert spec.class_names == [f"c{i}" for i in range(43)]


def test_spec_for_segmentation_rgbn_chesapeake_like():
    # m-chesapeake: RGBN (4 bands), 7 classes -- this should need zero
    # changes beyond the generic in_channels=n_bands path already in
    # train_segmentation.py, and spec_for is what reports that n_bands
    # correctly to it.
    label_type = geobench.SegmentationClasses(
        name="label", spatial_resolution=1.0, n_classes=7, class_names=[f"c{i}" for i in range(7)]
    )
    task_specs = _fake_task_specs(label_type, ["R", "G", "B", "N"])
    spec = spec_for("m-chesapeake", task_specs=task_specs)

    assert spec.label_kind == LABEL_KIND_SEGMENTATION
    assert spec.n_bands == 4
    assert spec.band_names == ["R", "G", "B", "N"]
    assert spec.is_segmentation is True
    assert spec.task_type == "segmentation"
    assert spec.benchmark == "segmentation_v1.0"


def test_spec_for_segmentation_5band_neontree_like_not_forced_to_rgb():
    # m-NeonTree: RGB + hyperspectral + elevation combined into 5 bands.
    # The task explicitly warns against silently forcing RGB-only
    # behavior -- this asserts n_bands/band_names come through as all 5,
    # not truncated to the first 3.
    label_type = geobench.SegmentationClasses(
        name="label", spatial_resolution=1.0, n_classes=2, class_names=["background", "tree"]
    )
    task_specs = _fake_task_specs(label_type, ["R", "G", "B", "Hyperspectral", "Elevation"])
    spec = spec_for("m-NeonTree", task_specs=task_specs)

    assert spec.n_bands == 5
    assert spec.band_names == ["R", "G", "B", "Hyperspectral", "Elevation"]


def test_spec_for_binary_classification_handles_two_classes_correctly():
    # m-brick-kiln / m-pv4ger are binary classification -- same
    # Classification label_type as 10-class m-eurosat, just n_classes=2.
    # Nothing here should special-case "2" as if it were secretly
    # segmentation or multi-label.
    label_type = geobench.Classification(n_classes=2, class_names=["no_kiln", "kiln"])
    task_specs = _fake_task_specs(label_type, ["R", "G", "B"])
    spec = spec_for("m-brick-kiln", task_specs=task_specs)

    assert spec.label_kind == LABEL_KIND_SINGLE
    assert spec.n_classes == 2
    assert spec.is_multi_label is False


# ---------------------------------------------------------------------------
# check_support -- the "dataset -> supported/unsupported + reason" tool,
# exercised against datasets that are NOT downloaded (the actual state of
# the 8 new hackathon datasets at the time this was written) to confirm it
# reports a clear reason instead of raising.
# ---------------------------------------------------------------------------


def test_check_support_reports_not_found_for_missing_dataset_without_raising():
    result = check_support("m-bigearthnet-definitely-not-downloaded-xyz")
    assert result["supported"] is False
    assert "not found locally" in result["reason"]
    assert result["label_kind"] is None


def test_check_support_ok_for_a_real_downloaded_dataset():
    # m-eurosat is one of the 4 already-working datasets -- this is the
    # one real end-to-end check that touches disk (skipped if the data
    # genuinely isn't there in whatever environment runs this).
    result = check_support("m-eurosat")
    if not result["supported"] and "not found locally" in result["reason"]:
        pytest.skip("m-eurosat not downloaded in this environment")
    assert result["supported"] is True
    assert result["label_kind"] == LABEL_KIND_SINGLE
    assert result["n_classes"] == 10
    assert result["n_bands"] == 13


# ---------------------------------------------------------------------------
# report_all_datasets / check_dataset_support (gotrain/tools/dataset_support.py)
# -- the actual tool surface, including all 12 hackathon dataset names.
# ---------------------------------------------------------------------------


def test_report_all_datasets_covers_all_12_hackathon_datasets_without_raising():
    from gotrain.tools.dataset_support import HACKATHON_DATASETS, report_all_datasets

    assert len(HACKATHON_DATASETS) == 12
    assert "m-bigearthnet" in HACKATHON_DATASETS
    assert "m-NeonTree" in HACKATHON_DATASETS
    assert "m-chesapeake" in HACKATHON_DATASETS

    report = report_all_datasets()
    assert len(report["datasets"]) == 12
    assert report["n_supported"] + report["n_unsupported"] == 12
    # Every row must have a dataset name and a reason string, whether
    # supported or not -- this is the exact contract the task's final
    # report (item 2: "datasets still blocked and exact reason") depends on.
    for row in report["datasets"]:
        assert row["dataset"] in HACKATHON_DATASETS
        assert isinstance(row["reason"], str) and row["reason"]
