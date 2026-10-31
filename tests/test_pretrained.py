"""Tests for the pretrained-model registry (gotrain/pretrained.py).

list_compatible_models, estimate_model_resources, and the verification
cache are pure Python/file I/O and fully exercised here.
validate_model_compatibility/load_pretrained_backbone need torch (and
timm/segmentation_models_pytorch for the "timm"/"smp" loader families) --
not installed in this environment; their happy-path logic is reviewed by
inspection and mirrors the already-tested build_model call sites in the
training templates, but needs a real run on the SageMaker lab to confirm.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import pretrained


# ---------------------------------------------------------------------------
# Catalog shape / honesty invariants
# ---------------------------------------------------------------------------


def test_every_catalog_entry_starts_unverified():
    for model_id, entry in pretrained.CATALOG.items():
        assert entry["compatibility_verified"] is False, f"{model_id} must start unverified"


def test_every_catalog_entry_has_required_shape_fields():
    required = {
        "model_id", "architecture", "pretraining_source", "pretraining_domain",
        "supported_modalities", "expected_channels", "input_band_order", "normalization",
        "parameter_count", "license", "checkpoint_revision", "compatibility_verified",
    }
    for model_id, entry in pretrained.CATALOG.items():
        missing = required - set(entry)
        assert not missing, f"{model_id} is missing fields: {missing}"


def test_aspirational_entries_have_no_loader_and_are_honest_about_unknowns():
    """The torchgeo/dinov2/Prithvi candidates must never claim a loader
    exists, and Prithvi's genuinely-unconfirmed fields must stay null
    rather than guessed.
    """
    for model_id in ("torchgeo/resnet18_sentinel2_all_moco", "huggingface/facebook-dinov2-small",
                      "ibm-nasa/prithvi-eo-2.0-tiny"):
        assert pretrained.CATALOG[model_id]["loader_family"] is None

    prithvi = pretrained.CATALOG["ibm-nasa/prithvi-eo-2.0-tiny"]
    assert prithvi["parameter_count"] is None
    assert prithvi["license"] is None
    assert prithvi["checkpoint_revision"] is None


def test_timm_and_smp_backed_entries_have_a_loader():
    for model_id in ("timm/resnet18", "timm/resnet50", "timm/efficientnet_b0", "timm/convnext_tiny"):
        assert pretrained.CATALOG[model_id]["loader_family"] == "timm"
    for model_id in ("smp/resnet34", "smp/resnet50", "smp/efficientnet-b3"):
        assert pretrained.CATALOG[model_id]["loader_family"] == "smp"


# ---------------------------------------------------------------------------
# list_compatible_models
# ---------------------------------------------------------------------------


def test_list_compatible_models_filters_by_task_type():
    classification_models = pretrained.list_compatible_models("classification")
    assert all("classification" in m["task_types"] for m in classification_models)
    assert any(m["model_id"] == "timm/resnet18" for m in classification_models)
    assert not any(m["model_id"] == "smp/resnet34" for m in classification_models)

    segmentation_models = pretrained.list_compatible_models("segmentation")
    assert all("segmentation" in m["task_types"] for m in segmentation_models)
    assert any(m["model_id"] == "smp/resnet34" for m in segmentation_models)


def test_list_compatible_models_without_dataset_has_no_verification_key():
    models = pretrained.list_compatible_models("classification")
    assert all("verification" not in m for m in models)


def test_list_compatible_models_reads_cached_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(pretrained, "VERIFICATION_DIR", tmp_path / "verification")
    pretrained._write_verification("timm/resnet18", "m-eurosat", {"compatible": True, "reason": "forward pass succeeded"})

    models = pretrained.list_compatible_models("classification", dataset="m-eurosat")
    resnet18 = next(m for m in models if m["model_id"] == "timm/resnet18")
    assert resnet18["verification"]["compatible"] is True

    efficientnet = next(m for m in models if m["model_id"] == "timm/efficientnet_b0")
    assert efficientnet["verification"] is None  # never checked for this dataset


# ---------------------------------------------------------------------------
# estimate_model_resources
# ---------------------------------------------------------------------------


def test_estimate_model_resources_labels_itself_as_an_estimate():
    result = pretrained.estimate_model_resources("timm/resnet18")
    assert result["parameter_count"] == 11_700_000
    assert "ESTIMATE" in result["estimate_label"]
    assert result["estimated_training_memory_mb"] > 0


def test_estimate_model_resources_handles_unknown_parameter_count():
    result = pretrained.estimate_model_resources("ibm-nasa/prithvi-eo-2.0-tiny")
    assert result["parameter_count"] is None
    assert result["estimated_training_memory_mb"] is None
    assert "UNKNOWN" in result["estimate_label"]


def test_estimate_model_resources_rejects_unknown_model_id():
    with pytest.raises(KeyError):
        pretrained.estimate_model_resources("not-a-real-model")


# ---------------------------------------------------------------------------
# Verification cache round-trip
# ---------------------------------------------------------------------------


def test_verification_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(pretrained, "VERIFICATION_DIR", tmp_path / "verification")
    assert pretrained._read_verification("timm/resnet18", "m-eurosat") is None

    pretrained._write_verification("timm/resnet18", "m-eurosat", {"compatible": True, "reason": "ok"})
    cached = pretrained._read_verification("timm/resnet18", "m-eurosat")
    assert cached == {"compatible": True, "reason": "ok"}

    # model_id containing "/" must not create a stray nested directory
    assert (tmp_path / "verification" / "timm__resnet18__m-eurosat.json").exists()


# ---------------------------------------------------------------------------
# validate_model_compatibility -- the parts that don't need torch
# ---------------------------------------------------------------------------


def test_validate_model_compatibility_rejects_wrong_task_type(tmp_path, monkeypatch):
    monkeypatch.setattr(pretrained, "VERIFICATION_DIR", tmp_path / "verification")
    result = pretrained.validate_model_compatibility(
        "timm/resnet18", dataset="m-pv4ger-seg", task_type="segmentation", n_bands=3, n_classes=2,
    )
    assert result["compatible"] is False
    assert "does not support" in result["reason"]


def test_validate_model_compatibility_reports_no_loader_for_aspirational_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(pretrained, "VERIFICATION_DIR", tmp_path / "verification")
    result = pretrained.validate_model_compatibility(
        "torchgeo/resnet18_sentinel2_all_moco", dataset="m-eurosat", task_type="classification",
        n_bands=13, n_classes=10,
    )
    assert result["compatible"] is False
    assert "no loader implemented" in result["reason"]


def test_validate_model_compatibility_rejects_unknown_model_id(tmp_path, monkeypatch):
    monkeypatch.setattr(pretrained, "VERIFICATION_DIR", tmp_path / "verification")
    with pytest.raises(KeyError):
        pretrained.validate_model_compatibility("not-a-real-model", "m-eurosat", "classification", 13, 10)


# ---------------------------------------------------------------------------
# load_pretrained_backbone -- the parts that don't need torch
# ---------------------------------------------------------------------------


def test_load_pretrained_backbone_raises_not_implemented_for_aspirational_entries():
    with pytest.raises(NotImplementedError):
        pretrained.load_pretrained_backbone(
            "ibm-nasa/prithvi-eo-2.0-tiny", n_bands=13, n_classes=10, task_type="classification",
        )


def test_load_pretrained_backbone_rejects_unknown_model_id():
    with pytest.raises(KeyError):
        pretrained.load_pretrained_backbone("not-a-real-model", n_bands=3, n_classes=2, task_type="classification")
