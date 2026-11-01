"""Tests for the DINOv2 loader path in gotrain/pretrained.py.

Both guards tested here (missing band_names, unpinned hf_revision) raise
BEFORE importing torch/transformers at all, so they're fully exercisable
in this environment even though neither package is installed here. The
actual model construction (AutoModel.from_pretrained + the forward pass)
is NOT tested here -- it needs transformers, network access to Hugging
Face Hub, and a real pinned revision, none of which exist in this
project's local/CI environment. That part needs the SageMaker lab.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import pretrained

M_EUROSAT_BANDS = [
    "01 - Coastal aerosol", "02 - Blue", "03 - Green", "04 - Red",
    "05 - Vegetation Red Edge", "06 - Vegetation Red Edge", "07 - Vegetation Red Edge",
    "08 - NIR", "08A - Vegetation Red Edge", "09 - Water vapour",
    "10 - SWIR - Cirrus", "11 - SWIR", "12 - SWIR",
]


def test_load_pretrained_backbone_requires_band_names_for_dinov2():
    with pytest.raises(ValueError) as exc_info:
        pretrained.load_pretrained_backbone(
            "huggingface/facebook-dinov2-small", n_bands=13, n_classes=10, task_type="classification",
        )
    assert "band_names" in str(exc_info.value)


def test_load_dinov2_classification_backbone_refuses_without_pinned_revision():
    """The actual discipline being enforced: no silent fallback to an
    unpinned "main" revision. This must raise before transformers is
    even imported, so it's testable without the package installed.
    """
    entry = dict(pretrained.CATALOG["huggingface/facebook-dinov2-small"])
    assert entry["hf_revision"] is None  # confirms the catalog itself hasn't been pinned yet

    with pytest.raises(RuntimeError) as exc_info:
        pretrained.load_dinov2_classification_backbone(entry, M_EUROSAT_BANDS, n_classes=10)
    assert "hf_revision" in str(exc_info.value)
    assert "facebook/dinov2-small" in str(exc_info.value)


def test_load_dinov2_classification_backbone_would_proceed_past_the_guard_once_pinned():
    """Confirms the guard is specifically about the missing revision, not
    something else silently also broken -- pin a (fake, for this test
    only) revision and confirm the NEXT failure is the expected
    "transformers not installed" ModuleNotFoundError, not the RuntimeError
    above re-appearing or some unrelated error.
    """
    entry = dict(pretrained.CATALOG["huggingface/facebook-dinov2-small"])
    entry["hf_revision"] = "deadbeef"  # a placeholder SHA, purely to clear this one guard

    with pytest.raises(ModuleNotFoundError):
        pretrained.load_dinov2_classification_backbone(entry, M_EUROSAT_BANDS, n_classes=10)


def test_load_dinov2_classification_backbone_still_validates_band_names_when_pinned():
    entry = dict(pretrained.CATALOG["huggingface/facebook-dinov2-small"])
    entry["hf_revision"] = "deadbeef"

    with pytest.raises(ValueError) as exc_info:
        pretrained.load_dinov2_classification_backbone(entry, ["Blue", "Green"], n_classes=10)  # no Red band
    assert "red" in str(exc_info.value).lower()


def test_validate_model_compatibility_reports_incompatible_without_transformers_or_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(pretrained, "VERIFICATION_DIR", tmp_path / "verification")

    # End to end through the public API: dinov2 now HAS a loader, so it
    # must reach the try/except around load_pretrained_backbone and come
    # back as a structured "not compatible" result, never an unhandled
    # exception bubbling out of a tool call.
    result = pretrained.validate_model_compatibility(
        "huggingface/facebook-dinov2-small", dataset="m-eurosat", task_type="classification",
        n_bands=13, n_classes=10, band_names=M_EUROSAT_BANDS,
    )
    assert result["compatible"] is False
    assert "hf_revision" in result["reason"] or "transformers" in result["reason"].lower()
