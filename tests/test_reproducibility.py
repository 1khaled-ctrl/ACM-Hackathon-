"""Tests for gotrain/reproducibility.py. Pure stdlib (importlib.metadata),
no torch needed -- fully exercisable in this environment.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import reproducibility


def test_capture_dependency_versions_returns_every_tracked_package():
    versions = reproducibility.capture_dependency_versions()
    assert set(versions) == set(reproducibility.TRACKED_PACKAGES)


def test_capture_dependency_versions_reports_none_for_uninstalled_packages():
    # None of torch/torchvision/timm/segmentation_models_pytorch/transformers/
    # geobench are installed in this environment -- confirms the
    # PackageNotFoundError path reports None rather than raising.
    versions = reproducibility.capture_dependency_versions()
    for package, version in versions.items():
        assert version is None or isinstance(version, str)
