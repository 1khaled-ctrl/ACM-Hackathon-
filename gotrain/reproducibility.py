"""Captures the installed versions of this project's ML dependencies at the
moment a training run actually executes, so a run's metrics.json records
not just its config/seed but the exact dependency versions it ran against
-- requirements.txt pins what SHOULD be installed, this records what
ACTUALLY was, which can drift (a lab reinstalling into a stale venv, a
teammate's local environment, etc.).

Pure stdlib (importlib.metadata) -- no torch import, so this is fully
testable without the ML stack installed.
"""

from importlib import metadata

TRACKED_PACKAGES = [
    "torch", "torchvision", "timm", "segmentation_models_pytorch",
    "transformers", "geobench",
]


def capture_dependency_versions() -> dict:
    """{"torch": "2.9.0", "transformers": None, ...} -- None for any
    package not installed in the current environment, never raises.
    """
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions
