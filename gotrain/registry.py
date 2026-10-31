"""Model registry: promotes a finished run's checkpoint into a durable,
versioned location under models/<dataset>/<version>/, separate from the
scratch runs/ directory (which exists only for the one process that wrote
it and isn't meant to be treated as a stable artifact store).

Each version directory holds:
  - config.json    (the run's config, copied from its runs/ directory)
  - metrics.json    (the run's metrics, copied from its runs/ directory)
  - model.pt        (the run's checkpoint, copied from its runs/ directory)

Versions are per-dataset, incrementing integers starting at 1
(models/<dataset>/v1/, v2/, ...). The ledger stores the resulting
version_id string on the run record, so a run can be traced to its
registry entry (and vice versa) without guessing a path.
"""

import json
import shutil
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


def _next_version(dataset: str) -> str:
    dataset_dir = MODELS_DIR / dataset
    if not dataset_dir.is_dir():
        return "v1"
    existing = [
        int(p.name[1:]) for p in dataset_dir.iterdir()
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
    ]
    return f"v{max(existing, default=0) + 1}"


def register_run(dataset: str, run_id: str, run_dir: str, checkpoint_path: Optional[str]) -> str:
    """Copy a finished run's config/metrics/checkpoint into
    models/<dataset>/<version>/ and return the version id (e.g. "v3").

    `run_dir` and `checkpoint_path` come straight from the training
    template's stdout JSON (added alongside val_metric specifically so
    this function has somewhere reliable to copy from). Missing files
    (e.g. no checkpoint was ever saved) are skipped rather than raising —
    registration still records whatever is available.
    """
    run_dir_path = Path(run_dir)
    version = _next_version(dataset)
    version_dir = MODELS_DIR / dataset / version
    version_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"dataset": dataset, "run_id": run_id, "version": version}

    config_src = run_dir_path / "config.json"
    if config_src.exists():
        shutil.copy2(config_src, version_dir / "config.json")

    metrics_src = run_dir_path / "metrics.json"
    if metrics_src.exists():
        shutil.copy2(metrics_src, version_dir / "metrics.json")

    if checkpoint_path and Path(checkpoint_path).exists():
        shutil.copy2(checkpoint_path, version_dir / "model.pt")
        manifest["checkpoint"] = "model.pt"
    else:
        manifest["checkpoint"] = None

    (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return version


def get_version_dir(dataset: str, version: str) -> Path:
    return MODELS_DIR / dataset / version


def list_versions(dataset: str) -> list:
    dataset_dir = MODELS_DIR / dataset
    if not dataset_dir.is_dir():
        return []
    return sorted(
        (p.name for p in dataset_dir.iterdir() if p.is_dir()),
        key=lambda v: int(v[1:]) if v[1:].isdigit() else 0,
    )
