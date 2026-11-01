"""Resolve a ledger run_id to its saved checkpoint / metrics on disk.

A run's artifacts can live in two places:
  - runs/<dataset>/<run_id>/            (the training template's scratch dir)
  - models/<dataset>/<version_id>/      (the durable registry copy, keyed by
                                         the ledger record's version_id)

Before the run-id fix in gotrain/tools/training.py, the ledger run_id was a
different uuid from the template's run directory name, so every lookup of
runs/<dataset>/<ledger_run_id>/ failed for real runs. The registry copy is
the one location that has always been correct, so it is the fallback here,
which keeps runs recorded before that fix resolvable too.
"""

from pathlib import Path
from typing import Optional

from gotrain import ledger, registry

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"


def _registry_dir(dataset: str, run_id: str) -> Optional[Path]:
    record = ledger.get_run(run_id)
    if record is None or record.get("dataset") != dataset or not record.get("version_id"):
        return None
    return registry.get_version_dir(dataset, record["version_id"])


def checkpoint_path(dataset: str, run_id: str, runs_dir: Optional[Path] = None) -> Path:
    runs_dir = runs_dir or RUNS_DIR
    candidates = [runs_dir / dataset / run_id / "best_model.pt"]
    version_dir = _registry_dir(dataset, run_id)
    if version_dir is not None:
        candidates.append(version_dir / "model.pt")
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(
        f"No checkpoint found for run {run_id!r} of {dataset!r}. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + ". The run may have timed out or crashed before saving one."
    )


def load_normalization(dataset: str, run_id: str, runs_dir: Optional[Path] = None):
    """(mean, std) numpy arrays the run was trained with, from its
    metrics.json, or None for runs recorded before the templates saved them
    (callers then recompute from the TRAIN split as before)."""
    import json

    import numpy as np

    path = metrics_path(dataset, run_id, runs_dir)
    if path is None:
        return None
    try:
        norm = json.loads(path.read_text(encoding="utf-8")).get("normalization")
    except (ValueError, OSError):
        return None
    if not norm or "mean" not in norm or "std" not in norm:
        return None
    return np.asarray(norm["mean"], dtype=np.float64), np.asarray(norm["std"], dtype=np.float64)


def metrics_path(dataset: str, run_id: str, runs_dir: Optional[Path] = None) -> Optional[Path]:
    runs_dir = runs_dir or RUNS_DIR
    candidates = [runs_dir / dataset / run_id / "metrics.json"]
    version_dir = _registry_dir(dataset, run_id)
    if version_dir is not None:
        candidates.append(version_dir / "metrics.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
