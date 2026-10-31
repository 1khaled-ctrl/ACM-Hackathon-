"""Versioned, reversible training-data manifests (DataIQ).

A manifest never touches, deletes, or copies original data -- it's just
a record of which TRAIN-split sample indices are excluded for one
dataset, applied only at data-loading time inside the training templates
(their build_dataset_arrays gains an `excluded_ids` parameter that simply
skips those indices while reading the same underlying GEO-Bench files
everyone else reads). Validation and test splits are never referenced
anywhere in this module -- there is no function here that can accept or
act on anything but a train-split index, by construction, so a manifest
cannot violate split isolation even by mistake.

"sample_id" throughout DataIQ means the integer index of a sample within
GEO-Bench's own deterministic train-split ordering for that dataset
(task_specs.get_dataset(split="train")[i]). GEO-Bench doesn't expose a
more stable per-sample identifier, and that ordering is itself
deterministic given the same downloaded data, so an index is a
reproducible-enough reference for this project -- recorded explicitly in
every manifest so it's never ambiguous later.

A manifest's exclusions must already be APPROVED data-quality decisions
(gotrain.dataiq.decision) before the manifest can be created at all --
enforced here, not discovered later when a training run tries to use an
unapproved exclusion. The one action that can reach this gate already
pre-approved is QUARANTINE_INVALID_SAMPLE, auto-approved under policy for
objectively corrupted samples (see decision.py) -- never
EXCLUDE_FROM_TRAINING_CANDIDATE or CORRECT_LABEL_CANDIDATE, which always
need a human first.
"""

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from gotrain.logging_utils import LOGS_DIR, log_event

MANIFESTS_DIR = LOGS_DIR / "dataiq" / "manifests"
BASE_MANIFEST_ID = "base"  # implicit: no exclusions, the full original train split


class ManifestError(ValueError):
    pass


class ManifestApprovalError(ManifestError):
    """Raised when create_manifest is asked to exclude a sample whose
    removal was never approved (or was rejected, or is still pending) --
    an exclusion must be approved BEFORE the manifest using it can exist.
    """


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_dir(dataset: str) -> Path:
    return MANIFESTS_DIR / dataset


def _manifest_path(dataset: str, manifest_id: str) -> Path:
    return _dataset_dir(dataset) / f"{manifest_id}.json"


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".manifest_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def get_manifest(dataset: str, manifest_id: str) -> Optional[dict]:
    if manifest_id == BASE_MANIFEST_ID:
        return {
            "manifest_id": BASE_MANIFEST_ID, "dataset": dataset, "base_manifest_id": None,
            "newly_excluded_sample_ids": [], "excluded_sample_ids": [], "n_excluded": 0,
            "reason": "original, unmodified train split", "created_by": "system", "created_at": None,
        }
    path = _manifest_path(dataset, manifest_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest_exists(dataset: str, manifest_id: Optional[str]) -> None:
    """Raise ManifestError if `manifest_id` is given but unknown for
    `dataset`. None or "base" are always valid (no exclusions) -- called
    from gotrain/tools/training.py before any training subprocess runs.
    """
    if manifest_id is None or manifest_id == BASE_MANIFEST_ID:
        return
    if get_manifest(dataset, manifest_id) is None:
        raise ManifestError(f"no manifest {manifest_id!r} found for dataset {dataset!r}")


def list_manifests(dataset: str) -> list:
    """All manifests for `dataset`, base first, then newest-created last."""
    base = get_manifest(dataset, BASE_MANIFEST_ID)
    directory = _dataset_dir(dataset)
    others = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            others.append(json.loads(path.read_text(encoding="utf-8")))
    return [base] + sorted(others, key=lambda m: m["created_at"])


def get_excluded_ids(dataset: str, manifest_id: Optional[str]) -> set:
    manifest = get_manifest(dataset, manifest_id or BASE_MANIFEST_ID)
    if manifest is None:
        raise ManifestError(f"no manifest {manifest_id!r} found for dataset {dataset!r}")
    return set(manifest["excluded_sample_ids"])


def create_manifest(
    dataset: str,
    excluded_sample_ids,
    reason: str,
    created_by: str = "agent",
    base_manifest_id: str = BASE_MANIFEST_ID,
) -> dict:
    """Create a new manifest excluding `excluded_sample_ids` (ints) from
    `dataset`'s TRAIN split, layered on top of `base_manifest_id`'s own
    exclusions (manifests compose -- e.g. "the auto-quarantine set plus
    one more engineer-approved exclusion"). Every NEWLY added id must
    already have an APPROVED EXCLUDE_FROM_TRAINING_CANDIDATE or
    QUARANTINE_INVALID_SAMPLE decision (gotrain.dataiq.decision) --
    raises ManifestApprovalError otherwise, so an unapproved exclusion
    can never enter a manifest.
    """
    from gotrain.dataiq import decision as dataiq_decision

    base_excluded = get_excluded_ids(dataset, base_manifest_id)
    new_ids = {int(i) for i in excluded_sample_ids}

    for sample_id in sorted(new_ids - base_excluded):
        approved_action = dataiq_decision.get_approved_action(dataset, sample_id)
        if approved_action is None:
            raise ManifestApprovalError(
                f"sample {sample_id} for dataset {dataset!r} has no approved exclusion "
                "decision -- propose_action() must be called and approved by an engineer "
                "before this sample can be included in a manifest."
            )

    all_excluded = sorted(base_excluded | new_ids)
    manifest = {
        "manifest_id": str(uuid.uuid4()),
        "dataset": dataset,
        "base_manifest_id": base_manifest_id,
        "newly_excluded_sample_ids": sorted(new_ids - base_excluded),
        "excluded_sample_ids": all_excluded,
        "n_excluded": len(all_excluded),
        "reason": reason,
        "created_by": created_by,
        "created_at": _now(),
        "sample_id_meaning": "integer index into the dataset's deterministic TRAIN-split ordering",
    }
    _write_json_atomic(_manifest_path(dataset, manifest["manifest_id"]), manifest)
    log_event(
        "manifest_created", dataset=dataset, manifest_id=manifest["manifest_id"],
        base_manifest_id=base_manifest_id, n_excluded=manifest["n_excluded"],
        newly_excluded_sample_ids=manifest["newly_excluded_sample_ids"], reason=reason, created_by=created_by,
    )
    return manifest
