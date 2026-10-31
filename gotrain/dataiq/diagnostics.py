"""DataIQ structural diagnostics: per-SAMPLE (not aggregate -- contrast
with gotrain.geoguard.diagnostics, which reports dataset-wide statistics)
detection of corrupted/invalid training examples and exact duplicates.

Every check here is deterministic and model-free. Model-ASSISTED signals
(suspiciously high per-example training loss, confident-but-wrong
predictions) live in gotrain/dataiq/loss_scan.py instead -- a
fundamentally weaker, non-proof kind of evidence that this module
deliberately keeps separate.
"""

import hashlib
from typing import Optional

import numpy as np

SEVERITY_HIGH = "high"      # objectively corrupted/invalid -- eligible for auto-quarantine
SEVERITY_NONE = "none"


def scan_samples(images: np.ndarray, labels: np.ndarray, task_type: str, n_classes: int) -> list:
    """Return one record per sample: {"sample_id", "severity", "problems"}.

    `images` is (N, H, W, C). `labels` is (N,) ints for classification or
    (N, H, W) ints for segmentation. `sample_id` is the index of a sample
    within `images`/`labels` -- see gotrain/dataiq/manifest.py's module
    docstring for exactly what that index means and its limits.
    """
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"images/labels count mismatch: {images.shape[0]} vs {labels.shape[0]}")

    records = []
    for i in range(images.shape[0]):
        problems = []
        image = images[i]

        if not np.all(np.isfinite(image)):
            n_nan = int(np.isnan(image).sum())
            n_inf = int(np.isinf(image).sum())
            problems.append(f"{n_nan} NaN / {n_inf} Inf pixel value(s)")
        elif image.size and float(np.std(image)) < 1e-6:
            # Only checked when the image is fully finite -- std() on data
            # containing NaN/Inf is itself NaN/Inf, which would never be
            # < 1e-6, so this elif never silently hides the NaN/Inf case.
            problems.append("image is constant (zero variance) across every pixel/band")

        if task_type == "segmentation":
            mask = labels[i]
            if mask.shape != image.shape[:2]:
                problems.append(f"mask shape {mask.shape} does not match image shape {image.shape[:2]}")
            elif mask.size and (int(mask.min()) < 0 or int(mask.max()) >= n_classes):
                problems.append(
                    f"mask values out of range [0, {n_classes}): min={int(mask.min())}, max={int(mask.max())}"
                )
        else:
            label = int(labels[i])
            if label < 0 or label >= n_classes:
                problems.append(f"label {label} out of range [0, {n_classes})")

        records.append({
            "sample_id": i,
            "severity": SEVERITY_HIGH if problems else SEVERITY_NONE,
            "problems": problems,
        })
    return records


def find_exact_duplicates(images: np.ndarray) -> list:
    """Group sample indices by exact pixel-content hash.

    Returns a list of groups (each a list of >=2 sample indices) whose
    pixel content is byte-identical. Never flags visually-similar-but-
    distinct samples (e.g. two different agricultural fields) -- only
    true, exact duplicates. Near-duplicate detection (perceptual hashing,
    embedding distance) is explicitly out of scope here; see the design
    brief's own P2 list.
    """
    hashes: dict = {}
    for i in range(images.shape[0]):
        digest = hashlib.sha256(np.ascontiguousarray(images[i]).tobytes()).hexdigest()
        hashes.setdefault(digest, []).append(i)
    return [group for group in hashes.values() if len(group) > 1]


def summarize_scan(structural_records: list, duplicate_groups: list) -> dict:
    """Roll per-sample structural records + duplicate groups into one
    summary dict suitable for a GeoGuard/DataIQ report or dashboard card.
    """
    high_severity = [r for r in structural_records if r["severity"] == SEVERITY_HIGH]
    return {
        "n_samples_scanned": len(structural_records),
        "n_invalid": len(high_severity),
        "invalid_sample_ids": [r["sample_id"] for r in high_severity],
        "n_duplicate_groups": len(duplicate_groups),
        "duplicate_sample_ids": [i for group in duplicate_groups for i in group],
    }
