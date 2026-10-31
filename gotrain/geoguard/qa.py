"""GeoGuard QA: deterministic post-preprocessing validation.

Every check here runs against an ALREADY-NORMALIZED batch -- the exact
(N, C, H, W) float tensor + (N,) label vector or (N, H, W) mask tensor a
training template would feed the model -- never against raw pixels
straight off disk. The goal is to catch a normalization step that produced
something the model can't consume (wrong channel count, NaNs introduced,
label/image misalignment, out-of-range mask values), not to re-derive the
diagnostics pass in diagnostics.py.

If a check fails, the caller must reject the preprocessing variant and keep
using the original/last-known-good one -- this module only reports
pass/fail + problems; it never mutates or "fixes" the data itself.
"""

import numpy as np


def check_classification_batch(images: np.ndarray, labels: np.ndarray, n_bands: int, n_classes: int) -> dict:
    problems = []

    if images.ndim != 4:
        problems.append(f"expected a 4D (N,C,H,W) image batch, got shape {images.shape}")
    elif images.shape[1] != n_bands:
        problems.append(f"expected {n_bands} channels, got {images.shape[1]}")

    if images.size and np.isnan(images).any():
        problems.append("NaN values present after normalization")
    if images.size and np.isinf(images).any():
        problems.append("Inf values present after normalization")

    if labels.ndim != 1:
        problems.append(f"expected a 1D label vector, got shape {labels.shape}")
    elif images.ndim == 4 and len(labels) != images.shape[0]:
        problems.append(f"image/label count mismatch: {images.shape[0]} images vs {len(labels)} labels")

    if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= n_classes):
        problems.append(
            f"label values out of range [0, {n_classes}): min={int(labels.min())}, max={int(labels.max())}"
        )

    return {"passed": not problems, "problems": problems}


def check_segmentation_batch(images: np.ndarray, masks: np.ndarray, n_bands: int, n_classes: int) -> dict:
    problems = []

    if images.ndim != 4:
        problems.append(f"expected a 4D (N,C,H,W) image batch, got shape {images.shape}")
    elif images.shape[1] != n_bands:
        problems.append(f"expected {n_bands} channels, got {images.shape[1]}")

    if images.size and np.isnan(images).any():
        problems.append("NaN values present after normalization")
    if images.size and np.isinf(images).any():
        problems.append("Inf values present after normalization")

    if masks.ndim != 3:
        problems.append(f"expected a 3D (N,H,W) mask batch, got shape {masks.shape}")
    elif images.ndim == 4 and (images.shape[0] != masks.shape[0] or images.shape[2:] != masks.shape[1:]):
        problems.append(f"image/mask shape mismatch: images {images.shape} vs masks {masks.shape}")

    if masks.size and (int(masks.min()) < 0 or int(masks.max()) >= n_classes):
        problems.append(
            f"mask values out of range [0, {n_classes}): min={int(masks.min())}, max={int(masks.max())}"
        )

    return {"passed": not problems, "problems": problems}
