"""The perturbation battery StressLab evaluates a checkpoint under:
realistic nuisances a clean validation split never exercises, but a
deployed model will eventually see -- sensor noise, a dropped/dead
channel, a lighting/atmospheric shift, and simple geometric changes.

Every perturbation operates on (N, H, W, C) arrays -- the exact shape
gotrain.templates.train_classification/train_segmentation's
build_dataset_arrays already returns, BEFORE normalization (mean/std
division happens later, in the training templates' Dataset classes and
in gotrain.stresslab.runner's own normalize step) -- so these compose
directly into that same pipeline with no extra conversion step.

gaussian_noise/band_dropout/brightness_shift are scaled relative to each
band's OWN observed std (computed from the images passed in), never an
absolute pixel value -- GEO-Bench datasets span wildly different native
scales (SAR amplitude vs. 8/16-bit optical reflectance), so a fixed
absolute constant would be meaningless for one and negligible for
another. horizontal_flip/rotate_90 are exact, lossless, and have no
severity concept (there's nothing to scale): they're included once
each, not swept across severities, via SEVERITIES_FOR below.

Pure numpy -- no torch, no geobench I/O -- fully testable with synthetic
arrays.
"""

from typing import Optional

import numpy as np

SEVERITIES = ["mild", "moderate", "severe"]

# Judgment calls, like calibration.py's ECE thresholds -- named here, not
# buried as magic numbers, specifically so they can be revisited later.
GAUSSIAN_NOISE_STD_FRACTION = {"mild": 0.05, "moderate": 0.15, "severe": 0.30}
BRIGHTNESS_SHIFT_FRACTION = {"mild": 0.05, "moderate": 0.15, "severe": 0.30}
BAND_DROPOUT_COUNT = {"mild": 1, "moderate": 2, "severe": 3}


def _band_std(images: np.ndarray) -> np.ndarray:
    return images.reshape(-1, images.shape[-1]).astype(np.float64).std(axis=0)


def add_gaussian_noise(images: np.ndarray, severity: str, seed: int = 0) -> np.ndarray:
    """Independent per-pixel-per-band Gaussian noise, std = severity
    fraction of that band's own observed std (simulates sensor noise).
    """
    rng = np.random.default_rng(seed)
    noise_std = GAUSSIAN_NOISE_STD_FRACTION[severity] * _band_std(images)
    noise = rng.normal(0.0, 1.0, size=images.shape) * noise_std
    return (images.astype(np.float64) + noise).astype(np.float32)


def dropout_bands(images: np.ndarray, severity: str, seed: int = 0) -> np.ndarray:
    """Zeroes out `BAND_DROPOUT_COUNT[severity]` randomly chosen bands,
    the same bands across every sample (simulates a dead/missing sensor
    channel, not per-sample random dropout). Always leaves at least one
    band intact, even at "severe", so a 3-band RGB dataset never gets
    reduced to a fully-blank image.
    """
    rng = np.random.default_rng(seed)
    n_bands = images.shape[-1]
    n_drop = min(BAND_DROPOUT_COUNT[severity], max(n_bands - 1, 0))
    out = images.astype(np.float32).copy()
    if n_drop == 0:
        return out
    drop_indices = rng.choice(n_bands, size=n_drop, replace=False)
    out[..., drop_indices] = 0.0
    return out


def shift_brightness(images: np.ndarray, severity: str, seed: int = 0) -> np.ndarray:
    """Additive brightness/illumination shift, magnitude = severity
    fraction of each band's own observed std (simulates a lighting or
    atmospheric-correction shift between capture conditions).
    """
    shift = BRIGHTNESS_SHIFT_FRACTION[severity] * _band_std(images)
    return (images.astype(np.float64) + shift).astype(np.float32)


def horizontal_flip(images: np.ndarray, severity: Optional[str] = None, seed: int = 0) -> np.ndarray:
    """Exact, lossless horizontal flip -- a sanity check that the model
    hasn't learned a left/right orientation shortcut that real-world
    imagery (captured at any heading) won't respect. `severity` is
    accepted for interface uniformity with the scaled perturbations
    above but has no effect.
    """
    return np.flip(images, axis=2).astype(np.float32)


def rotate_90(images: np.ndarray, severity: Optional[str] = None, seed: int = 0) -> np.ndarray:
    """Exact, lossless 90-degree rotation -- same rationale as
    horizontal_flip. `severity` is accepted for interface uniformity but
    has no effect.
    """
    return np.rot90(images, k=1, axes=(1, 2)).astype(np.float32)


PERTURBATIONS = {
    "gaussian_noise": add_gaussian_noise,
    "band_dropout": dropout_bands,
    "brightness_shift": shift_brightness,
    "horizontal_flip": horizontal_flip,
    "rotation_90": rotate_90,
}

# Which severities to sweep for each perturbation -- the 3 scaled ones get
# the full SEVERITIES ladder; the two exact geometric transforms have
# nothing to scale, so they're evaluated once each under a single
# placeholder "applied" entry instead.
SEVERITIES_FOR = {
    "gaussian_noise": SEVERITIES,
    "band_dropout": SEVERITIES,
    "brightness_shift": SEVERITIES,
    "horizontal_flip": ["applied"],
    "rotation_90": ["applied"],
}
