"""Shared eager-loading helpers for both training templates (and every tool
that reuses their build_dataset_arrays: StressLab, final_eval, DataIQ).

Fixes found in the pre-submission audit:
  - Mixed-resolution datasets (m-NeonTree: RGB 400x400, Canopy Height 40x40,
    Neon hyperspectral 40x40x369) crashed in geobench's pack_to_3d because it
    was called with resample=False. pack_sample always resamples to the
    largest band shape (a no-op when every band already matches).
  - Loading a split into a Python list and then np.stack-ing it held two full
    copies at peak; build_dataset_arrays now preallocates instead.
  - Normalization stats cast the WHOLE train array to float64 (another full
    copy, 2x the float32 size); band_stats streams it in chunks instead.
  - A slow split could consume the whole per-run subprocess timeout before
    training started; check_loading_budget fails fast with a concrete
    max_*_samples suggestion instead of the process being SIGKILLed silently.
"""

import time
from typing import Optional

import numpy as np

# Data loading may use at most this fraction of the run's subprocess timeout.
LOADING_PHASE_BUDGET_FRACTION = 0.5
# Training batches stop at this fraction of the timeout, leaving the rest for
# validation, saving the checkpoint and printing the result line.
TRAINING_CUTOFF_FRACTION = 0.75
LOADING_DEADLINE_CHECK_EVERY = 1
# (samples loaded, tolerance) points at which the whole split's load time is
# projected from the measured rate. The early check only fires on a gross
# overrun (2x), since the first few reads can be slow from a cold cache.
PROJECTION_POINTS = ((8, 2.0), (64, 1.0))


class DataLoadingBudgetExceeded(RuntimeError):
    """Loading alone would exceed the run's time budget."""


def pack_sample(sample, band_names, cast=None) -> np.ndarray:
    arr, _ = sample.pack_to_3d(band_names=band_names, resample=True)
    if cast is not None:
        return arr.astype(cast, copy=False)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr


def check_loading_budget(split: str, k: int, n: int, started_at: float, loading_deadline: Optional[float]) -> None:
    """Call before loading sample k of n. Raises DataLoadingBudgetExceeded when
    the deadline has passed, or (at PROJECTION_POINTS) when the measured
    per-sample rate projects past it."""
    if loading_deadline is None:
        return
    now = time.time()
    if k % LOADING_DEADLINE_CHECK_EVERY == 0 and now > loading_deadline:
        raise DataLoadingBudgetExceeded(
            f"Loading the {split!r} split is taking too long: only {k}/{n} samples loaded before "
            "the data-loading time budget was reached. Retry with a smaller "
            "max_train_samples/max_val_samples, or a larger run budget."
        )
    tolerance = dict(PROJECTION_POINTS).get(k)
    if tolerance is not None and k < n:
        per_sample = (now - started_at) / k
        if per_sample * (n - k) > (loading_deadline - now) * tolerance:
            available = max(loading_deadline - started_at, 0.0)
            fits = int(available / per_sample * 0.9) if per_sample > 0 else n
            knob = "max_train_samples" if split == "train" else "max_val_samples"
            raise DataLoadingBudgetExceeded(
                f"Loading the {split!r} split would take ~{per_sample * n:.0f}s "
                f"({per_sample * 1000:.0f} ms/sample x {n} samples) but only ~{available:.0f}s of the "
                f"data-loading budget remain. Retry with {knob} <= {fits}, or a larger run budget."
            )


def band_stats(images: np.ndarray, chunk_values: int = 1 << 24):
    """Per-channel population mean/std of an (N, H, W, C) array, accumulated
    in float64 chunks (Chan et al. parallel variance) so no full float64 copy
    of the array is ever made. Constant channels get std 1.0."""
    channels = images.shape[-1]
    flat = images.reshape(-1, channels)
    rows = max(1, chunk_values // channels)
    count = 0
    mean = np.zeros(channels, dtype=np.float64)
    m2 = np.zeros(channels, dtype=np.float64)
    for start in range(0, flat.shape[0], rows):
        chunk = flat[start:start + rows].astype(np.float64)
        n_b = chunk.shape[0]
        mean_b = chunk.mean(axis=0)
        m2_b = ((chunk - mean_b) ** 2).sum(axis=0)
        delta = mean_b - mean
        total = count + n_b
        mean = mean + delta * n_b / total
        m2 = m2 + m2_b + delta ** 2 * count * n_b / total
        count = total
    std = np.sqrt(m2 / max(count, 1))
    std[std < 1e-6] = 1.0
    return mean, std
