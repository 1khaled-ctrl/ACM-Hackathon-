"""GeoGuard diagnostics: dataset-aware, sensor-aware inspection of a
GEO-Bench task's TRAIN split, before any training run.

Every pixel-level statistic here is computed from a bounded sample of the
TRAIN split only -- never validation, never test. This module never fits or
returns anything that gets *applied* to validation/test data (that remains
the training templates' job, via compute_train_band_stats); it only reports
what the train split looks like, so data-leakage can't enter through this
door even accidentally.

Reuses the exact same band-count/image-shape/label metadata GEO-Bench
exposes, and the same split-loading pattern as
gotrain/templates/train_classification.py and train_segmentation.py
(sample.pack_to_3d / sample.label), since this needs to walk the same
samples they would -- just for inspection, not training.
"""

import os
from pathlib import Path

import numpy as np

BENCHMARK_DIRS = ["classification_v1.0", "segmentation_v1.0"]

# Name-pattern markers for Sentinel-1 SAR channels, as they actually appear
# in GEO-Bench's m-so2sat band names (see data/dataset_summary.md): "VH"/"VV"
# (cross/co-polarization) and "LEE Filtered" (a SAR speckle-reduction filter
# with no optical equivalent). Anything not matching is treated as optical.
SAR_BAND_MARKERS = ("vh", "vv", "lee filtered")

# Fixed, documented context a statistics pass can't derive on its own --
# kept here (not invented per-run) so GeoGuard's output is reproducible and
# traceable to a source, per the "do not invent missing metadata" rule.
DATASET_NOTES = {
    "m-eurosat": {
        "sensor": "Sentinel-2 multispectral (13 bands)",
        "warnings": [
            "geo-bench-1.0 stores incorrect band names/wavelengths for most "
            "m-eurosat channels (the pixel data itself is unaffected). "
            "Selecting bands by name silently grabs the wrong channel -- "
            "use all bands (bands=null). "
            "See https://github.com/ServiceNow/geo-bench/issues/28."
        ],
    },
    "m-so2sat": {
        "sensor": "Sentinel-1 SAR + Sentinel-2 optical, fused (18 bands)",
        "warnings": [],
    },
    "m-pv4ger-seg": {"sensor": "RGB aerial imagery", "warnings": []},
    "m-nz-cattle": {"sensor": "RGB aerial imagery", "warnings": []},
}


def classify_bands(band_names: list) -> dict:
    """Split band names into SAR vs. optical by name pattern -- the only
    modality signal available without a pixel-level physical model. Bands
    matching no SAR marker are treated as optical/RGB.
    """
    sar, optical = [], []
    for name in band_names:
        lname = name.lower()
        if any(marker in lname for marker in SAR_BAND_MARKERS):
            sar.append(name)
        else:
            optical.append(name)
    return {
        "sar_bands": sar,
        "optical_bands": optical,
        "is_multimodal": bool(sar) and bool(optical),
    }


def compute_pixel_diagnostics(images: np.ndarray, band_names: list) -> dict:
    """Per-band pixel statistics over an (N, H, W, C) array.

    Flags, per band: dynamic range, possible saturation/clipping (a spike of
    pixels sitting exactly at the observed min/max -- real continuous sensor
    readings rarely repeat an extreme value that often), and near-constant
    bands. NaN/Inf pixels are counted but excluded from mean/std/min/max so
    one corrupt pixel can't make an entire band's statistics meaningless.
    """
    if images.ndim != 4:
        raise ValueError(f"expected (N, H, W, C) images, got shape {images.shape}")
    n, h, w, c = images.shape
    if c != len(band_names):
        raise ValueError(f"{c} channels in images but {len(band_names)} band_names given")

    flat = images.reshape(-1, c).astype(np.float64)
    nan_count = int(np.isnan(flat).sum())
    inf_count = int(np.isinf(flat).sum())

    per_band = []
    for i, name in enumerate(band_names):
        col = flat[:, i]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            per_band.append({"band": name, "all_invalid": True})
            continue
        band_min, band_max = float(finite.min()), float(finite.max())
        mean, std = float(finite.mean()), float(finite.std())
        at_max = float((finite == band_max).mean())
        at_min = float((finite == band_min).mean())
        per_band.append({
            "band": name,
            "min": band_min,
            "max": band_max,
            "mean": mean,
            "std": std,
            "pct_at_max": at_max,
            "pct_at_min": at_min,
            "possible_saturation": bool(at_max > 0.005 or at_min > 0.005),
            "near_constant": bool(std < 1e-6),
        })

    return {
        "n_samples": n,
        "height": h,
        "width": w,
        "n_channels": c,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "per_band": per_band,
    }


def compute_class_diagnostics(labels: np.ndarray, class_names: list, is_pixelwise: bool) -> dict:
    """Class-count (classification) or pixel-count (segmentation) frequency
    diagnostics -- the two are the same computation over a differently
    shaped `labels` array (per-sample vs. per-pixel), so one function
    serves both task types.
    """
    n_classes = len(class_names)
    counts = np.bincount(labels.reshape(-1), minlength=n_classes).astype(np.float64)
    total = counts.sum()
    freqs = (counts / total).tolist() if total else [0.0] * n_classes

    rare_threshold = 0.02
    rare_classes = [class_names[c] for c, f in enumerate(freqs) if 0 < f < rare_threshold]
    absent_classes = [class_names[c] for c, f in enumerate(freqs) if f == 0.0]
    positive_freqs = [f for f in freqs if f > 0]
    imbalance_ratio = (max(positive_freqs) / min(positive_freqs)) if positive_freqs else None
    background_dominant = is_pixelwise and n_classes == 2 and freqs[0] > 0.9

    return {
        "n_classes": n_classes,
        "class_names": class_names,
        "counts": counts.astype(int).tolist(),
        "frequencies": freqs,
        "rare_classes": rare_classes,
        "classes_absent_from_sample": absent_classes,
        "imbalance_ratio": imbalance_ratio,
        "background_dominant": background_dominant,
        "unit": "pixels" if is_pixelwise else "samples",
    }


def _find_task_dir(dataset: str) -> Path:
    import geobench

    geo_bench_dir = Path(os.environ.get("GEO_BENCH_DIR", str(geobench.GEO_BENCH_DIR)))
    for benchmark_dir in BENCHMARK_DIRS:
        candidate = geo_bench_dir / benchmark_dir / dataset
        if candidate.is_dir() and (candidate / "task_specs.pkl").exists():
            return candidate
    raise ValueError(f"Dataset {dataset!r} not found under {geo_bench_dir}")


def load_train_sample(dataset: str, task_type: str, max_samples: int = 300):
    """Load up to `max_samples` TRAIN-split images (+ labels/masks) for
    inspection. Mirrors build_dataset_arrays in the training templates --
    same split ("train"), same sample.pack_to_3d/sample.label access -- but
    bounded and read-only: GeoGuard never trains anything and never opens
    validation or test.

    Returns (images, labels_or_masks, band_names, task_specs).
    """
    import geobench

    task_dir = _find_task_dir(dataset)
    task_specs = geobench.load_task_specs(task_dir)
    band_names = [b.name for b in task_specs.bands_info]

    ds = task_specs.get_dataset(split="train")
    n = min(len(ds), max_samples)

    images, labels = [], []
    for i in range(n):
        sample = ds[i]
        arr, _ = sample.pack_to_3d(band_names=band_names)
        images.append(arr.astype(np.float32))
        if task_type == "segmentation":
            labels.append(sample.label.data)
        else:
            labels.append(int(sample.label))

    images = np.stack(images, axis=0)
    labels = np.stack(labels, axis=0) if task_type == "segmentation" else np.array(labels, dtype=np.int64)
    return images, labels, band_names, task_specs
