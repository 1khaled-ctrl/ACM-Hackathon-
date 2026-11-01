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


def compute_class_diagnostics(
    labels: np.ndarray, class_names: list, is_pixelwise: bool, is_multi_label: bool = False
) -> dict:
    """Class-count (classification) or pixel-count (segmentation) frequency
    diagnostics -- the two are the same computation over a differently
    shaped `labels` array (per-sample vs. per-pixel), so one function
    serves both task types.

    `is_multi_label=True` (e.g. m-bigearthnet): `labels` is (N, n_classes)
    {0,1} multi-hot, not a per-sample/per-pixel class index -- bincount over
    a flattened multi-hot array would just count 0s and 1s, not class
    frequencies, so this computes "fraction of samples where each class is
    active" directly from the multi-hot columns instead. Classes aren't
    mutually exclusive here, so frequencies don't sum to 1 the way the
    single-label/segmentation case does -- that's expected, not a bug.
    """
    n_classes = len(class_names)
    if is_multi_label:
        freqs = labels.mean(axis=0).astype(np.float64).tolist()
        counts = labels.sum(axis=0).astype(int).tolist()
    else:
        counts_arr = np.bincount(labels.reshape(-1), minlength=n_classes).astype(np.float64)
        total = counts_arr.sum()
        freqs = (counts_arr / total).tolist() if total else [0.0] * n_classes
        counts = counts_arr.astype(int).tolist()

    rare_threshold = 0.02
    rare_classes = [class_names[c] for c, f in enumerate(freqs) if 0 < f < rare_threshold]
    absent_classes = [class_names[c] for c, f in enumerate(freqs) if f == 0.0]
    positive_freqs = [f for f in freqs if f > 0]
    imbalance_ratio = (max(positive_freqs) / min(positive_freqs)) if positive_freqs else None
    background_dominant = is_pixelwise and n_classes == 2 and freqs[0] > 0.9

    return {
        "n_classes": n_classes,
        "class_names": class_names,
        "counts": counts,
        "frequencies": freqs,
        "rare_classes": rare_classes,
        "classes_absent_from_sample": absent_classes,
        "imbalance_ratio": imbalance_ratio,
        "background_dominant": background_dominant,
        "unit": "active-samples" if is_multi_label else ("pixels" if is_pixelwise else "samples"),
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

    `task_type` only distinguishes segmentation (per-pixel mask) from
    everything else here; whether "everything else" is single- or
    multi-label classification is resolved from the dataset's own
    geobench.LabelType via gotrain.dataset_spec, not guessed from task_type
    or the dataset name -- m-bigearthnet's sample.label is already a
    {0,1}^n_classes multi-hot np.ndarray, not an int, so coercing it with
    int() (the pre-existing single-label path) would raise on it.

    Returns (images, labels_or_masks, band_names, task_specs).
    """
    import geobench
    from gotrain.dataset_spec import spec_for

    task_dir = _find_task_dir(dataset)
    task_specs = geobench.load_task_specs(task_dir)
    is_multi_label = spec_for(dataset, task_specs=task_specs).is_multi_label

    ds = task_specs.get_dataset(split="train")
    n = min(len(ds), max_samples)

    images, labels, band_names = [], [], None
    for i in range(n):
        sample = ds[i]
        arr, mask, names = _inspection_arrays(sample, task_type == "segmentation")
        if band_names is None:
            band_names = names
            # Bound by bytes as well as count: a 13-band 256x256 or
            # 373-channel sample is far bigger than the 3-band ones.
            n = min(n, max(1, INSPECTION_MAX_BYTES // max(arr.nbytes, 1)))
        if i >= n:
            break
        images.append(arr)
        if task_type == "segmentation":
            labels.append(mask)
        elif is_multi_label:
            labels.append(np.asarray(sample.label, dtype=np.float32))
        else:
            labels.append(int(sample.label))

    images = np.stack(images, axis=0)
    if task_type == "segmentation" or is_multi_label:
        labels = np.stack(labels, axis=0)
    else:
        labels = np.array(labels, dtype=np.int64)
    return images, labels, band_names, task_specs


# Upper bound on the inspection sample (GeoGuard / DataIQ structural scan).
INSPECTION_MAX_BYTES = 1_500_000_000


def _inspection_arrays(sample, is_segmentation: bool):
    """(image float32 HxWxC, mask or None, per-channel names) for inspection.

    Bands of different native sizes (m-NeonTree: RGB 400x400, Canopy Height
    and the 369-channel Neon hyperspectral band at 40x40) are brought DOWN
    to the smallest common size by integer striding, and the mask with them.
    Upsampling instead (what training does via pack_to_3d(resample=True))
    costs ~40 s and ~239 MB per NeonTree sample, and mismatched image/mask
    sizes would make DataIQ flag every sample as invalid. Same-size datasets
    are unaffected (no striding)."""
    datas, names = [], []
    for band in sample.bands:
        data = band.data if band.data.ndim == 3 else band.data[:, :, None]
        datas.append(data)
        expanded = band.band_info.expand_name()
        names.extend(expanded if len(expanded) == data.shape[2] else [band.band_info.name] * data.shape[2])
    min_h = min(d.shape[0] for d in datas)
    min_w = min(d.shape[1] for d in datas)

    def _shrink(array):
        if array.shape[0] == min_h and array.shape[1] == min_w:
            return array
        step_h, step_w = array.shape[0] // min_h, array.shape[1] // min_w
        return array[: min_h * step_h : step_h, : min_w * step_w : step_w]

    image = np.concatenate([_shrink(d).astype(np.float32) for d in datas], axis=2)
    mask = _shrink(np.asarray(sample.label.data)) if is_segmentation else None
    return image, mask, names
