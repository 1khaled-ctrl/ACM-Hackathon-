"""Dataset inspection tools: list_datasets, get_dataset_info.

These return metadata only (sample counts, bands, classes, shapes) — never
raw pixel data or labels. GEO-Bench's internal split name for the validation
split is "valid"; this module exposes it to callers as "validation" to match
the project-wide split vocabulary.
"""

import os
from pathlib import Path
from typing import Optional

import geobench

GEOBENCH_SPLIT_MAP = {"train": "train", "validation": "valid"}

BENCHMARK_DIRS = ["classification_v1.0", "segmentation_v1.0"]


def _geo_bench_dir() -> Path:
    return Path(os.environ.get("GEO_BENCH_DIR", geobench.GEO_BENCH_DIR))


def _find_task_dir(dataset: str) -> Path:
    for benchmark_dir in BENCHMARK_DIRS:
        candidate = _geo_bench_dir() / benchmark_dir / dataset
        if candidate.is_dir() and (candidate / "task_specs.pkl").exists():
            return candidate
    raise ValueError(f"Dataset {dataset!r} not found under {_geo_bench_dir()}")


def list_datasets() -> list:
    """List the GEO-Bench task names available locally under GEO_BENCH_DIR."""
    found = []
    for benchmark_dir in BENCHMARK_DIRS:
        base = _geo_bench_dir() / benchmark_dir
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and (entry / "task_specs.pkl").exists():
                found.append(entry.name)
    return found


def get_dataset_info(dataset: str) -> dict:
    """Return metadata about a dataset: split sizes, bands, label info, shape.

    No pixel data or labels are returned, only counts/names/shapes. The test
    split is never opened here — its size is reported as "locked" rather
    than loading it, per the project rule that the test split is touched
    exactly once, at the final evaluation.
    """
    task_dir = _find_task_dir(dataset)
    task_specs = geobench.load_task_specs(task_dir)

    split_sizes: dict = {"test": "locked"}
    for public_split, geobench_split in GEOBENCH_SPLIT_MAP.items():
        try:
            split_sizes[public_split] = len(task_specs.get_dataset(split=geobench_split))
        except Exception:
            split_sizes[public_split] = None

    from gotrain.dataset_spec import classify_label_type, UnsupportedLabelTypeError

    label_type = task_specs.label_type
    label_info: dict = {"kind": type(label_type).__name__}
    if hasattr(label_type, "n_classes"):
        label_info["n_classes"] = label_type.n_classes
    # _class_names_of covers the class_names/class_name attribute-name
    # inconsistency between geobench's label types (see
    # gotrain/dataset_spec.py's docstring) -- using it directly here (not
    # duplicating the hasattr check) means a multi-label dataset's classes
    # are reported here too, not silently empty.
    from gotrain.dataset_spec import _class_names_of

    class_names = _class_names_of(label_type)
    if class_names is not None:
        label_info["classes"] = class_names
    try:
        label_info["label_kind"] = classify_label_type(label_type)
    except UnsupportedLabelTypeError:
        label_info["label_kind"] = None

    return {
        "dataset": dataset,
        "benchmark": task_dir.parent.name,
        "split_sizes": split_sizes,
        "band_names": [b.name for b in task_specs.bands_info],
        "n_bands": len(task_specs.bands_info),
        # Model input channels per band name, and in total with every band.
        # One name can be many channels: m-NeonTree's "Neon" band is 369
        # hyperspectral channels, so its 5 band names are 373 channels.
        "band_channels": {b.name: len(b.expand_name()) for b in task_specs.bands_info},
        "n_channels": sum(len(b.expand_name()) for b in task_specs.bands_info),
        "patch_size": list(task_specs.patch_size),
        "n_time_steps": task_specs.n_time_steps,
        "label_info": label_info,
    }
