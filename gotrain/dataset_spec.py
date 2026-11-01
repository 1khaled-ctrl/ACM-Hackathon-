"""Central dataset specification: the one place that classifies a GEO-Bench
dataset's label kind and exposes what every other subsystem needs to handle
it correctly.

This is NOT a hardcoded per-dataset-name table. list_datasets()/get_dataset_info()
in gotrain/tools/datasets.py already discover datasets dynamically from
whatever's on disk under GEO_BENCH_DIR; this module classifies *any* of them
by inspecting their actual geobench.LabelType instance, so a 13th or 20th
GEO-Bench-compatible dataset dropped in later is classified correctly with
no code change here.

Three label kinds exist in geobench (see .venv/.../geobench/task.py):
  - Classification            -> one int label per sample, in [0, n_classes).
                                   Trained with CrossEntropyLoss / softmax / argmax.
  - MultiLabelClassification  -> a {0,1}^n_classes multi-hot vector per sample
                                   (e.g. m-bigearthnet). Trained with
                                   BCEWithLogitsLoss / sigmoid / per-class threshold.
  - SegmentationClasses       -> a per-pixel class id mask, same int-label rules
                                   as Classification but pixelwise.

Every chokepoint that assumed "classification label is a scalar int" --
training templates, DataIQ, GeoGuard, StressLab, final_eval -- reads
LABEL_KIND_* via spec_for() instead of re-deriving this itself, so there is
exactly one place to extend if geobench ever adds a fourth label kind.
"""

from dataclasses import dataclass
from typing import Optional

LABEL_KIND_SINGLE = "single_label"
LABEL_KIND_MULTI = "multi_label"
LABEL_KIND_SEGMENTATION = "segmentation"

# geobench class name -> our label kind. Keyed by class name (not dataset
# name) so this stays correct for any dataset, present or future.
_LABEL_TYPE_CLASS_TO_KIND = {
    "Classification": LABEL_KIND_SINGLE,
    "MultiLabelClassification": LABEL_KIND_MULTI,
    "SegmentationClasses": LABEL_KIND_SEGMENTATION,
}


class UnsupportedLabelTypeError(ValueError):
    """Raised when a dataset's geobench label_type isn't one of the three
    kinds this project knows how to train/evaluate (see module docstring).
    Distinct from a missing dataset (ValueError from _find_task_dir) so
    callers can tell "not downloaded" apart from "downloaded but we don't
    know how to train this kind of task yet".
    """


@dataclass(frozen=True)
class DatasetSpec:
    """Everything training/DataIQ/GeoGuard/StressLab need to know about one
    dataset's task shape, independent of which dataset it is.
    """

    dataset: str
    benchmark: str  # "classification_v1.0" or "segmentation_v1.0"
    label_kind: str  # one of LABEL_KIND_*
    n_classes: int
    class_names: Optional[list]
    n_bands: int
    band_names: list
    # Real model input channels with every band selected. Differs from
    # n_bands when one named band holds many channels (m-NeonTree's "Neon"
    # band is 369 hyperspectral channels: 5 band names, 373 channels).
    n_channels: int = 0

    @property
    def is_multi_label(self) -> bool:
        return self.label_kind == LABEL_KIND_MULTI

    @property
    def is_segmentation(self) -> bool:
        return self.label_kind == LABEL_KIND_SEGMENTATION

    @property
    def task_type(self) -> str:
        """The coarse task_type string the rest of the codebase (config
        validation, launch_training, run directories) already keys on.
        Both single-label and multi-label classification map to the same
        "classification" task_type and training template -- they differ in
        label_kind, not in which template/config schema applies.
        """
        return "segmentation" if self.is_segmentation else "classification"


def _class_names_of(label_type) -> Optional[list]:
    """geobench's three label types don't agree on the attribute name:
    Classification/SegmentationClasses expose `class_names` (plural), but
    MultiLabelClassification exposes `class_name` (singular) -- see
    .venv/.../geobench/task.py. Checking both here means
    get_dataset_info()/DatasetSpec never silently report "no class names"
    for a multi-label dataset just because of that naming inconsistency
    upstream.
    """
    names = getattr(label_type, "class_names", None)
    if names is None:
        names = getattr(label_type, "class_name", None)
    return list(names) if names is not None else None


def classify_label_type(label_type) -> str:
    """Map a geobench LabelType instance to one of LABEL_KIND_*.

    Raises UnsupportedLabelTypeError for any label type this project hasn't
    been taught to train/evaluate (e.g. geobench.Detection,
    geobench.CloudProbability) -- callers should surface this as
    "unsupported + reason", not try to train on it.
    """
    kind = _LABEL_TYPE_CLASS_TO_KIND.get(type(label_type).__name__)
    if kind is None:
        raise UnsupportedLabelTypeError(
            f"label_type {type(label_type).__name__!r} is not one of "
            f"{sorted(_LABEL_TYPE_CLASS_TO_KIND)} -- GoTrain has no training/"
            "evaluation path for this task kind yet."
        )
    return kind


def spec_for(dataset: str, task_specs=None) -> DatasetSpec:
    """Build the DatasetSpec for `dataset`. Pass an already-loaded
    `task_specs` (geobench.load_task_specs(...)) to avoid loading it twice
    when a caller already has it; otherwise this loads it itself the same
    way gotrain/tools/datasets.py does.

    Raises ValueError (dataset not found under GEO_BENCH_DIR) or
    UnsupportedLabelTypeError (found, but not a label kind we support).
    """
    if task_specs is None:
        from gotrain.tools.datasets import _find_task_dir
        import geobench

        task_dir = _find_task_dir(dataset)
        task_specs = geobench.load_task_specs(task_dir)

    label_type = task_specs.label_type
    label_kind = classify_label_type(label_type)
    # Every SegmentationClasses-labeled task lives under segmentation_v1.0,
    # every Classification/MultiLabelClassification one under
    # classification_v1.0 (confirmed against the actual recursix/geo-bench-1.0
    # layout) -- derived from label_kind rather than the task_dir's parent so
    # this works the same whether or not a caller already had task_specs.
    benchmark = "segmentation_v1.0" if label_kind == LABEL_KIND_SEGMENTATION else "classification_v1.0"
    n_classes = label_type.n_classes
    class_names = _class_names_of(label_type)
    band_names = [b.name for b in task_specs.bands_info]
    n_channels = sum(
        len(b.expand_name()) if hasattr(b, "expand_name") else 1 for b in task_specs.bands_info
    )

    return DatasetSpec(
        dataset=dataset,
        benchmark=benchmark,
        label_kind=label_kind,
        n_classes=n_classes,
        class_names=class_names,
        n_bands=len(band_names),
        band_names=band_names,
        n_channels=n_channels,
    )


def estimate_training_memory(dataset: str, task_type: str, config: dict) -> Optional[dict]:
    """Bytes the training template will hold for its eagerly loaded train +
    validation arrays under `config` (bands, max_train_samples,
    max_val_samples). Reads ONE train sample's raw band shapes -- no
    resampling -- so it stays cheap even for m-NeonTree, where packing all
    bands costs ~40 s per sample. Returns None when it can't estimate (e.g.
    data missing); callers then skip the check rather than block on it.
    """
    import numpy as np

    try:
        from gotrain.tools.datasets import _find_task_dir
        import geobench

        task_specs = geobench.load_task_specs(_find_task_dir(dataset))
        band_names = config.get("bands") or [b.name for b in task_specs.bands_info]
        train = task_specs.get_dataset(split="train")
        val = task_specs.get_dataset(split="valid")
        sample = train[0]
    except Exception:
        return None

    wanted = set(band_names)
    selected = [
        b for b in sample.bands
        if ({b.band_info.name} | set(getattr(b.band_info, "alt_names", ()) or ())) & wanted
    ]
    if not selected:
        return None

    height = max(b.data.shape[0] for b in selected)
    width = max(b.data.shape[1] for b in selected)
    channels = sum(1 if b.data.ndim == 2 else b.data.shape[2] for b in selected)
    dtype = np.result_type(*[b.data.dtype for b in selected])
    itemsize = 4 if dtype == np.float64 else dtype.itemsize  # pack_sample narrows float64
    per_sample = height * width * channels * itemsize
    if task_type == "segmentation":
        per_sample += height * width * np.asarray(sample.label.data).dtype.itemsize

    def _bounded(n, cap):
        return n if cap is None else min(n, cap)

    n_train = _bounded(len(train), config.get("max_train_samples"))
    n_val = _bounded(len(val), config.get("max_val_samples"))
    return {
        "per_sample_bytes": per_sample,
        "n_train": n_train,
        "n_val": n_val,
        "total_bytes": per_sample * (n_train + n_val),
        "channels": channels,
        "height": height,
        "width": width,
        "band_channels": {b.band_info.name: (1 if b.data.ndim == 2 else b.data.shape[2]) for b in selected},
        "band_shapes": {b.band_info.name: list(b.data.shape[:2]) for b in selected},
    }


def check_support(dataset: str) -> dict:
    """Report whether `dataset` is supported, and exactly why not if it
    isn't, WITHOUT downloading anything -- only reads task_specs.pkl, which
    must already exist locally under GEO_BENCH_DIR (geobench's own metadata
    file, a few KB, bundled with the dataset's zip).

    Returns {"dataset", "supported": bool, "reason": str, "label_kind": str|None,
    "n_classes": int|None, "n_bands": int|None}. Used by the
    `dataset -> supported/unsupported + reason` CLI tool
    (gotrain/tools/dataset_support.py) to check all 12 datasets before any
    download.
    """
    try:
        spec = spec_for(dataset)
    except UnsupportedLabelTypeError as e:
        return {
            "dataset": dataset, "supported": False, "reason": str(e),
            "label_kind": None, "n_classes": None, "n_bands": None,
        }
    except ValueError as e:
        return {
            "dataset": dataset, "supported": False,
            "reason": f"not found locally: {e}",
            "label_kind": None, "n_classes": None, "n_bands": None,
        }
    except Exception as e:
        return {
            "dataset": dataset, "supported": False,
            "reason": f"unexpected error while reading task_specs: {e}",
            "label_kind": None, "n_classes": None, "n_bands": None,
        }

    result = {
        "dataset": dataset,
        "supported": True,
        "reason": "ok",
        "label_kind": spec.label_kind,
        "n_classes": spec.n_classes,
        "n_bands": spec.n_bands,
        "n_channels": spec.n_channels,
    }
    if spec.n_channels != spec.n_bands:
        result["note"] = (
            f"{spec.n_bands} band names expand to {spec.n_channels} input channels; check "
            "get_dataset_info/launch_training's memory estimate before training on all bands."
        )
    return result
