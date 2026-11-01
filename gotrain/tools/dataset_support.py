"""`dataset -> supported / unsupported + reason` reporting, for all 12
GEO-Bench hackathon datasets (and any others dropped under GEO_BENCH_DIR),
WITHOUT downloading anything.

A dataset is "supported" here if GoTrain's training/evaluation code has a
path for its label_kind (single-label classification, multi-label
classification, or segmentation -- see gotrain/dataset_spec.py) -- not
whether it has actually been trained successfully yet. "Unsupported" means
either it isn't downloaded locally (task_specs.pkl missing) or its
label_type isn't one GoTrain has a training/eval path for at all (e.g. a
future geobench.Detection task).
"""

from gotrain.dataset_spec import check_support

# The 12 GEO-Bench v1.0 datasets named in the hackathon brief, exactly as
# they appear in the authoritative source repo recursix/geo-bench-1.0 (see
# report_all_datasets' docstring for how that was confirmed) -- used only
# as the default list for report_all_datasets(); check_dataset_support
# itself works on any dataset name, not just these 12.
HACKATHON_DATASETS = [
    "m-eurosat",
    "m-so2sat",
    "m-bigearthnet",
    "m-brick-kiln",
    "m-forestnet",
    "m-pv4ger",
    "m-pv4ger-seg",
    "m-nz-cattle",
    "m-chesapeake",
    "m-cashew-plant",
    "m-SA-crop-type",
    "m-NeonTree",
]


def check_dataset_support(dataset: str) -> dict:
    """Report whether `dataset` is supported and exactly why not if not,
    without downloading or reading any pixel data -- only task_specs.pkl
    (a small geobench metadata file) needs to already exist locally.
    """
    return check_support(dataset)


def report_all_datasets(datasets: list = None) -> dict:
    """Report support status for every dataset in `datasets` (default:
    the 12 hackathon datasets). Returns
    {"datasets": [check_dataset_support(d) for d in datasets],
     "n_supported": int, "n_unsupported": int}.

    Safe to call before any dataset is downloaded -- every unresolved
    dataset is reported with reason="not found locally: ...", not an
    exception.
    """
    datasets = datasets if datasets is not None else HACKATHON_DATASETS
    results = [check_dataset_support(d) for d in datasets]
    n_supported = sum(1 for r in results if r["supported"])
    return {
        "datasets": results,
        "n_supported": n_supported,
        "n_unsupported": len(results) - n_supported,
    }
