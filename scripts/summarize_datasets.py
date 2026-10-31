"""Summarize the 4 downloaded GEO-Bench tasks: train/val sample counts,
band names, image shape, label info. Writes data/dataset_summary.md.

The test split is never opened here — its count is reported via
geobench's internal partition JSON (read as plain metadata, not through
get_dataset()) so even the sample count doesn't require instantiating a
test dataset object backed by actual sample loading.
"""

import json
import os
from pathlib import Path

import geobench

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

TASKS = [
    ("classification_v1.0", "m-eurosat"),
    ("classification_v1.0", "m-so2sat"),
    ("segmentation_v1.0", "m-pv4ger-seg"),
    ("segmentation_v1.0", "m-nz-cattle"),
]


def test_split_count(task_dir: Path) -> "int | str":
    """Read the test split's sample count from the partition file without
    instantiating a dataset object or touching any sample content.
    """
    partition_path = task_dir / "default_partition.json"
    if not partition_path.exists():
        return "unknown"
    with open(partition_path, "r", encoding="utf-8") as f:
        partition = json.load(f)
    test_key = next((k for k in partition if k.lower().startswith("test")), None)
    if test_key is None:
        return "unknown"
    return len(partition[test_key])


def summarize_task(benchmark_dir: str, dataset: str) -> dict:
    task_dir = Path(os.environ["GEO_BENCH_DIR"]) / benchmark_dir / dataset
    task_specs = geobench.load_task_specs(task_dir)

    train_ds = task_specs.get_dataset(split="train")
    valid_ds = task_specs.get_dataset(split="valid")

    sample = train_ds[0]
    band_names = [b.name for b in task_specs.bands_info]
    image_shape = list(task_specs.patch_size) + [len(band_names)]

    label_type = task_specs.label_type
    label_info: dict = {"kind": type(label_type).__name__}
    if hasattr(label_type, "class_names"):
        label_info["classes"] = list(label_type.class_names)
        label_info["n_classes"] = getattr(label_type, "n_classes", len(label_info["classes"]))
    elif hasattr(label_type, "n_classes"):
        label_info["n_classes"] = label_type.n_classes

    return {
        "dataset": dataset,
        "benchmark": benchmark_dir,
        "n_train": len(train_ds),
        "n_validation": len(valid_ds),
        "n_test": test_split_count(task_dir),
        "band_names": band_names,
        "n_bands": len(band_names),
        "image_shape_hw_c": image_shape,
        "label_info": label_info,
    }


def render_markdown(summaries: list) -> str:
    lines = ["# GEO-Bench Dataset Summary", ""]
    lines.append(
        "Counts only — the test split's contents are never loaded here, "
        "only its sample count read from the partition file metadata."
    )
    lines.append("")

    for s in summaries:
        lines.append(f"## {s['dataset']} ({s['benchmark']})")
        lines.append("")
        lines.append(f"- Train samples: {s['n_train']}")
        lines.append(f"- Validation samples: {s['n_validation']}")
        lines.append(f"- Test samples: {s['n_test']} (count only, split not opened)")
        lines.append(f"- Bands ({s['n_bands']}): {', '.join(s['band_names'])}")
        lines.append(f"- Image shape (H, W, C): {tuple(s['image_shape_hw_c'])}")

        label_info = s["label_info"]
        if "classes" in label_info:
            lines.append(
                f"- Label: {label_info['kind']}, {label_info['n_classes']} classes: "
                f"{', '.join(label_info['classes'])}"
            )
        elif "n_classes" in label_info:
            lines.append(f"- Label: {label_info['kind']}, {label_info['n_classes']} classes")
        else:
            lines.append(f"- Label: {label_info['kind']}")
        lines.append("")

    return "\n".join(lines)


def main():
    summaries = [summarize_task(b, d) for b, d in TASKS]
    markdown = render_markdown(summaries)

    out_path = REPO_ROOT / "data" / "dataset_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
