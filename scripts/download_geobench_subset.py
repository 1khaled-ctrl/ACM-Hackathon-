"""
Download only the 4 GEO-Bench tasks this project uses, instead of the full
~65 GB benchmark.

geobench's own `geobench-download` CLI (geobench/geobench_download.py) always
fetches every file in the `recursix/geo-bench-1.0` Hugging Face dataset repo,
with no task filter. This script replicates its download + extract logic
(hf_hub_download -> unzip in place -> delete zip) but calls hf_hub_download
only for the files we want, via huggingface_hub (already a dependency of
geobench).

Confirmed exact files/sizes on https://huggingface.co/datasets/recursix/geo-bench-1.0:
  classification_v1.0/m-eurosat.zip      276 MB
  classification_v1.0/m-so2sat.zip       964 MB
  segmentation_v1.0/m-pv4ger-seg.zip     300 MB
  segmentation_v1.0/m-nz-cattle.zip      327 MB
Each top-level folder also has a small normalizer.json needed by geobench's
task loader.
"""

import os
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "recursix/geo-bench-1.0"
REPO_TYPE = "dataset"

# All 12 hackathon datasets -> their archive in recursix/geo-bench-1.0
# (zip sizes as listed on the Hub: bigearthnet 6.6 GB, forestnet 1.9,
# brick-kiln 1.6, SA-crop-type 1.4, cashew-plant 1.3, chesapeake 1.3,
# pv4ger 1.0, so2sat 1.0, NeonTree 0.6, nz-cattle 0.3, pv4ger-seg 0.3,
# eurosat 0.3).
BENCHMARK_OF = {
    "m-eurosat": "classification_v1.0", "m-so2sat": "classification_v1.0",
    "m-bigearthnet": "classification_v1.0", "m-brick-kiln": "classification_v1.0",
    "m-forestnet": "classification_v1.0", "m-pv4ger": "classification_v1.0",
    "m-pv4ger-seg": "segmentation_v1.0", "m-nz-cattle": "segmentation_v1.0",
    "m-chesapeake": "segmentation_v1.0", "m-cashew-plant": "segmentation_v1.0",
    "m-SA-crop-type": "segmentation_v1.0", "m-NeonTree": "segmentation_v1.0",
}
DEFAULT_TASKS = ["m-eurosat", "m-so2sat", "m-pv4ger-seg", "m-nz-cattle"]


def files_for(tasks):
    benchmarks = sorted({BENCHMARK_OF[t] for t in tasks})
    return [f"{BENCHMARK_OF[t]}/{t}.zip" for t in tasks] + [f"{b}/normalizer.json" for b in benchmarks]


def main():
    """python scripts/download_geobench_subset.py               # the original 4
    python scripts/download_geobench_subset.py all           # all 12 (~17 GB of zips)
    python scripts/download_geobench_subset.py m-NeonTree m-chesapeake"""
    import sys

    args = sys.argv[1:]
    tasks = list(BENCHMARK_OF) if args == ["all"] else (args or DEFAULT_TASKS)
    unknown = [t for t in tasks if t not in BENCHMARK_OF]
    if unknown:
        raise SystemExit(f"Unknown dataset(s) {unknown}; choose from {list(BENCHMARK_OF)} or 'all'.")

    geo_bench_dir = Path(os.environ.get("GEO_BENCH_DIR", "./data/geobench")).resolve()
    geo_bench_dir.mkdir(parents=True, exist_ok=True)
    print(f"GEO_BENCH_DIR = {geo_bench_dir}")

    for rel_path in files_for(tasks):
        print(f"\nDownloading {rel_path} ...")
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=rel_path,
            local_dir=geo_bench_dir,
        )
        downloaded_path = Path(downloaded_path)

        if downloaded_path.suffix == ".zip":
            extract_to = downloaded_path.parent
            print(f"Extracting {downloaded_path.name} -> {extract_to}")
            with zipfile.ZipFile(downloaded_path, "r") as zf:
                zf.extractall(extract_to)
            downloaded_path.unlink()
            print(f"Removed {downloaded_path.name} after extraction")

    print("\nDone. Downloaded tasks:")
    for task in tasks:
        print(" -", task)


if __name__ == "__main__":
    main()
