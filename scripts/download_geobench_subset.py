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

FILES = [
    "classification_v1.0/m-eurosat.zip",
    "classification_v1.0/m-so2sat.zip",
    "classification_v1.0/normalizer.json",
    "segmentation_v1.0/m-pv4ger-seg.zip",
    "segmentation_v1.0/m-nz-cattle.zip",
    "segmentation_v1.0/normalizer.json",
]


def main():
    geo_bench_dir = Path(os.environ.get("GEO_BENCH_DIR", "./data/geobench")).resolve()
    geo_bench_dir.mkdir(parents=True, exist_ok=True)
    print(f"GEO_BENCH_DIR = {geo_bench_dir}")

    for rel_path in FILES:
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
    for task in ["m-eurosat", "m-so2sat", "m-pv4ger-seg", "m-nz-cattle"]:
        print(" -", task)


if __name__ == "__main__":
    main()
