#!/usr/bin/env bash
# One-command setup for a fresh Linux machine: install deps + download the
# 4 GEO-Bench tasks used in this project.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Creating virtual environment (.venv) ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip setuptools wheel

echo "=== Installing requirements ==="
pip install -r requirements.txt

echo "=== Downloading GEO-Bench subset (m-eurosat, m-so2sat, m-pv4ger-seg, m-nz-cattle) ==="
export GEO_BENCH_DIR="$(pwd)/data/geobench"
mkdir -p "$GEO_BENCH_DIR"
python scripts/download_geobench_subset.py

echo "=== Done. Activate with: source .venv/bin/activate ==="
