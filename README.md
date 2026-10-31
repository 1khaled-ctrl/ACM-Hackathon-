# GoTrain — AUB ACM x RASID Hackathon

## What it is

An AI agent (Strands Agents + Claude Sonnet on Amazon Bedrock) that automatically
trains geospatial models on 4 GEO-Bench tasks — classification (`m-eurosat`,
`m-so2sat`) and segmentation (`m-pv4ger-seg`, `m-nz-cattle`) — iterating on
hyperparameters, recording what it tried and why, and submitting its best model.

Bedrock is only reachable from inside SageMaker — local development covers
everything except actually calling the Bedrock model (see `--dry-run` below).

## Setup

```bash
py -3.13 -m venv .venv       # any Python 3.11+ works if 3.13 isn't installed
                              # (requirements.txt pins packages that need >=3.11);
                              # check what's available with `py -0`
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

`torch`/`torchvision` are pinned without a `+cpu`/`+cuXXX` suffix on purpose — those
local version tags only exist on PyTorch's own wheel index, not on PyPI, so a plain
`pip install -r requirements.txt` would fail to resolve them otherwise. The plain PyPI
wheel is CUDA-capable on Linux (SageMaker) and falls back to CPU automatically where
there's no GPU (a laptop). `torch` is pinned as a **minimum** version (`>=2.9`), not
an exact one, specifically so it doesn't fight with whatever build is already present.

**On the SageMaker GPU lab, `torch` is already preinstalled with CUDA — do not
reinstall it.** Do not run a plain `pip install torch` or `pip install -U torch`
there; that would silently replace the preinstalled CUDA build with a CPU-only one
from PyPI. A plain `pip install -r requirements.txt` is safe to run as-is on the lab:
pip sees the installed torch already satisfies `>=2.9` and leaves it alone,
installing only the other packages.

Or run `setup.sh` on a fresh Linux machine (installs deps + downloads the 4 GEO-Bench
tasks used in this project).

## Data

Only 4 GEO-Bench tasks are used (not the full benchmark):

- Classification: `m-eurosat`, `m-so2sat`
- Segmentation: `m-pv4ger-seg`, `m-nz-cattle`

Set `GEO_BENCH_DIR=./data/geobench` before downloading/loading.

## How to run

```bash
# Local, no AWS access needed — proves the tool wiring, never calls Bedrock:
python run_gotrain.py --dataset m-eurosat --dry-run

# On SageMaker, with Bedrock access:
python run_gotrain.py --dataset m-eurosat --max-runs 6 --run-minutes 5

# After the agent submits a model, a human runs the one-time final evaluation:
python run_final_eval.py --dataset m-eurosat

# Dashboard — read-only view over status/ledger/events:
streamlit run dashboard.py

# Tests:
pytest tests/ -v
```

## Dashboard

```bash
streamlit run dashboard.py
```

A single-page, read-only, dark-themed view of what the agent is doing, built
around one goal: understand it within 5 seconds. Shows live status, the
workflow stage (analyze/train/validate/compare/decide), the current
experiment's live per-epoch progress, the best model so far, a validation-score
chart across runs, the agent's last logged decision/rationale, run history,
model lineage, and a live activity feed. Never launches training, changes a
config, or triggers the final test evaluation — it only reads
`logs/status.json`, `logs/ledger.jsonl`, and `logs/events.jsonl`.

On Windows, `launch_dashboard.bat` (not tracked by git) sets up a `.venv` and
runs the above for you — a desktop shortcut can point straight at it.

**Viewing a teammate's training from a different machine:** the dashboard has
no networking built in — it only reads local files. Either (a) run it on the
same machine that's training (e.g. the SageMaker lab) and view it from your
own browser through Jupyter's proxy (`<notebook-url>/proxy/8501/`, with
`--server.address 0.0.0.0`), or (b) mirror that machine's 3 log files onto
yours by any means (shared drive, `scp`, manual copy) — point at a different
local folder than this repo's own `logs/` with the `GOTRAIN_LOGS_DIR`
environment variable if needed (same pattern as `GEO_BENCH_DIR`).

## The safety rail

- The agent trains on the **train** split and tunes/compares on the **validation**
  split only. It has no tool that can see the **test** split — `final_test_eval`
  is deliberately not in its tool list.
- The test split is evaluated exactly once, by a human running
  `run_final_eval.py`, after the agent has submitted its final model. A lock
  file (`logs/test_lock_<dataset>.json`) refuses a second evaluation per dataset.
- Hard limits (max runs, per-run time budget, max tool calls) are enforced in
  code via a hook, not left to the prompt — a run rejected for an unsupported
  config value doesn't cost the agent one of its max runs.
- Every tool call is logged with its arguments, result, and duration.

## Logs, models, and runs

- `logs/events.jsonl` — every tool call the agent made (name, arguments, truncated
  result, duration).
- `logs/ledger.jsonl` — one record per training run: config, validation metric,
  lineage (`parent_id`/`config_diff`), decision, and `version_id`.
- `logs/status.json` — the agent's current stage/action, for the dashboard or a
  quick check of what's happening right now.
- `runs/<dataset>/<run_id>/` — scratch output from one training process
  (`config.json`, `metrics.json`, `train.log`, checkpoint). Not tracked by git.
- `models/<dataset>/<version>/` — the durable model registry: a run's config,
  metrics, and checkpoint promoted to a stable `v1`, `v2`, ... location. Not
  tracked by git (large/regenerable); `upload_to_s3`/`download_from_s3` sync a
  version to S3 when `GOTRAIN_S3_BUCKET` is set.
- `examples/` — one real sample log set, committed, so the shape of a run's
  logs/metrics is visible without having to run anything first.

See `gotrain/templates/INTERFACE.md` for the exact CLI/stdout contract every
training template follows.
