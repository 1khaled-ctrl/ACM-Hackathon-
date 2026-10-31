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

# Dashboard — view + engineer control (pause/resume/manual-takeover/approvals/rollback):
streamlit run dashboard.py

# Tests:
pytest tests/ -v
```

## Dashboard

```bash
streamlit run dashboard.py
```

A single-page, dark-themed view of what the agent is doing, built around one
goal: understand it within 5 seconds. Shows live status, the workflow stage
(analyze/train/validate/compare/decide), the current experiment's live
per-epoch progress, the best model so far, a validation-score chart across
runs, the agent's last logged decision/rationale (including its structured
hypothesis and the code-computed `hypothesis_outcome`), GeoGuard's findings,
run history, model lineage, and a live activity feed.

It also has real write access to the backend, via the **Control** tab and
the always-visible Engineer Control card (see "Engineer control plane"
below): Pause / Resume / Take Manual Control, approve/reject pending
config-approval requests, and roll back to a previous model version. It
still never touches the test split and never calls `final_test_eval` — that
stays exclusively in `run_final_eval.py`, run by a human.

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

## GeoGuard — data-aware preprocessing diagnostics

Before its first `launch_training` call, the agent calls `run_geoguard`,
which inspects a bounded sample (default 300 images) of the dataset's
**train split only** and reports:

- per-band pixel statistics (min/max/mean/std, possible saturation/clipping,
  near-constant bands), and NaN/Inf counts;
- whether the dataset mixes sensor modalities (e.g. m-so2sat's 8 SAR + 10
  optical bands), detected by band name;
- class/pixel frequency, rare classes, and (for binary segmentation)
  background dominance;
- a **recommended config** — drawn only from the same values
  `launch_training` already accepts (`gotrain/config_schema.py`) — with a
  one-line rationale per recommendation, so the reasoning is auditable, not
  just the output.

GeoGuard never fits or returns anything applied to validation/test data,
and never executes model-authored code: `gotrain/geoguard/decision.py` is a
plain deterministic function, and every config it proposes is re-validated
against `config_schema.validate_config` before being returned. The report
is written to `logs/geoguard/<dataset>.json` (see `gotrain/geoguard/`).

Post-preprocessing QA checks (`gotrain/geoguard/qa.py`) validate an
already-normalized batch — shape, channel count, NaN/Inf, label/mask
alignment and range — independent of GeoGuard's dataset-level diagnostics.

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

## Engineer control plane

The dashboard is no longer read-only: it's a real control surface backed by
an authoritative state machine (`gotrain/control.py`, `logs/control.json`),
checked both by the agent's hook (`gotrain/agent.py`) and, redundantly,
inside `launch_training` itself (`gotrain/tools/training.py`) — so it holds
even for a direct/script-level call that bypasses the agent entirely.

**States:** `RUNNING_AUTONOMOUSLY` (default) → `PAUSE_REQUESTED` →
`PAUSED_FOR_ENGINEER` → `RESUME_REQUESTED` → back to
`RUNNING_AUTONOMOUSLY`; or `MANUAL_CONTROL` (engineer takes over directly);
or `ROLLBACK_REQUESTED` → `ROLLBACK_VERIFYING` → `PAUSED_FOR_ENGINEER`
(success) / `FAILED_SAFE` (verification failed); or terminal `COMPLETED`
once a final model is submitted. Only the transitions listed in
`control.VALID_TRANSITIONS` are allowed — an illegal one raises rather than
silently no-op-ing — and every transition is logged to `events.jsonl`.

**Pause/manual control**: clicking Pause or Take Manual Control in the
dashboard sets `control.json`; the agent's hook then cancels its *very next*
tool call (any tool, not just `launch_training`) with a clear message, and
that cancellation doesn't cost run or tool-call budget. If training is
actively running, the training templates (`gotrain/templates/train_*.py`)
also check `control.should_stop_training()` once per epoch, so a pause
takes effect within one epoch rather than waiting for the full run's time
budget — the run still completes normally otherwise, just tagged
`"interrupted": true` with a reason in its metrics. `run_gotrain.py`'s main
loop then blocks on `control.wait_for_resume()` and, once an engineer
resumes, continues the **same** agent conversation rather than starting
over. (This specific multi-turn resume path needs a live SageMaker/Bedrock
run to fully confirm — see the module's docstring for the exact caveat;
the state machine and polling logic it depends on are unit-tested.)

**Approval gate** (`gotrain/approval.py`): `backbone` (classification) and
`architecture`/`encoder` (segmentation) are gated — the training templates'
own defaults (`resnet18`; `unet` + `resnet34`) always run immediately, but
any other value requires engineer approval the first time it's proposed.
Gate state is keyed by `(dataset, field, value)`, never by call identity —
so a rejected value cannot be let through by resubmitting it under a
different config or rationale (see
`tests/test_approval.py::test_rejected_value_cannot_be_bypassed_by_resubmission`
for the exact "ResNet-50 rejected for GPU limits" scenario from the design
brief). Decisions are only ever made from the dashboard's **Control** tab —
there is no agent tool to approve/reject/pause/resume/rollback anything;
giving the agent that tool would defeat the entire point.

**Verified rollback** (`gotrain/rollback.py`): rolling back to a model
version checks (in order) that no training run is currently in progress,
that the target checkpoint file is non-empty and its config/metrics exist,
and — not yet exercised against real weights in this environment, since it
needs torch — that the checkpoint actually loads into the architecture its
own `config.json` describes and produces correctly-shaped output on a
synthetic input. Only if every check passes does it atomically rewrite
`models/<dataset>/active_version.json`; a failed check never touches that
pointer, and a successful rollback always leaves the system
`PAUSED_FOR_ENGINEER` (never auto-resumes).

## Logs, models, and runs

- `logs/events.jsonl` — every tool call the agent made (name, arguments, truncated
  result, duration).
- `logs/ledger.jsonl` — one record per training run: config, validation metric,
  lineage (`parent_id`/`config_diff`), decision, and `version_id`.
- `logs/status.json` — the agent's current stage/action, for the dashboard or a
  quick check of what's happening right now.
- `logs/geoguard/<dataset>.json` — the GeoGuard diagnostic report for a
  dataset (see above), written by `run_geoguard`.
- `logs/control.json` — the authoritative engineer-control state (see
  above), written by `gotrain/control.py`.
- `logs/approvals_state.json` / `logs/approvals.jsonl` — current status and
  full audit trail of every gated-config approval request/decision (see
  above), written by `gotrain/approval.py`.
- `models/<dataset>/active_version.json` — the dataset's currently active
  model version after a verified rollback, written by `gotrain/rollback.py`.
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
