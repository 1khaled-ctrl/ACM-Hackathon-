# GoTrain — AUB ACM x RASID Hackathon

## Goal
Build an AI agent (Strands Agents + Claude Sonnet on Amazon Bedrock) that automatically
trains geospatial models on GEO-Bench datasets.

Bedrock is only reachable from inside SageMaker — local development covers everything
except actually calling the Bedrock model.

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
there's no GPU (a laptop).

Or run `setup.sh` on a fresh Linux machine (installs deps + downloads the 4 GEO-Bench
tasks used in this project).

## Data

Only 4 GEO-Bench tasks are used (not the full benchmark):

- Classification: `m-eurosat`, `m-so2sat`
- Segmentation: `m-pv4ger-seg`, `m-nz-cattle`

Set `GEO_BENCH_DIR=./data/geobench` before downloading/loading.

## Rules

- Train on the **train** split.
- Tune, score, and compare on the **validation** split only.
- The **test** split is touched exactly once, at the end, with the final model.

## Running the agent

Bedrock (the agent's model) is only reachable from inside SageMaker. On a laptop, use
`--dry-run` — it prints the system prompt and tool list and proves the tool wiring
locally without ever calling Bedrock.

```bash
# Local, no AWS access needed:
python run_gotrain.py --dataset m-eurosat --dry-run

# On SageMaker, with Bedrock access:
python run_gotrain.py --dataset m-eurosat --max-runs 6 --run-minutes 5
```

The agent can only launch training runs and read/compare validation metrics. It has
no tool that can see the test split. When it's done, it calls `submit_final_model`
to record its chosen run — this does not evaluate anything.

To run the one-time final test-split evaluation yourself, after the agent has
submitted a model:

```bash
python run_final_eval.py --dataset m-eurosat
```

This asks for interactive confirmation, then evaluates the submitted run on the test
split exactly once; a second attempt for the same dataset is refused.

## Dashboard

```bash
streamlit run dashboard.py
```

Read-only view over `logs/status.json`, `logs/ledger.jsonl` (run history, lineage,
decisions), and `logs/events.jsonl` (every tool call the agent made).

## Tests

```bash
pytest tests/ -v
```
