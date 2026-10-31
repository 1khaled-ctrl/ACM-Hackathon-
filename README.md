# GoTrain — AUB ACM x RASID Hackathon

## Goal
Build an AI agent (Strands Agents + Claude Sonnet on Amazon Bedrock) that automatically
trains geospatial models on GEO-Bench datasets.

Bedrock is only reachable from inside SageMaker — local development covers everything
except actually calling the Bedrock model.

## Setup

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

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
