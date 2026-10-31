"""Fake training template for exercising the agent/tooling flow without
actually training a model. Takes the same CLI contract as the real
templates (--dataset, --config as a JSON string) and prints one JSON line
with a val_metric, so launch_training() can be tested end-to-end quickly.

Never touches the test split.
"""

import argparse
import hashlib
import json
import random


def fake_val_metric(dataset: str, config: dict) -> float:
    """Deterministic pseudo-metric derived from dataset + config, so repeated
    calls with the same config are reproducible but different configs vary.
    """
    key = f"{dataset}:{json.dumps(config, sort_keys=True)}"
    seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    return round(rng.uniform(0.5, 0.95), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True, help="JSON-encoded config dict")
    args = parser.parse_args()

    config = json.loads(args.config)
    val_metric = fake_val_metric(args.dataset, config)

    result = {
        "val_metric": val_metric,
        "gpu_minutes": 0.0,
        "rationale": "fake_train.py smoke-test run, no real training performed",
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
