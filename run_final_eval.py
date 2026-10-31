#!/usr/bin/env python
"""Human-run, one-time final evaluation on the TEST split.

    python run_final_eval.py --dataset m-eurosat

Reads the run the agent submitted via submit_final_model (the ledger
record with decision="kept" and submitted_final=True for this dataset),
then calls gotrain.tools.final_eval.final_test_eval exactly once. A
second invocation for the same dataset is refused by the lock file in
final_eval.py — this script does not try to work around that.

This script is never called by the agent. Run it yourself, once, when
you are satisfied with the agent's choice.
"""

import argparse
import json
import os
import sys

from gotrain import ledger
from gotrain.tools.final_eval import TestAlreadyEvaluatedError, final_test_eval


def find_submitted_run(dataset: str) -> dict:
    candidates = [
        r for r in ledger.get_history(dataset)
        if r.get("submitted_final") is True
    ]
    if not candidates:
        raise SystemExit(
            f"No run has been submitted as final for {dataset!r} yet. "
            "Run the agent (run_gotrain.py) and have it call submit_final_model first."
        )
    # last submission wins if there were ever more than one (shouldn't normally happen)
    return sorted(candidates, key=lambda r: r["timestamp"])[-1]


def build_eval_fn(run: dict):
    """Build the no-arg callable final_test_eval() will invoke to produce
    test-split metrics for the submitted run.

    Only the "fake" task_type (used for flow-testing) is wired up today,
    since gotrain/templates/train_classification.py and
    train_segmentation.py are still empty templates with no checkpoint-
    saving logic yet. Fill those in, then extend this function to load
    the saved checkpoint for `run["run_id"]` and evaluate it on the real
    test split before using this script for a real submission.
    """
    task_type = run.get("task_type")
    if task_type == "fake":
        from gotrain.templates.fake_train import fake_val_metric

        def eval_fn():
            return {"accuracy": fake_val_metric(run["dataset"], run["config"])}

        return eval_fn

    raise NotImplementedError(
        f"No test-eval path wired up yet for task_type={task_type!r}. "
        "Real classification/segmentation training templates don't save "
        "checkpoints yet — implement that, then extend build_eval_fn() here "
        "to load the checkpoint for this run_id and evaluate it on the test split."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-time final test-split evaluation.")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    os.environ.setdefault("GEO_BENCH_DIR", os.path.join(os.getcwd(), "data", "geobench"))

    run = find_submitted_run(args.dataset)
    print(f"Submitted final run for {args.dataset!r}: {run['run_id']}")
    print(f"  validation metric: {run['val_metric']}")
    print(f"  config: {json.dumps(run['config'])}")

    confirm = input(
        f"\nThis will evaluate run {run['run_id']} on the TEST split for "
        f"{args.dataset!r}, exactly once, permanently. Type 'yes' to continue: "
    )
    if confirm.strip().lower() != "yes":
        print("Aborted. No evaluation was run.")
        sys.exit(1)

    eval_fn = build_eval_fn(run)
    try:
        record = final_test_eval(args.dataset, run["run_id"], eval_fn)
    except TestAlreadyEvaluatedError as e:
        print(f"Refused: {e}")
        sys.exit(1)

    print("\nTEST SPLIT RESULT (one-time):")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
