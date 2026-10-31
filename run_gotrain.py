#!/usr/bin/env python
"""CLI entry point for the GoTrain agent.

    python run_gotrain.py --dataset m-eurosat --max-runs 6 --run-minutes 5
    python run_gotrain.py --dataset m-eurosat --dry-run

--dry-run never constructs a BedrockModel invocation and never calls
Bedrock (Bedrock is only reachable from inside SageMaker). It prints the
system prompt and the tool list, then runs one tool locally (list_datasets)
through the full agent/hook pipeline to prove the wiring works, and exits.
"""

import argparse
import json
import os

from gotrain import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GoTrain agent on one GEO-Bench dataset.")
    parser.add_argument("--dataset", required=True, help="GEO-Bench dataset name, e.g. m-eurosat")
    parser.add_argument("--max-runs", type=int, default=config.DEFAULT_MAX_RUNS)
    parser.add_argument("--run-minutes", type=float, default=config.DEFAULT_RUN_MINUTES)
    parser.add_argument("--max-tool-calls", type=int, default=config.DEFAULT_MAX_TOOL_CALLS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the system prompt and tool list, run one tool call locally, and exit. Never calls Bedrock.",
    )
    return parser.parse_args()


def run_dry(dataset: str, max_runs: int, run_minutes: float, max_tool_calls: int) -> None:
    from gotrain import control
    from gotrain.agent import build_agent, load_system_prompt

    control.reset()  # a stale paused/manual-control state from a previous session must not block this demo

    print("=" * 70)
    print("DRY RUN — Bedrock is never called in this mode.")
    print("=" * 70)

    print("\n--- System prompt (prompts/system_prompt.md) ---\n")
    print(load_system_prompt())

    agent = build_agent(dataset, max_runs=max_runs, run_minutes=run_minutes, max_tool_calls=max_tool_calls)

    print("\n--- Tool list ---")
    for name in agent.tool_names:
        print(f" - {name}")
    assert "final_test_eval" not in agent.tool_names, "final_test_eval must never be an agent tool"

    print("\n--- Proving the wiring: calling list_datasets() directly (no LLM involved) ---")
    result = agent.tool.list_datasets()
    print(json.dumps(result, indent=2))

    print("\nDry run complete. No Bedrock call was made.")


def run_live(dataset: str, max_runs: int, run_minutes: float, max_tool_calls: int) -> None:
    """Run the agent, with real pause/resume across turns via the engineer
    control plane (gotrain.control): a dashboard click sets control.json
    to PAUSE_REQUESTED/MANUAL_CONTROL; the agent's hook
    (gotrain/agent.py::LimitsAndLoggingHooks) then cancels its very next
    tool call and acknowledges the pause, which makes agent(prompt)
    return (the LLM has nothing left to do but respond in text). This
    loop then blocks on control.wait_for_resume() and, once an engineer
    resumes, calls agent(...) again on the SAME Agent instance to
    continue the same conversation.

    Caveat: this relies on strands.Agent.__call__ preserving conversation
    history across repeated calls on one instance, which could not be
    exercised against a live Bedrock call in this repo's test suite (no
    AWS access in this environment) -- the state-machine/polling logic
    itself (gotrain.control) is unit-tested; this specific multi-call
    integration needs a real run on the SageMaker lab to confirm.
    """
    from gotrain import control
    from gotrain.agent import build_agent

    control.reset()
    agent = build_agent(dataset, max_runs=max_runs, run_minutes=run_minutes, max_tool_calls=max_tool_calls)

    prompt = (
        f"Train the best model you can for the GEO-Bench dataset {dataset!r}. "
        f"You have a budget of {max_runs} training runs, {run_minutes} minutes per run, "
        f"and {max_tool_calls} total tool calls. Follow the workflow in your system prompt."
    )

    while True:
        agent(prompt)

        if control.get_state() == control.COMPLETED:
            break
        if not control.is_blocking():
            # The agent stopped on its own (gave up, hit a hard limit, or
            # decided it was done) without submitting a final model, and
            # nobody paused it -- nothing left for this loop to drive.
            break

        print(f"\n[GoTrain] Paused ({control.get_state()}). Waiting for the engineer to resume via the dashboard...")
        control.wait_for_resume()
        print("[GoTrain] Resumed. Continuing...")
        prompt = (
            "You have been resumed by the engineer after a pause or manual-control "
            "interval. Continue your workflow from where you left off -- call "
            "query_history first to see everything already tried, including "
            "anything the engineer may have done directly. Do not repeat a run "
            "that's already recorded."
        )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("GEO_BENCH_DIR", os.path.join(os.getcwd(), "data", "geobench"))

    if args.dry_run:
        run_dry(args.dataset, args.max_runs, args.run_minutes, args.max_tool_calls)
    else:
        run_live(args.dataset, args.max_runs, args.run_minutes, args.max_tool_calls)


if __name__ == "__main__":
    main()
