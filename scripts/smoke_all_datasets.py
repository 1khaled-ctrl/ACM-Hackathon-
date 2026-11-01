#!/usr/bin/env python
"""End-to-end smoke check of every GEO-Bench dataset through the real
pipeline, with tiny bounded configs. Never touches the TEST split (it never
calls final_eval and never opens split="test").

    python scripts/smoke_all_datasets.py                       # all 12
    python scripts/smoke_all_datasets.py m-eurosat m-NeonTree  # a subset

Per dataset, in an isolated child process (its own logs/ledger/history and
model registry under --out, so real mission history and model versions are
untouched): dataset spec, GeoGuard, DataIQ structural scan, launch_training
through the agent's own hooks (so the per-run timeout is injected exactly as
in a mission), checkpoint resolution + reload, saved normalization,
submit_final_model with its automatic StressLab, and history attribution.
The template's scratch run directory under runs/ is removed afterwards.

Output is ASCII only (the SageMaker console is cp1252-safe this way).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALL_DATASETS = [
    "m-eurosat", "m-so2sat", "m-bigearthnet", "m-brick-kiln", "m-forestnet", "m-pv4ger",
    "m-pv4ger-seg", "m-nz-cattle", "m-chesapeake", "m-cashew-plant", "m-SA-crop-type", "m-NeonTree",
]
# m-NeonTree with every band means 373 channels at 400x400 (the 369-channel
# Neon hyperspectral band upsampled 10x): ~239 MB and ~40 s per sample. The
# smoke run trains on RGB + Canopy Height explicitly and only REPORTS the
# all-bands estimate.
BANDS_OVERRIDE = {"m-NeonTree": ["Red", "Green", "Blue", "Canopy Height Model"]}


def _smoke_one(dataset: str, out_dir: Path, run_minutes: float) -> dict:
    os.environ["GOTRAIN_LOGS_DIR"] = str(out_dir / "logs")
    sys.path.insert(0, str(REPO_ROOT))

    from gotrain import agent as agent_mod
    from gotrain import control, history, ledger, registry, run_artifacts
    from gotrain.dataiq import run_structural_scan
    from gotrain.dataset_spec import estimate_training_memory, spec_for
    from gotrain.geoguard import run_geoguard
    from gotrain.stresslab import runner as stresslab_runner

    registry.MODELS_DIR = out_dir / "models"
    # Keep the automatic post-submission StressLab small: this checks that it
    # runs and persists, not the model's robustness.
    agent_mod.AUTO_STRESSLAB_MAX_SAMPLES = 8
    agent_mod.AUTO_STRESSLAB_TIME_BUDGET_SECONDS = 120
    report = {"dataset": dataset, "steps": {}}

    def step(name, fn):
        started = time.time()
        try:
            report["steps"][name] = {"ok": True, **(fn() or {}), "seconds": round(time.time() - started, 1)}
            return True
        except Exception as e:
            report["steps"][name] = {
                "ok": False, "error": f"{type(e).__name__}: {e}"[:400],
                "trace": traceback.format_exc()[-1200:], "seconds": round(time.time() - started, 1),
            }
            return False

    spec = spec_for(dataset)
    task_type = spec.task_type
    config = {
        "epochs": 1, "batch_size": 4, "max_train_samples": 8, "max_val_samples": 8,
        "pretrained": False, "amp": False,
    }
    if dataset in BANDS_OVERRIDE:
        config["bands"] = BANDS_OVERRIDE[dataset]

    step("spec", lambda: {
        "label_kind": spec.label_kind, "n_classes": spec.n_classes,
        "band_names": spec.n_bands, "channels_all_bands": spec.n_channels,
    })

    def _memory():
        all_bands = estimate_training_memory(dataset, task_type, {})
        chosen = estimate_training_memory(dataset, task_type, config)
        return {
            "full_split_all_bands_gb": round(all_bands["total_bytes"] / 1e9, 2) if all_bands else None,
            "smoke_config_mb": round(chosen["total_bytes"] / 1e6, 1) if chosen else None,
            "smoke_channels": chosen["channels"] if chosen else None,
        }

    step("memory_estimate", _memory)
    step("geoguard", lambda: {
        "findings": len(run_geoguard(dataset, task_type, max_samples=32)["findings"]),
    })

    def _dataiq():
        # Flagged samples are not a pipeline failure: GEO-Bench's own train
        # splits contain all-zero (no-data) patches, which DataIQ is meant to
        # quarantine. Fail only if something is flagged without a manifest.
        scan = run_structural_scan(dataset, task_type, max_samples=32)
        if scan["n_invalid"] and not scan["quarantine_manifest_id"]:
            raise AssertionError(f"{scan['n_invalid']} invalid sample(s) but no quarantine manifest was created")
        return {"scanned": scan["n_samples_scanned"], "invalid": scan["n_invalid"],
                "invalid_ids": scan["invalid_sample_ids"][:5], "quarantine_manifest": scan["quarantine_manifest_id"]}

    step("dataiq_structural", _dataiq)

    control.reset()  # exactly as run_gotrain.py does before building the agent
    session = history.start_session(dataset=dataset, max_runs=2, run_minutes=run_minutes)
    agent = agent_mod.build_agent(dataset, max_runs=2, run_minutes=run_minutes, max_tool_calls=20)
    state = {}

    def _text(result):
        return "".join(block.get("text", "") for block in result.get("content", []))

    def _train():
        result = agent.tool.launch_training(dataset=dataset, task_type=task_type, config=config)
        if result["status"] != "success":
            raise RuntimeError(_text(result)[-600:])
        record = json.loads(_text(result))
        state["run_id"] = record["run_id"]
        return {"run_id": record["run_id"], "val_metric": record["val_metric"], "version": record["version_id"]}

    if step("launch_training", _train):
        run_id = state["run_id"]

        def _artifacts():
            checkpoint = run_artifacts.checkpoint_path(dataset, run_id)
            metrics = json.loads(run_artifacts.metrics_path(dataset, run_id).read_text(encoding="utf-8"))
            if not metrics.get("normalization"):
                raise AssertionError("metrics.json has no normalization stats")
            if (REPO_ROOT / "runs" / dataset / run_id).name != run_id:
                raise AssertionError("ledger run_id does not name the run directory")
            return {"checkpoint": checkpoint.name, "channels": metrics.get("n_bands"),
                    "metric": metrics.get("metric_name") or "val_mean_iou"}

        step("checkpoint_and_metrics", _artifacts)

        def _submit():
            result = agent.tool.submit_final_model(dataset=dataset, run_id=run_id)
            if result["status"] != "success":
                raise RuntimeError(_text(result)[-600:])
            stress = stresslab_runner.read_report(dataset, run_id)
            if stress is None:
                events = [json.loads(line) for line in (out_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
                problem = [e for e in events if e.get("event") in ("stresslab_failed", "stresslab_unavailable")]
                raise AssertionError(f"StressLab produced no report: {problem[-1] if problem else 'no event'}")
            return {"control": control.get_state(), "stresslab": stress["summary"]["overall_verdict"],
                    "stresslab_samples": stress["summary"].get("n_samples")}

        step("submit_and_stresslab", _submit)

        def _history():
            history.finish_session(session["session_id"], status=history.COMPLETED)
            loaded = history.load_session(session["session_id"])
            if loaded["runs_completed"] != 1 or loaded["best_run_id"] != run_id or loaded["final_model_run_id"] != run_id:
                raise AssertionError(
                    f"history attribution wrong: runs={loaded['runs_completed']} best={loaded['best_run_id']} "
                    f"final={loaded['final_model_run_id']}"
                )
            return {"runs_completed": loaded["runs_completed"]}

        step("history_attribution", _history)
        shutil.rmtree(REPO_ROOT / "runs" / dataset / run_id, ignore_errors=True)
        try:
            (REPO_ROOT / "runs" / dataset).rmdir()  # only if now empty
        except OSError:
            pass

    report["passed"] = all(s["ok"] for s in report["steps"].values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=ALL_DATASETS)
    parser.add_argument("--out", default=None, help="scratch dir (default: a new temp dir)")
    parser.add_argument("--run-minutes", type=float, default=10.0)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        report = _smoke_one(args.datasets[0], Path(args.out), args.run_minutes)
        (Path(args.out) / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("SMOKE_RESULT " + json.dumps(report))
        return 0

    root = Path(args.out or tempfile.mkdtemp(prefix="gotrain_smoke_"))
    env = {**os.environ, "GEO_BENCH_DIR": os.environ.get("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))}
    reports = []
    for dataset in args.datasets:
        out_dir = root / dataset
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True)
        proc = subprocess.run(
            [sys.executable, __file__, dataset, "--child", "--out", str(out_dir), "--run-minutes", str(args.run_minutes)],
            capture_output=True, text=True, env=env, encoding="utf-8", errors="replace",
        )
        line = next((l for l in proc.stdout.splitlines() if l.startswith("SMOKE_RESULT ")), None)
        report = json.loads(line[len("SMOKE_RESULT "):]) if line else {
            "dataset": dataset, "passed": False,
            "steps": {"child": {"ok": False, "error": f"exit {proc.returncode}: {proc.stderr[-800:]}"}},
        }
        reports.append(report)
        status = "PASS" if report["passed"] else "FAIL"
        print(f"{status}  {dataset}")
        for name, s in report["steps"].items():
            detail = {k: v for k, v in s.items() if k not in ("ok", "trace")}
            print(f"    {'ok  ' if s['ok'] else 'FAIL'} {name}: {json.dumps(detail)}".encode("ascii", "replace").decode())
    (root / "smoke_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\n{sum(r['passed'] for r in reports)}/{len(reports)} datasets passed. Full report: {root / 'smoke_report.json'}")
    return 0 if all(r["passed"] for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
