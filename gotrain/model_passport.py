"""Model Passport: a single, reproducible record of what a dataset's
submitted model actually is and how it got there -- model identity,
training configuration, validation results, governance/approval history,
and (new in this pass) a Data Provenance and Quality section covering
exactly which TRAIN samples were excluded and why, and what was approved.

Built entirely from data this project ALREADY recorded (the ledger, the
model registry, GeoGuard/DataIQ reports, the approval queue) -- never
invents a value. A field with nothing to report is explicitly null/empty
rather than guessed, matching the "do not invent missing metadata"
convention established in gotrain/geoguard. Test-split results are only
ever included if a human has already run run_final_eval.py (read from
events.jsonl's one-time "evaluation"/split="test" record) -- generating a
passport never triggers or requires a test evaluation.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

from gotrain import approval, ledger, registry
from gotrain.dataiq import decision as dataiq_decision
from gotrain.dataiq import manifest as dataiq_manifest
from gotrain.dataiq import report as dataiq_report
from gotrain.geoguard import report as geoguard_report
from gotrain.logging_utils import EVENTS_PATH, LOGS_DIR

PASSPORTS_DIR = LOGS_DIR / "model_passports"


def _sha256_of_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_test_result(dataset: str, run_id: str) -> Optional[dict]:
    """Read (never trigger) a prior one-time test-split evaluation for
    this run, from events.jsonl's "evaluation" record
    (gotrain.logging_utils.log_evaluation, written only by
    run_final_eval.py). Returns None if no such record exists.
    """
    if not EVENTS_PATH.exists():
        return None
    with open(EVENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "evaluation" and event.get("split") == "test" and event.get("model") == run_id:
                return event
    return None


def _data_provenance(dataset: str, run_record: dict) -> dict:
    manifest_id = (run_record.get("config") or {}).get("manifest_id")
    manifest = dataiq_manifest.get_manifest(dataset, manifest_id) if manifest_id else dataiq_manifest.get_manifest(
        dataset, dataiq_manifest.BASE_MANIFEST_ID
    )

    structural = dataiq_report.read_report(dataset)
    approved_actions = [
        p for p in dataiq_decision.list_data_proposals(dataset=dataset) if p["status"] == approval.APPROVED
    ]
    rejected_actions = [
        p for p in dataiq_decision.list_data_proposals(dataset=dataset) if p["status"] == approval.REJECTED
    ]

    return {
        "original_dataset_version": "GEO-Bench (as downloaded; see data/dataset_summary.md)",
        "training_manifest_id": manifest["manifest_id"] if manifest else None,
        "training_manifest_base_id": manifest.get("base_manifest_id") if manifest else None,
        "excluded_sample_count": manifest["n_excluded"] if manifest else 0,
        "excluded_sample_ids": manifest["excluded_sample_ids"] if manifest else [],
        "approved_data_actions": [
            {"sample_id": p["sample_id"], "action": p["action"], "decided_by": p["decided_by"], "reason": p.get("decision_reason")}
            for p in approved_actions
        ],
        "rejected_data_actions": [
            {"sample_id": p["sample_id"], "action": p["action"], "decided_by": p["decided_by"], "reason": p.get("decision_reason")}
            for p in rejected_actions
        ],
        "structural_scan_summary": {
            "n_invalid": structural.get("n_invalid"), "n_duplicate_groups": structural.get("n_duplicate_groups"),
        } if structural else None,
    }


def generate_passport(dataset: str, run_id: Optional[str] = None) -> dict:
    """Build a Model Passport for `run_id` (defaulting to the run
    submitted as final for `dataset`, via submit_final_model). Raises
    ValueError if no run_id is given and none has been submitted yet.
    """
    history = ledger.get_history(dataset)
    if run_id is None:
        submitted = [r for r in history if r.get("submitted_final")]
        if not submitted:
            raise ValueError(f"no run has been submitted as final for {dataset!r} yet")
        run_record = sorted(submitted, key=lambda r: r["timestamp"])[-1]
        run_id = run_record["run_id"]
    else:
        run_record = next((r for r in history if r["run_id"] == run_id), None)
        if run_record is None:
            raise ValueError(f"run_id {run_id!r} not found for dataset {dataset!r}")

    config = run_record.get("config") or {}
    task_type = run_record.get("task_type")
    version_id = run_record.get("version_id")

    checkpoint_sha256 = None
    if version_id:
        checkpoint_path = registry.get_version_dir(dataset, version_id) / "model.pt"
        checkpoint_sha256 = _sha256_of_file(checkpoint_path)

    geoguard = geoguard_report.read_report(dataset)
    test_result = _find_test_result(dataset, run_id)
    lineage = ledger.get_lineage(run_id)

    passport = {
        "model_id": run_id,
        "dataset": dataset,
        "task_type": task_type,
        "architecture": config.get("backbone") or f"{config.get('architecture')}/{config.get('encoder')}",
        "pretrained": config.get("pretrained"),
        "fine_tuning_strategy": config.get("fine_tuning_strategy", "full"),
        "input_channels": (geoguard or {}).get("pixel_diagnostics", {}).get("n_channels"),
        "sensor": (geoguard or {}).get("sensor"),
        "preprocessing_configuration": {
            "band_classification": (geoguard or {}).get("band_classification"),
            "geoguard_recommended_config": (geoguard or {}).get("recommended_config"),
        } if geoguard else None,
        "hyperparameters": config,
        "random_seed": config.get("seed"),
        "validation_metrics": {
            "val_metric": run_record.get("val_metric"),
            "extra_metrics": run_record.get("extra_metrics"),
        },
        "test_metrics": test_result.get("metrics") if test_result else None,
        "test_evaluated_at": test_result.get("timestamp") if test_result else None,
        "experiment_history": {
            "n_ancestors": len(lineage) - 1,
            "lineage_run_ids": [r["run_id"] for r in lineage],
            "hypothesis": run_record.get("hypothesis"),
            "hypothesis_outcome": run_record.get("hypothesis_outcome"),
        },
        "data_provenance": _data_provenance(dataset, run_record),
        "checkpoint_version": version_id,
        "checkpoint_sha256": checkpoint_sha256,
        "engineer_approval": {
            "config_gate_decisions": [
                p for p in approval.list_proposals(kind="config_gate") if p["dataset"] == dataset
            ],
            "submitted_final": bool(run_record.get("submitted_final")),
        },
        "known_limitations": _known_limitations(dataset, run_record, geoguard, test_result),
        "robustness_results": None,  # StressLab not implemented in this pass -- explicitly absent, not fabricated
    }
    return passport


def _known_limitations(dataset: str, run_record: dict, geoguard: Optional[dict], test_result: Optional[dict]) -> list:
    limitations = []
    if test_result is None:
        limitations.append("Test-split results not yet available -- run_final_eval.py has not been run for this dataset.")
    if run_record.get("hypothesis_outcome") == "inconclusive":
        limitations.append("The hypothesis behind this run's final config change was graded 'inconclusive' (delta too small to read as signal).")
    if geoguard:
        for finding in geoguard.get("findings", []):
            if "saturation" in finding.lower() or "nan" in finding.lower() or "near-constant" in finding.lower():
                limitations.append(f"GeoGuard finding not fully resolved: {finding}")
    if dataset == "m-nz-cattle":
        limitations.append("Validation set is very small (66 samples) -- validation metric differences between runs are noisy.")
    return limitations


def write_passport(dataset: str, passport: dict) -> Path:
    import os
    import tempfile

    PASSPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PASSPORTS_DIR / f"{dataset}.json"
    fd, tmp_path = tempfile.mkstemp(dir=PASSPORTS_DIR, prefix=".passport_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(passport, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path


def read_passport(dataset: str) -> Optional[dict]:
    path = PASSPORTS_DIR / f"{dataset}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_passport_markdown(passport: dict) -> str:
    lines = [
        f"# Model Passport — {passport['dataset']}",
        "",
        f"**Model ID (run_id):** `{passport['model_id']}`  ",
        f"**Architecture:** {passport['architecture']} (pretrained={passport['pretrained']}, "
        f"fine_tuning_strategy={passport['fine_tuning_strategy']})  ",
        f"**Sensor:** {passport.get('sensor') or 'unknown'} · **Input channels:** {passport.get('input_channels') or 'unknown'}",
        "",
        "## Validation",
        f"- val_metric: {passport['validation_metrics']['val_metric']}",
        "",
        "## Test (human-triggered, one-time)",
        f"- {'Not yet evaluated' if passport['test_metrics'] is None else passport['test_metrics']}",
        "",
        "## Data Provenance and Quality",
        f"- Training manifest: `{passport['data_provenance']['training_manifest_id']}` "
        f"({passport['data_provenance']['excluded_sample_count']} sample(s) excluded)",
        f"- Approved data actions: {len(passport['data_provenance']['approved_data_actions'])}",
        f"- Rejected data actions: {len(passport['data_provenance']['rejected_data_actions'])}",
        "",
        "## Experiment History",
        f"- {passport['experiment_history']['n_ancestors']} ancestor run(s)",
        f"- Hypothesis: {passport['experiment_history']['hypothesis']}",
        f"- Hypothesis outcome: {passport['experiment_history']['hypothesis_outcome']}",
        "",
        "## Checkpoint",
        f"- Version: {passport['checkpoint_version']}",
        f"- SHA-256: `{passport['checkpoint_sha256']}`",
        "",
        "## Known Limitations",
    ]
    if passport["known_limitations"]:
        lines += [f"- {item}" for item in passport["known_limitations"]]
    else:
        lines.append("- None recorded.")
    lines += ["", "## Robustness Results", "- Not run in this pass (StressLab not implemented)."]
    return "\n".join(lines)
