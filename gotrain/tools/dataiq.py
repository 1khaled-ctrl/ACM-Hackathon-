"""Agent-facing wrappers around gotrain.dataiq: logs the call and current
status the same way every other tool in this package does (see
gotrain/tools/geoguard.py for the identical pattern).
"""

from typing import Optional

from gotrain import dataiq, status
from gotrain.dataiq import decision as dataiq_decision
from gotrain.dataiq import loss_scan as dataiq_loss_scan
from gotrain.dataiq import manifest as dataiq_manifest
from gotrain.logging_utils import log_event


def run_dataiq_structural_scan(dataset: str, task_type: str, max_samples: int = 300) -> dict:
    status.set_action("scanning for corrupted/duplicate samples (DataIQ)", where=dataset)
    log_event("dataiq_structural_scan_started", dataset=dataset, task_type=task_type, max_samples=max_samples)

    report = dataiq.run_structural_scan(dataset, task_type, max_samples=max_samples)

    log_event(
        "dataiq_structural_scan_finished", dataset=dataset, n_invalid=report["n_invalid"],
        n_duplicate_groups=report["n_duplicate_groups"], quarantine_manifest_id=report["quarantine_manifest_id"],
    )
    status.set_last_event(f"DataIQ structural scan of {dataset}: {report['n_invalid']} invalid sample(s) found")
    return report


def run_dataiq_loss_scan(dataset: str, task_type: str, run_id: str, max_samples: Optional[int] = None) -> dict:
    """Model-assisted per-example loss analysis against `run_id`'s saved
    checkpoint. Needs torch + the real downloaded dataset -- see
    gotrain/dataiq/loss_scan.py's module docstring for the full caveat.
    """
    status.set_action("scanning for suspicious training examples (DataIQ loss scan)", where=dataset)
    log_event("dataiq_loss_scan_started", dataset=dataset, task_type=task_type, run_id=run_id)

    if task_type == "classification":
        summary = dataiq_loss_scan.scan_classification_run(dataset, run_id, max_samples=max_samples)
    elif task_type == "segmentation":
        summary = dataiq_loss_scan.scan_segmentation_run(dataset, run_id, max_samples=max_samples)
    else:
        raise ValueError(f"task_type must be 'classification' or 'segmentation', got {task_type!r}")

    status.set_last_event(f"DataIQ loss scan of {dataset}/{run_id}: {summary['n_flagged']} sample(s) flagged")
    return summary


def create_data_manifest(dataset: str, excluded_sample_ids: list, reason: str, base_manifest_id: str = dataiq_manifest.BASE_MANIFEST_ID) -> dict:
    manifest = dataiq_manifest.create_manifest(
        dataset=dataset, excluded_sample_ids=excluded_sample_ids, reason=reason,
        created_by="agent", base_manifest_id=base_manifest_id,
    )
    status.set_last_event(f"Created DataIQ manifest {manifest['manifest_id']} for {dataset} ({manifest['n_excluded']} excluded)")
    return manifest


def list_data_proposals(dataset: Optional[str] = None, status_filter: Optional[str] = None) -> list:
    return dataiq_decision.list_data_proposals(dataset=dataset, status=status_filter)
