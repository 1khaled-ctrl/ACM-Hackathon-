"""DataIQ: autonomous training-data intelligence for GoTrain.

run_structural_scan() is the entry point for the deterministic, model-free
checks (corrupted/invalid samples, exact duplicates) -- it reuses
gotrain.geoguard.diagnostics.load_train_sample so GEO-Bench's TRAIN split
is read exactly the same way GeoGuard already reads it (never validation,
never test). Objectively corrupted samples found here are auto-quarantined
under policy and immediately folded into a usable manifest; nothing else
is ever excluded without an engineer approving it first.

Model-assisted diagnostics (per-example training loss, suspected label
noise) live in gotrain.dataiq.loss_scan and are invoked separately, after
a baseline model exists to compute them against.
"""

from gotrain.dataiq import decision, diagnostics, manifest, report
from gotrain.geoguard import diagnostics as geoguard_diagnostics

__all__ = ["decision", "diagnostics", "manifest", "report", "run_structural_scan"]


def run_structural_scan(dataset: str, task_type: str, max_samples: int = 300) -> dict:
    images, labels, band_names, task_specs = geoguard_diagnostics.load_train_sample(
        dataset, task_type, max_samples
    )

    # n_classes must come from the dataset's own spec, not be inferred from
    # label VALUES (int(labels.max())+1): for multi-label (e.g.
    # m-bigearthnet), labels.max() is always 1 (it's a {0,1} multi-hot
    # array), which would wrongly compute n_classes=2 instead of the real
    # 43 -- a silent, not a crashing, bug, and a worse one for that.
    from gotrain.dataset_spec import spec_for

    spec = spec_for(dataset, task_specs=task_specs)
    n_classes = spec.n_classes

    structural_records = diagnostics.scan_samples(
        images, labels, task_type, n_classes, is_multi_label=spec.is_multi_label
    )
    duplicate_groups = diagnostics.find_exact_duplicates(images)
    summary = diagnostics.summarize_scan(structural_records, duplicate_groups)

    quarantine_manifest_id = None
    if summary["invalid_sample_ids"]:
        for sample_id in summary["invalid_sample_ids"]:
            row = next(r for r in structural_records if r["sample_id"] == sample_id)
            decision.propose_action(
                dataset=dataset,
                sample_id=sample_id,
                action="QUARANTINE_INVALID_SAMPLE",
                evidence={"problems": row["problems"]},
                severity=row["severity"],
                reason="objectively invalid: " + "; ".join(row["problems"]),
            )
        quarantine_manifest = manifest.create_manifest(
            dataset=dataset,
            excluded_sample_ids=summary["invalid_sample_ids"],
            reason="auto-quarantine of objectively corrupted/invalid TRAIN samples",
            created_by="dataiq_structural_scan",
        )
        quarantine_manifest_id = quarantine_manifest["manifest_id"]

    full_report = {
        "dataset": dataset,
        "task_type": task_type,
        "n_samples_scanned": summary["n_samples_scanned"],
        "n_invalid": summary["n_invalid"],
        "invalid_sample_ids": summary["invalid_sample_ids"],
        "n_duplicate_groups": summary["n_duplicate_groups"],
        "duplicate_sample_ids": summary["duplicate_sample_ids"],
        "quarantine_manifest_id": quarantine_manifest_id,
        "band_names": band_names,
    }
    report.write_report(dataset, full_report)
    return full_report
