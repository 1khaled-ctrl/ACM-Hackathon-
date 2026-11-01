"""GeoGuard: intelligent, dataset-aware, sensor-aware preprocessing
diagnostics for GoTrain.

Public entry point: run_geoguard(dataset, task_type, max_samples). Reads a
bounded sample of the TRAIN split only, computes pixel/class diagnostics,
runs the deterministic decision engine for a config recommendation, and
persists the result to logs/geoguard/<dataset>.json.

GeoGuard never trains anything, never touches validation or test data, and
never executes model-authored code -- decision.recommend only ever returns
values already allowlisted by gotrain.config_schema.
"""

from gotrain.geoguard import decision, diagnostics, qa, report
from gotrain.geoguard.report import format_summary, read_report, write_report

__all__ = [
    "decision",
    "diagnostics",
    "qa",
    "report",
    "run_geoguard",
    "format_summary",
    "read_report",
    "write_report",
]


def run_geoguard(dataset: str, task_type: str, max_samples: int = 300) -> dict:
    images, labels, band_names, task_specs = diagnostics.load_train_sample(dataset, task_type, max_samples)

    pixel_diag = diagnostics.compute_pixel_diagnostics(images, band_names)
    band_class = diagnostics.classify_bands(band_names)

    from gotrain.dataset_spec import spec_for

    spec = spec_for(dataset, task_specs=task_specs)
    class_names = spec.class_names or [str(i) for i in range(spec.n_classes)]
    class_diag = diagnostics.compute_class_diagnostics(
        labels, class_names, is_pixelwise=(task_type == "segmentation"), is_multi_label=spec.is_multi_label
    )

    dataset_notes = diagnostics.DATASET_NOTES.get(dataset, {"sensor": "unknown", "warnings": []})

    decision_result = decision.recommend(
        dataset=dataset,
        task_type=task_type,
        pixel_diagnostics=pixel_diag,
        class_diagnostics=class_diag,
        band_classification=band_class,
        dataset_notes=dataset_notes,
    )

    full_report = {
        "dataset": dataset,
        "task_type": task_type,
        "sensor": dataset_notes.get("sensor", "unknown"),
        "n_samples_inspected": pixel_diag["n_samples"],
        "band_classification": band_class,
        "pixel_diagnostics": pixel_diag,
        "class_diagnostics": class_diag,
        "findings": decision_result["findings"],
        "rationale": decision_result["rationale"],
        "recommended_config": decision_result["recommended_config"],
    }
    report.write_report(dataset, full_report)
    return full_report
