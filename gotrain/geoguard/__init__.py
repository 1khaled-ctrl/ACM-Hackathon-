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

    class_names = list(getattr(task_specs.label_type, "class_names", []))
    if not class_names:
        n_classes = getattr(task_specs.label_type, "n_classes", int(labels.max()) + 1)
        class_names = [str(i) for i in range(n_classes)]
    class_diag = diagnostics.compute_class_diagnostics(labels, class_names, is_pixelwise=(task_type == "segmentation"))

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
