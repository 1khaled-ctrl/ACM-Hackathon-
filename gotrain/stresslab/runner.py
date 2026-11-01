"""Model-assisted robustness evaluation (StressLab): after a run's
checkpoint is trained, replay the SAME validation-split evaluation under
gotrain.stresslab.perturbations' battery (sensor noise, dropped bands, a
lighting shift, simple geometric flips) to check whether good clean
validation accuracy also means the model is robust, or whether it's
brittle to realistic nuisances GEO-Bench's clean benchmark never
exercises.

Deliberately evaluated on VALIDATION, never TEST -- same split-isolation
discipline as gotrain/dataiq/loss_scan.py and gotrain/tools/final_eval.py.
Reuses the exact build_model/build_dataset_arrays/compute_*_metrics
functions the training templates and final_eval.py already use -- no
separate, divergent loading path.

NEEDS TORCH (+ geobench + the real downloaded dataset) to run -- not
executable in this project's local/CI environment, same caveat as
loss_scan.py/final_eval.py. The orchestration logic itself (looping over
perturbations, classifying fragility, writing/reading the report) is
exercised in tests via injected fake evaluate functions, so none of that
logic is unverified -- only the actual torch forward pass is.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from gotrain import ledger
from gotrain.logging_utils import LOGS_DIR, log_event
from gotrain.stresslab import perturbations

STRESSLAB_DIR = LOGS_DIR / "stresslab"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "runs"

# Judgment calls, like calibration.py's ECE thresholds -- how much of a
# relative metric drop under a perturbation counts as merely
# noise-tolerant degradation vs. "fragile" vs. "severely fragile".
FRAGILE_RELATIVE_DROP = 0.15
SEVERELY_FRAGILE_RELATIVE_DROP = 0.40


def _run_dir(dataset: str, run_id: str) -> Path:
    run_dir = RUNS_DIR / dataset / run_id
    checkpoint = run_dir / "best_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run {run_id!r} at {checkpoint}. Expected "
            "runs/<dataset>/<run_id>/best_model.pt, written by the training templates."
        )
    return run_dir


def _get_run_record(dataset: str, run_id: str) -> dict:
    record = ledger.get_run(run_id)
    if record is None or record["dataset"] != dataset:
        raise ValueError(f"run_id {run_id!r} not found for dataset {dataset!r}")
    return record


def _silent_logger(name: str) -> "logging.Logger":
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def classify_fragility(relative_drop: Optional[float]) -> str:
    """"robust" / "fragile" / "severely_fragile" / "unknown" (when the
    clean baseline metric was 0 or otherwise undefined, so a relative
    drop can't be computed) -- never silently defaults to "robust" when
    the data to judge that is actually missing.
    """
    if relative_drop is None:
        return "unknown"
    if relative_drop < FRAGILE_RELATIVE_DROP:
        return "robust"
    if relative_drop < SEVERELY_FRAGILE_RELATIVE_DROP:
        return "fragile"
    return "severely_fragile"


def run_perturbation_battery(evaluate: Callable[[np.ndarray], float], clean_images: np.ndarray) -> dict:
    """Shared orchestration: score `evaluate(images) -> metric` on the
    unperturbed images, then on every (perturbation, severity) pair from
    gotrain.stresslab.perturbations, computing each one's relative drop
    and fragility classification against the clean baseline.

    Pulled out from run_stresslab_classification/segmentation so this
    control-flow logic (not the torch forward pass inside `evaluate`) is
    directly testable with a fake `evaluate` and a tiny synthetic array
    -- see tests/test_stresslab_runner.py.
    """
    clean_metric = evaluate(clean_images)

    results = []
    for name, fn in perturbations.PERTURBATIONS.items():
        for severity in perturbations.SEVERITIES_FOR[name]:
            perturbed_images = fn(clean_images, severity)
            perturbed_metric = evaluate(perturbed_images)
            relative_drop = (
                (clean_metric - perturbed_metric) / clean_metric if clean_metric else None
            )
            results.append({
                "perturbation": name,
                "severity": severity,
                "clean_metric": clean_metric,
                "perturbed_metric": perturbed_metric,
                "relative_drop": relative_drop,
                "fragility": classify_fragility(relative_drop),
            })

    n_fragile = sum(1 for r in results if r["fragility"] in ("fragile", "severely_fragile"))
    return {
        "clean_metric": clean_metric,
        "results": results,
        "n_perturbations_tested": len(results),
        "n_fragile": n_fragile,
        "overall_verdict": "robust" if n_fragile == 0 else f"fragile under {n_fragile}/{len(results)} perturbation(s)",
    }


def _report_dir(dataset: str) -> Path:
    return STRESSLAB_DIR / dataset


def _record_and_log(dataset: str, run_id: str, task_type: str, battery_result: dict) -> dict:
    report_dir = _report_dir(dataset)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": dataset, "run_id": run_id, "task_type": task_type,
        "clean_metric": battery_result["clean_metric"],
        "n_perturbations_tested": battery_result["n_perturbations_tested"],
        "n_fragile": battery_result["n_fragile"],
        "overall_verdict": battery_result["overall_verdict"],
    }
    payload = {"summary": summary, "results": battery_result["results"]}

    fd, tmp_path = tempfile.mkstemp(dir=report_dir, prefix=".stresslab_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, report_dir / f"{run_id}.json")
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    log_event("stresslab_finished", **summary)
    return summary


def read_report(dataset: str, run_id: str) -> Optional[dict]:
    path = _report_dir(dataset) / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_stresslab_classification(dataset: str, run_id: str, max_samples: Optional[int] = None) -> dict:
    import geobench
    import torch

    from gotrain.templates import train_classification as tc

    run_dir = _run_dir(dataset, run_id)
    record = _get_run_record(dataset, run_id)
    config = record["config"]
    logger = _silent_logger(f"stresslab.classification.{run_id}")

    task_dir = tc.find_task_dir(dataset)
    task_specs = geobench.load_task_specs(task_dir)
    all_band_names = [b.name for b in task_specs.bands_info]
    band_names = config.get("bands") or all_band_names
    n_bands = len(band_names)
    n_classes = task_specs.label_type.n_classes

    train_images, _ = tc.build_dataset_arrays(task_specs, "train", band_names, config.get("max_train_samples"), logger)
    mean, std = tc.compute_train_band_stats(train_images, logger)
    del train_images

    val_images, val_labels = tc.build_dataset_arrays(task_specs, "validation", band_names, max_samples, logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tc.build_model(config.get("backbone", "resnet18"), False, n_bands, n_classes, band_names)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model = model.to(device)
    model.eval()

    def evaluate(images: np.ndarray) -> float:
        normalized = (images.astype(np.float64) - mean) / std
        normalized = np.transpose(normalized, (0, 3, 1, 2)).astype(np.float32)
        batch_size = 32
        all_preds = []
        with torch.no_grad():
            for start in range(0, len(normalized), batch_size):
                batch = torch.from_numpy(normalized[start:start + batch_size]).float().to(device)
                logits = model(batch)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
        preds = np.concatenate(all_preds, axis=0)
        accuracy, _ = tc.compute_accuracy_metrics(preds, val_labels, n_classes)
        return accuracy

    battery_result = run_perturbation_battery(evaluate, val_images)
    return _record_and_log(dataset, run_id, "classification", battery_result)


def run_stresslab_segmentation(dataset: str, run_id: str, max_samples: Optional[int] = None) -> dict:
    import geobench
    import torch

    from gotrain.templates import train_segmentation as ts

    run_dir = _run_dir(dataset, run_id)
    record = _get_run_record(dataset, run_id)
    config = record["config"]
    logger = _silent_logger(f"stresslab.segmentation.{run_id}")

    task_dir = ts.find_task_dir(dataset)
    task_specs = geobench.load_task_specs(task_dir)
    all_band_names = [b.name for b in task_specs.bands_info]
    band_names = config.get("bands") or all_band_names
    n_bands = len(band_names)
    n_classes = task_specs.label_type.n_classes

    train_images, _ = ts.build_dataset_arrays(task_specs, "train", band_names, config.get("max_train_samples"), logger)
    mean, std = ts.compute_train_band_stats(train_images, logger)
    del train_images

    val_images, val_masks = ts.build_dataset_arrays(task_specs, "validation", band_names, max_samples, logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ts.build_model(
        config.get("architecture", "unet"), config.get("encoder", "resnet34"), False, n_bands, n_classes
    )
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model = model.to(device)
    model.eval()

    def evaluate(images: np.ndarray) -> float:
        normalized = (images.astype(np.float64) - mean) / std
        normalized = np.transpose(normalized, (0, 3, 1, 2)).astype(np.float32)
        batch_size = 8
        all_preds = []
        with torch.no_grad():
            for start in range(0, len(normalized), batch_size):
                batch = torch.from_numpy(normalized[start:start + batch_size]).float().to(device)
                logits = model(batch)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
        preds = np.concatenate(all_preds, axis=0)
        mean_iou, _ = ts.compute_mean_iou(preds, val_masks, n_classes)
        return mean_iou

    battery_result = run_perturbation_battery(evaluate, val_images)
    return _record_and_log(dataset, run_id, "segmentation", battery_result)
