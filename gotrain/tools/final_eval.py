"""The ONLY place in this codebase allowed to touch the test split.

final_test_eval runs a trained model's evaluation on the test split for one
dataset, exactly once. A lock file (logs/test_lock_<dataset>.json) is
written atomically on first use; any subsequent call for the same dataset
is refused rather than silently re-running — re-evaluating on test defeats
the point of holding it out, so this has to fail loudly, not retry quietly.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gotrain.logging_utils import LOGS_DIR, log_evaluation

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "runs"

TEST_LOCK_TEMPLATE = "test_lock_{dataset}.json"


class TestAlreadyEvaluatedError(RuntimeError):
    pass


def _lock_path(dataset: str) -> Path:
    return LOGS_DIR / TEST_LOCK_TEMPLATE.format(dataset=dataset)


def _acquire_lock(dataset: str, run_id: str) -> None:
    """Atomically create the lock file; raises if it already exists.

    Uses O_CREAT|O_EXCL so a concurrent second call can't race past the
    exists-check between check and write.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path(dataset)
    payload = {
        "dataset": dataset,
        "run_id": run_id,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        raise TestAlreadyEvaluatedError(
            f"Test split for {dataset!r} was already evaluated "
            f"(run_id={existing.get('run_id')!r} at {existing.get('evaluated_at')!r}). "
            "The test split may only be evaluated once per dataset."
        ) from None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def final_test_eval(dataset: str, run_id: str, eval_fn) -> dict:
    """Run `eval_fn` (a no-arg callable returning a metrics dict) against the
    test split for `dataset`, tied to `run_id` (the chosen final model),
    exactly once.

    Raises TestAlreadyEvaluatedError if this dataset's test split has
    already been evaluated. `eval_fn` is the caller's responsibility to
    build — it must load the test split itself; this function only
    enforces the one-time lock and logs the result.
    """
    _acquire_lock(dataset, run_id)
    metrics = eval_fn()
    record = log_evaluation(task=dataset, split="test", metrics=metrics, model=run_id)
    return record


def resolve_checkpoint(dataset: str, run_id: str) -> Path:
    """Locate the run's checkpoint WITHOUT touching the test split -- call
    this before final_test_eval so a missing checkpoint can't burn the
    one-time test lock (the lock is taken before eval_fn runs and is never
    released)."""
    from gotrain import run_artifacts

    return run_artifacts.checkpoint_path(dataset, run_id, RUNS_DIR)


def _normalization(dataset: str, run_id: str, load_train_images, template, logger):
    """The run's own saved (mean, std) from training; recomputed from the
    TRAIN split (same max_train_samples as training) only for runs recorded
    before the templates saved it. Never reads the test split."""
    from gotrain import run_artifacts

    saved = run_artifacts.load_normalization(dataset, run_id, RUNS_DIR)
    if saved is not None:
        return saved
    return template.compute_train_band_stats(load_train_images(), logger)


def _silent_logger(name: str) -> "logging.Logger":
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def build_classification_test_eval_fn(dataset: str, run_id: str, config: dict):
    """Build the no-arg eval_fn for a real `task_type="classification"` run.

    Loads the saved checkpoint from runs/<dataset>/<run_id>/best_model.pt,
    recomputes TRAIN-only normalization stats exactly as the training
    template did (same train split, same deterministic code -> same
    mean/std, with no need to persist them separately), then loads the
    TEST split -- the one and only test-split read in this call chain --
    and reports accuracy. Reuses the training template's model/array
    helpers (which are themselves test-split-blind) for everything except
    the final "read the test split" step, which happens only here.
    """
    import numpy as np
    import torch
    import geobench

    from gotrain.templates import train_classification as tc

    checkpoint = resolve_checkpoint(dataset, run_id)
    logger = _silent_logger(f"final_eval.classification.{run_id}")

    from gotrain.dataset_spec import spec_for

    task_dir = tc.find_task_dir(dataset)
    task_specs = geobench.load_task_specs(task_dir)
    all_band_names = [b.name for b in task_specs.bands_info]
    band_names = config.get("bands") or all_band_names
    spec = spec_for(dataset, task_specs=task_specs)
    n_classes = spec.n_classes
    is_multi_label = spec.is_multi_label

    mean, std = _normalization(dataset, run_id, lambda: tc.build_dataset_arrays(
        task_specs, "train", band_names, config.get("max_train_samples"), logger)[0], tc, logger)

    from gotrain import data_loading

    test_ds = task_specs.get_dataset(split="test")
    test_images, test_labels = [], []
    for i in range(len(test_ds)):
        sample = test_ds[i]
        test_images.append(data_loading.pack_sample(sample, band_names, cast=np.float32))
        if is_multi_label:
            test_labels.append(np.asarray(sample.label, dtype=np.float32))
        else:
            test_labels.append(int(sample.label))
    test_images = np.stack(test_images, axis=0)
    test_labels = np.stack(test_labels, axis=0) if is_multi_label else np.array(test_labels, dtype=np.int64)
    n_bands = test_images.shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tc.build_model(config.get("backbone", "resnet18"), False, n_bands, n_classes, band_names)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    image_size = config.get("image_size")

    batch_size = 32
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(test_images), batch_size):
            batch = tc.prepare_batch(test_images[start:start + batch_size], mean, std, image_size).to(device)
            logits = model(batch)
            if is_multi_label:
                preds = (torch.sigmoid(logits) >= 0.5).to(torch.int64).cpu().numpy()
            else:
                preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
    preds = np.concatenate(all_preds, axis=0)

    if is_multi_label:
        ml_metrics = tc.compute_multilabel_metrics(preds, test_labels.astype(np.int64), n_classes)
        return {
            "micro_f1": ml_metrics["micro_f1"],
            "macro_f1": ml_metrics["macro_f1"],
            "exact_match": ml_metrics["exact_match"],
            "per_class_f1": ml_metrics["per_class_f1"],
            "n_test_samples": int(len(test_labels)),
            "band_names": band_names,
        }

    accuracy, per_class_accuracy = tc.compute_accuracy_metrics(preds, test_labels, n_classes)
    return {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "n_test_samples": int(len(test_labels)),
        "band_names": band_names,
    }


def build_segmentation_test_eval_fn(dataset: str, run_id: str, config: dict):
    """Build the no-arg eval_fn for a real `task_type="segmentation"` run.

    Mirrors build_classification_test_eval_fn's approach: recompute TRAIN
    normalization stats, load the checkpoint, and evaluate on TEST (read
    only here) using the same mean-IoU computation as the training
    template's validation loop.
    """
    import numpy as np
    import torch
    import geobench

    from gotrain.templates import train_segmentation as ts

    checkpoint = resolve_checkpoint(dataset, run_id)
    logger = _silent_logger(f"final_eval.segmentation.{run_id}")

    task_dir = ts.find_task_dir(dataset)
    task_specs = geobench.load_task_specs(task_dir)
    all_band_names = [b.name for b in task_specs.bands_info]
    band_names = config.get("bands") or all_band_names
    n_classes = task_specs.label_type.n_classes

    mean, std = _normalization(dataset, run_id, lambda: ts.build_dataset_arrays(
        task_specs, "train", band_names, config.get("max_train_samples"), logger)[0], ts, logger)

    from gotrain import data_loading

    test_ds = task_specs.get_dataset(split="test")
    test_images, test_masks = [], []
    for i in range(len(test_ds)):
        sample = test_ds[i]
        test_images.append(data_loading.pack_sample(sample, band_names))
        test_masks.append(np.asarray(sample.label.data))
    test_images = np.stack(test_images, axis=0)
    test_masks = np.stack(test_masks, axis=0)
    n_bands = test_images.shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ts.build_model(
        config.get("architecture", "unet"), config.get("encoder", "resnet34"), False, n_bands, n_classes
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    batch_size = 8
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(test_images), batch_size):
            batch = ts.prepare_batch(test_images[start:start + batch_size], mean, std).to(device)
            logits = model(batch)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)

    mean_iou, per_class_iou = ts.compute_mean_iou(preds, test_masks, n_classes)
    foreground_iou = per_class_iou[1] if n_classes == 2 and len(per_class_iou) > 1 else None
    return {
        "mean_iou": mean_iou,
        "per_class_iou": per_class_iou,
        "foreground_iou": foreground_iou,
        "n_test_samples": int(len(test_masks)),
        "band_names": band_names,
    }
