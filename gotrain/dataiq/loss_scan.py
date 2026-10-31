"""Model-assisted per-example training-loss analysis (DataIQ) -- the
"suspicious label" signal from the design brief's sections 3.3/3.4.

Deliberately NOT computed during training's hot loop (that would need
invasive per-batch bookkeeping across every epoch, slowing training for a
signal that's only looked at occasionally). Instead, this loads a run's
ALREADY-SAVED best checkpoint and does ONE extra forward pass over the
TRAIN split in eval mode (no gradients), recording each sample's loss/
predicted-class/confidence under that checkpoint -- an efficient, honest
approximation of "how well does the model that actually got shipped
explain this training example," not an exact per-example influence score
and not proof of a bad label. A genuinely hard (but correctly labeled)
example can look identical to a mislabeled one by this metric alone --
every flagged sample becomes a PENDING data-quality proposal
(gotrain.dataiq.decision), never auto-approved, for exactly this reason.

NEEDS TORCH (and geobench + the real downloaded dataset) to run -- not
executable in this project's local/CI environment. Mirrors
gotrain/tools/final_eval.py's proven pattern for rebuilding a run's model
+ data from its saved config (same band/class resolution, same TRAIN-only
normalization stats recomputation) and reuses the exact
build_model/build_dataset_arrays functions the training templates and
final_eval.py already use -- no separate, divergent loading path.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from gotrain import ledger
from gotrain.dataiq import decision as dataiq_decision
from gotrain.logging_utils import LOGS_DIR, log_event

SAMPLE_LOSSES_DIR = LOGS_DIR / "dataiq" / "sample_losses"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "runs"

# Deliberately conservative thresholds -- see the module docstring on why
# this is a signal, not proof. HIGH_LOSS_PERCENTILE flags only the
# genuine tail of the per-sample loss distribution for THIS run, not an
# absolute loss value (which has no fixed meaning across configs).
HIGH_LOSS_PERCENTILE = 95
HIGH_CONFIDENCE_WRONG_THRESHOLD = 0.9


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
    record = next((r for r in ledger.get_history(dataset) if r["run_id"] == run_id), None)
    if record is None:
        raise ValueError(f"run_id {run_id!r} not found for dataset {dataset!r}")
    return record


def _record_and_flag(dataset: str, run_id: str, task_type: str, rows: list) -> dict:
    """Shared tail end for both scan functions below: write the full
    per-sample report to disk and raise a PENDING DataIQ proposal for
    every flagged row.
    """
    flagged = [r for r in rows if r["possible_issue"] is not None]

    report_dir = SAMPLE_LOSSES_DIR / dataset
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": dataset, "run_id": run_id, "task_type": task_type,
        "n_samples": len(rows), "n_flagged": len(flagged),
    }
    (report_dir / f"{run_id}.json").write_text(
        json.dumps({"summary": summary, "samples": rows}, indent=2), encoding="utf-8"
    )

    for row in flagged:
        action = (
            "CORRECT_LABEL_CANDIDATE" if row["possible_issue"] == "high_confidence_wrong_prediction"
            else "EXCLUDE_FROM_TRAINING_CANDIDATE"
        )
        evidence = {k: v for k, v in row.items() if k not in ("sample_id", "possible_issue")}
        dataiq_decision.propose_action(
            dataset=dataset, sample_id=row["sample_id"], action=action,
            evidence=evidence, severity="medium", confidence=row.get("confidence"),
            reason=f"possible_issue={row['possible_issue']!r} from run {run_id} (model-assisted signal, not proof)",
        )

    log_event("dataiq_loss_scan_finished", dataset=dataset, run_id=run_id, task_type=task_type, **summary)
    return summary


def scan_classification_run(dataset: str, run_id: str, max_samples: Optional[int] = None) -> dict:
    import torch
    import torch.nn.functional as F
    import geobench

    from gotrain.templates import train_classification as tc

    run_record = _get_run_record(dataset, run_id)
    config = run_record["config"]
    run_dir = _run_dir(dataset, run_id)

    import logging

    logger = logging.getLogger(f"dataiq.loss_scan.{run_id}")
    logger.addHandler(logging.NullHandler())

    task_dir = tc.find_task_dir(dataset)
    task_specs = geobench.load_task_specs(task_dir)
    all_band_names = [b.name for b in task_specs.bands_info]
    band_names = config.get("bands") or all_band_names
    n_bands = len(band_names)
    n_classes = task_specs.label_type.n_classes

    train_images, train_labels = tc.build_dataset_arrays(task_specs, "train", band_names, max_samples, logger)
    mean, std = tc.compute_train_band_stats(train_images, logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tc.build_model(config.get("backbone", "resnet18"), False, n_bands, n_classes)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model = model.to(device)
    model.eval()

    normalized = (train_images - mean.astype(np.float32)) / std.astype(np.float32)
    normalized = np.transpose(normalized, (0, 3, 1, 2))

    batch_size = 32
    all_losses, all_preds, all_confidences = [], [], []
    with torch.no_grad():
        for start in range(0, len(normalized), batch_size):
            batch = torch.from_numpy(normalized[start:start + batch_size]).float().to(device)
            targets = torch.from_numpy(train_labels[start:start + batch_size]).long().to(device)
            logits = model(batch)
            losses = F.cross_entropy(logits, targets, reduction="none")
            probs = torch.softmax(logits, dim=1)
            confidences, preds = probs.max(dim=1)
            all_losses.append(losses.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_confidences.append(confidences.cpu().numpy())

    losses = np.concatenate(all_losses)
    preds = np.concatenate(all_preds)
    confidences = np.concatenate(all_confidences)
    high_loss_threshold = float(np.percentile(losses, HIGH_LOSS_PERCENTILE))

    rows = []
    for i in range(len(losses)):
        is_wrong = int(preds[i]) != int(train_labels[i])
        is_high_confidence_wrong = is_wrong and float(confidences[i]) >= HIGH_CONFIDENCE_WRONG_THRESHOLD
        is_high_loss = float(losses[i]) >= high_loss_threshold

        possible_issue = None
        if is_high_confidence_wrong:
            possible_issue = "high_confidence_wrong_prediction"
        elif is_high_loss and is_wrong:
            possible_issue = "high_loss_hard_or_noisy"

        rows.append({
            "sample_id": i,
            "assigned_label": int(train_labels[i]),
            "predicted_class": int(preds[i]),
            "confidence": float(confidences[i]),
            "loss": float(losses[i]),
            "possible_issue": possible_issue,
        })

    return _record_and_flag(dataset, run_id, "classification", rows)


def scan_segmentation_run(dataset: str, run_id: str, max_samples: Optional[int] = None) -> dict:
    """Per-IMAGE summary (mean pixel loss + foreground IoU), per the
    design brief's "for segmentation, use per-image summaries of
    pixel-level losses" -- a per-pixel report would be far too large
    and too fine-grained to act on via engineer review.
    """
    import torch
    import torch.nn.functional as F
    import geobench

    from gotrain.templates import train_segmentation as ts

    run_record = _get_run_record(dataset, run_id)
    config = run_record["config"]
    run_dir = _run_dir(dataset, run_id)

    import logging

    logger = logging.getLogger(f"dataiq.loss_scan.{run_id}")
    logger.addHandler(logging.NullHandler())

    task_dir = ts.find_task_dir(dataset)
    task_specs = geobench.load_task_specs(task_dir)
    all_band_names = [b.name for b in task_specs.bands_info]
    band_names = config.get("bands") or all_band_names
    n_bands = len(band_names)
    n_classes = task_specs.label_type.n_classes

    train_images, train_masks = ts.build_dataset_arrays(task_specs, "train", band_names, max_samples, logger)
    mean, std = ts.compute_train_band_stats(train_images, logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ts.build_model(
        config.get("architecture", "unet"), config.get("encoder", "resnet34"), False, n_bands, n_classes
    )
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model = model.to(device)
    model.eval()

    normalized = (train_images.astype(np.float64) - mean) / std
    normalized = np.transpose(normalized, (0, 3, 1, 2)).astype(np.float32)

    batch_size = 8
    all_mean_losses, all_fg_ious = [], []
    with torch.no_grad():
        for start in range(0, len(normalized), batch_size):
            batch = torch.from_numpy(normalized[start:start + batch_size]).float().to(device)
            targets = torch.from_numpy(train_masks[start:start + batch_size]).long().to(device)
            logits = model(batch)
            pixel_losses = F.cross_entropy(logits, targets, reduction="none")  # (B, H, W)
            all_mean_losses.append(pixel_losses.mean(dim=(1, 2)).cpu().numpy())

            preds = logits.argmax(dim=1).cpu().numpy()
            target_np = targets.cpu().numpy()
            for b in range(preds.shape[0]):
                pred_fg, target_fg = preds[b] == 1, target_np[b] == 1
                union = np.logical_or(pred_fg, target_fg).sum()
                fg_iou = float(np.logical_and(pred_fg, target_fg).sum() / union) if union > 0 else None
                all_fg_ious.append(fg_iou)

    mean_losses = np.concatenate(all_mean_losses)
    high_loss_threshold = float(np.percentile(mean_losses, HIGH_LOSS_PERCENTILE))

    rows = []
    for i in range(len(mean_losses)):
        fg_iou = all_fg_ious[i]
        is_high_loss = float(mean_losses[i]) >= high_loss_threshold
        is_low_fg_iou = fg_iou is not None and fg_iou < 0.1

        possible_issue = None
        if is_high_loss and is_low_fg_iou:
            possible_issue = "high_loss_low_foreground_iou_hard_or_noisy"

        rows.append({
            "sample_id": i,
            "mean_pixel_loss": float(mean_losses[i]),
            "foreground_iou": fg_iou,
            "possible_issue": possible_issue,
            "confidence": None,
        })

    return _record_and_flag(dataset, run_id, "segmentation", rows)
