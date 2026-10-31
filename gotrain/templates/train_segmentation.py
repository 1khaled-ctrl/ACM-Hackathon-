"""Real segmentation training template for GEO-Bench segmentation tasks
(m-pv4ger-seg, m-nz-cattle). See gotrain/templates/INTERFACE.md for the
full CLI/stdout/run-directory contract this follows.

Trains on the TRAIN split, evaluates on VALIDATION every epoch, and never
opens the TEST split. The encoder (segmentation_models_pytorch) is built
with in_channels set to the dataset's actual band count, so it natively
adapts its first conv layer and (if pretrained) channel-expands the
imagenet weights — no manual surgery needed. Normalization statistics are
computed from the TRAIN split only, never from validation or test.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

DEFAULTS = {
    "encoder": "resnet34",
    "pretrained": True,
    "architecture": "unet",
    "lr": 1e-3,
    "batch_size": 8,
    "epochs": 2,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "loss": "ce",
    "class_weights": None,
    "augmentation": "none",
    "bands": None,
    "time_budget_minutes": None,
    "seed": 0,
    "max_train_samples": None,
    "max_val_samples": None,
}

BENCHMARK_DIRS = ["classification_v1.0", "segmentation_v1.0"]


def find_task_dir(dataset: str):
    import geobench

    geo_bench_dir = Path(os.environ.get("GEO_BENCH_DIR", str(geobench.GEO_BENCH_DIR)))
    for benchmark_dir in BENCHMARK_DIRS:
        candidate = geo_bench_dir / benchmark_dir / dataset
        if candidate.is_dir() and (candidate / "task_specs.pkl").exists():
            return candidate
    raise ValueError(f"Dataset {dataset!r} not found under {geo_bench_dir}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset_arrays(task_specs, split: str, band_names, max_samples, logger):
    """Load every sample of `split` into memory as (N, H, W, C) uint8 images
    and (N, H, W) uint8 masks. GEO-Bench segmentation tasks here are small
    enough (a few thousand 320x320x3 or 500x500x3 samples) that this is
    simpler and fast enough for a hackathon training loop than writing a
    lazy on-disk Dataset; it also makes computing TRAIN-only normalization
    stats a single pass instead of two.
    """
    geobench_split = {"train": "train", "validation": "valid"}[split]
    ds = task_specs.get_dataset(split=geobench_split)

    n = len(ds)
    if max_samples is not None:
        n = min(n, max_samples)

    images = []
    masks = []
    for i in range(n):
        sample = ds[i]
        arr, _ = sample.pack_to_3d(band_names=band_names)
        images.append(arr)
        masks.append(sample.label.data)

    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)
    logger.info(f"Loaded {split} split: {images.shape[0]} samples, images {images.shape}, masks {masks.shape}")
    return images, masks


def compute_train_band_stats(train_images: np.ndarray, logger):
    """Per-band mean/std computed from TRAIN images only. train_images is
    (N, H, W, C) uint8; stats are returned as float64 arrays of length C.
    """
    flat = train_images.reshape(-1, train_images.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0  # guard against a constant band
    logger.info(f"TRAIN band stats — mean: {mean.tolist()}, std: {std.tolist()}")
    return mean, std


def compute_class_pixel_frequencies(masks: np.ndarray, n_classes: int, logger):
    """Pixel frequency of each class across the TRAIN split, logged for
    visibility into class imbalance and returned for optional use as
    inverse-frequency class weights.
    """
    counts = np.bincount(masks.reshape(-1), minlength=n_classes).astype(np.float64)
    freqs = counts / counts.sum()
    for c, f in enumerate(freqs):
        logger.info(f"TRAIN class {c} pixel frequency: {f:.6f} ({int(counts[c])} pixels)")
    return freqs


class SegmentationArrayDataset:
    """Thin torch Dataset over in-memory (N,H,W,C) images / (N,H,W) masks."""

    def __init__(self, images, masks, mean, std, augment: str, seed: int):
        self.images = images
        self.masks = masks
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.images)

    def _augment(self, image, mask):
        if self.augment == "none":
            return image, mask
        if self.rng.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if self.augment == "strong":
            if self.rng.random() < 0.5:
                image = np.ascontiguousarray(image[::-1, :, :])
                mask = np.ascontiguousarray(mask[::-1, :])
            k = self.rng.choice([0, 1, 2, 3])
            if k:
                image = np.ascontiguousarray(np.rot90(image, k, axes=(0, 1)))
                mask = np.ascontiguousarray(np.rot90(mask, k, axes=(0, 1)))
        return image, mask

    def __getitem__(self, idx):
        import torch

        image = self.images[idx].astype(np.float32)
        mask = self.masks[idx].astype(np.int64)
        image, mask = self._augment(image, mask)

        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        return torch.from_numpy(image.copy()).float(), torch.from_numpy(mask.copy()).long()


def build_model(architecture: str, encoder: str, pretrained: bool, in_channels: int, n_classes: int):
    import segmentation_models_pytorch as smp

    encoder_weights = "imagenet" if pretrained else None
    kwargs = dict(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=n_classes,
    )
    if architecture == "unet":
        return smp.Unet(**kwargs)
    if architecture == "fpn":
        return smp.FPN(**kwargs)
    raise ValueError(f"architecture must be 'unet' or 'fpn', got {architecture!r}")


def build_loss(loss_name: str, n_classes: int, class_weights, device):
    import torch
    import segmentation_models_pytorch as smp

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    ce = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    if loss_name == "ce":
        return ce
    if loss_name == "dice":
        return dice
    if loss_name == "ce+dice":
        def combined(logits, target):
            return ce(logits, target) + dice(logits, target)
        return combined
    raise ValueError(f"loss must be 'ce', 'dice', or 'ce+dice', got {loss_name!r}")


def build_optimizer(name: str, params, lr: float, weight_decay: float):
    import torch

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"optimizer must be 'adam' or 'sgd', got {name!r}")


def compute_mean_iou(pred: "np.ndarray", target: "np.ndarray", n_classes: int):
    """pred/target are (N,H,W) int arrays. Returns (mean_iou, per_class_iou list).
    A class absent from both pred and target in the whole batch contributes
    no term (avoids 0/0 dragging the mean down for classes that never occur).
    """
    per_class = []
    for c in range(n_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        if union == 0:
            continue
        per_class.append(intersection / union)
    mean_iou = float(np.mean(per_class)) if per_class else 0.0
    # pad per_class_iou back out to n_classes with None for classes that never occurred
    per_class_full = []
    idx = 0
    for c in range(n_classes):
        pred_c = pred == c
        target_c = target == c
        if (pred_c | target_c).sum() == 0:
            per_class_full.append(None)
        else:
            per_class_full.append(float(per_class[idx]))
            idx += 1
    return mean_iou, per_class_full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True, help="JSON-encoded config dict")
    args = parser.parse_args()

    user_config = json.loads(args.config)
    cfg = {**DEFAULTS, **user_config}
    dataset = args.dataset

    run_id = str(uuid.uuid4())
    run_dir = REPO_ROOT / "runs" / dataset / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    logger = logging.getLogger(f"train_segmentation.{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stderr_handler)

    logger.info(f"Starting segmentation training run {run_id} for dataset={dataset!r}")
    logger.info(f"Config: {json.dumps(cfg)}")

    set_seed(cfg["seed"])

    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    task_dir = find_task_dir(dataset)
    import geobench

    task_specs = geobench.load_task_specs(task_dir)
    all_band_names = [b.name for b in task_specs.bands_info]
    band_names = cfg["bands"] if cfg["bands"] else all_band_names
    n_bands = len(band_names)
    n_classes = task_specs.label_type.n_classes
    logger.info(f"Bands ({n_bands}/{len(all_band_names)} used): {band_names}")
    logger.info(f"Image size: {task_specs.patch_size}, classes: {n_classes} ({task_specs.label_type.class_names})")

    train_images, train_masks = build_dataset_arrays(
        task_specs, "train", band_names, cfg["max_train_samples"], logger
    )
    val_images, val_masks = build_dataset_arrays(
        task_specs, "validation", band_names, cfg["max_val_samples"], logger
    )

    mean, std = compute_train_band_stats(train_images, logger)
    class_freqs = compute_class_pixel_frequencies(train_masks, n_classes, logger)

    class_weights = cfg["class_weights"]
    if class_weights is None and cfg["loss"] in ("ce", "ce+dice"):
        pass  # default: no reweighting unless explicitly requested in config
    logger.info(f"Class weights used: {class_weights}")

    train_dataset = SegmentationArrayDataset(
        train_images, train_masks, mean, std, cfg["augmentation"], cfg["seed"]
    )
    val_dataset = SegmentationArrayDataset(val_images, val_masks, mean, std, "none", cfg["seed"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    try:
        model = build_model(cfg["architecture"], cfg["encoder"], cfg["pretrained"], n_bands, n_classes)
        model = model.to(device)
    except torch.OutOfMemoryError:
        logger.error("Out of memory while building/moving the model to device.")
        print("OUT OF MEMORY while constructing the model. Try a smaller --batch_size or a lighter encoder.", file=sys.stderr)
        sys.exit(1)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {cfg['architecture']}/{cfg['encoder']} (pretrained={cfg['pretrained']}), {n_params:,} parameters")

    loss_fn = build_loss(cfg["loss"], n_classes, class_weights, device)
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["weight_decay"])

    best_val_iou = -1.0
    best_metrics = None
    best_checkpoint_path = run_dir / "best_model.pt"
    time_budget_seconds = cfg["time_budget_minutes"] * 60 if cfg["time_budget_minutes"] else None
    training_start = time.time()

    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        try:
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = loss_fn(logits, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1
        except torch.OutOfMemoryError:
            logger.error(f"Out of memory during training at epoch {epoch}.")
            print(
                f"OUT OF MEMORY during training (epoch {epoch}, batch_size={cfg['batch_size']}). "
                "Try a smaller --batch_size.",
                file=sys.stderr,
            )
            sys.exit(1)

        avg_train_loss = running_loss / max(n_batches, 1)

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                try:
                    logits = model(images)
                except torch.OutOfMemoryError:
                    logger.error("Out of memory during validation.")
                    print(
                        f"OUT OF MEMORY during validation (batch_size={cfg['batch_size']}). "
                        "Try a smaller --batch_size.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(masks.numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        mean_iou, per_class_iou = compute_mean_iou(all_preds, all_targets, n_classes)

        # For binary tasks, report the foreground (class 1) IoU explicitly.
        foreground_iou = per_class_iou[1] if n_classes == 2 and len(per_class_iou) > 1 else None

        epoch_seconds = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch + 1}/{cfg['epochs']} — train_loss={avg_train_loss:.4f} "
            f"val_mean_iou={mean_iou:.4f} per_class_iou={per_class_iou} "
            f"foreground_iou={foreground_iou} time={epoch_seconds:.1f}s (split=validation)"
        )

        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            best_metrics = {
                "epoch": epoch + 1,
                "val_mean_iou": mean_iou,
                "val_per_class_iou": per_class_iou,
                "val_foreground_iou": foreground_iou,
                "train_loss": avg_train_loss,
            }
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info(f"New best val_mean_iou={mean_iou:.4f} — checkpoint saved to {best_checkpoint_path}")

        elapsed = time.time() - training_start
        if time_budget_seconds is not None and elapsed > time_budget_seconds:
            logger.info(f"Time budget of {cfg['time_budget_minutes']} minutes reached — stopping early.")
            break

    total_seconds = time.time() - training_start
    gpu_minutes = total_seconds / 60 if device.type == "cuda" else 0.0

    final_val_metric = best_val_iou
    metrics = {
        "run_id": run_id,
        "dataset": dataset,
        "device": str(device),
        "n_params": n_params,
        "n_bands": n_bands,
        "band_names": band_names,
        "n_classes": n_classes,
        "class_pixel_frequencies": class_freqs.tolist(),
        "best": best_metrics,
        "final_val_metric": final_val_metric,
        "split_used": "validation",
        "total_train_seconds": total_seconds,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(f"Run finished in {total_seconds:.1f}s. Best val_mean_iou={best_val_iou:.4f}")
    logger.info(f"FINAL_VAL_METRIC: {final_val_metric}")

    print(f"FINAL_VAL_METRIC: {final_val_metric}")
    print(json.dumps({
        "val_metric": final_val_metric,
        "gpu_minutes": gpu_minutes,
        "rationale": f"best val_mean_iou={final_val_metric:.4f} at epoch {best_metrics['epoch']} (run_dir={run_dir})",
    }))


if __name__ == "__main__":
    main()
