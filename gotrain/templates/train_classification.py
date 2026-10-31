"""Real classification training template for GEO-Bench classification tasks
(m-eurosat, m-so2sat). See gotrain/templates/INTERFACE.md for the full
CLI/stdout/run-directory contract this follows.

Trains on the TRAIN split, evaluates on VALIDATION every epoch, and never
opens the TEST split. The backbone (timm) is built with in_chans set to the
dataset's actual band count (13 for m-eurosat, 18 for m-so2sat — SAR +
optical), so it natively adapts its first conv layer and (if pretrained)
channel-expands the imagenet weights, mirroring how train_segmentation.py
handles multi-band input via segmentation_models_pytorch's in_channels.
Normalization statistics are computed from the TRAIN split only, never
from validation or test.

Per gotrain/templates/INTERFACE.md's stdout contract: everything meant for
a human (progress, FINAL_VAL_METRIC, warnings) goes to train.log/stderr via
the logger — never to stdout before the final JSON line, so that a human
piping stdout only ever sees that one line.
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
if str(REPO_ROOT) not in sys.path:
    # Needed to `import gotrain.status` below -- this script runs as a
    # subprocess (sys.executable + this file's path, not `-m`), so Python
    # only puts this file's own directory on sys.path by default.
    sys.path.insert(0, str(REPO_ROOT))


def report_progress(**fields) -> None:
    """Best-effort live per-epoch status update for the dashboard. Never
    allowed to fail a training run -- if gotrain.status can't be imported
    or the write fails for any reason, training continues unaffected.
    """
    try:
        from gotrain.status import set_current_run

        set_current_run(fields)
    except Exception:
        pass


DEFAULTS = {
    "backbone": "resnet18",
    "pretrained": True,
    "lr": 1e-3,
    "batch_size": 32,
    "epochs": 2,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "scheduler": "none",
    "warmup_epochs": 0,
    "label_smoothing": 0.0,
    "image_size": None,
    "class_weights": None,
    "augmentation": "none",
    "bands": None,
    "amp": True,
    "time_budget_minutes": None,
    "seed": 0,
    "max_train_samples": None,
    "max_val_samples": None,
    "tta": False,
}

BENCHMARK_DIRS = ["classification_v1.0", "segmentation_v1.0"]


def is_oom_error(exc: BaseException) -> bool:
    """True for a CUDA/MPS OutOfMemoryError, and also for plain CPU
    allocation failures — torch raises a generic RuntimeError for those
    (e.g. "DefaultCPUAllocator: not enough memory"), not OutOfMemoryError,
    so catching only torch.OutOfMemoryError misses CPU-only OOMs entirely.
    """
    import torch

    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "not enough memory" in message


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
    """Load every sample of `split` into memory as (N, H, W, C) float32 images
    and (N,) int64 labels. GEO-Bench classification tasks here are small
    enough (a few thousand small patches) that this is simpler than writing
    a lazy on-disk Dataset, and makes computing TRAIN-only normalization
    stats a single pass instead of two.
    """
    geobench_split = {"train": "train", "validation": "valid"}[split]
    ds = task_specs.get_dataset(split=geobench_split)

    n = len(ds)
    if max_samples is not None:
        n = min(n, max_samples)

    images = []
    labels = []
    for i in range(n):
        sample = ds[i]
        arr, _ = sample.pack_to_3d(band_names=band_names)
        images.append(arr.astype(np.float32))
        labels.append(int(sample.label))

    images = np.stack(images, axis=0)
    labels = np.array(labels, dtype=np.int64)
    logger.info(f"Loaded {split} split: {images.shape[0]} samples, images {images.shape}, labels {labels.shape}")
    return images, labels


def compute_train_band_stats(train_images: np.ndarray, logger):
    """Per-band mean/std computed from TRAIN images only. train_images is
    (N, H, W, C) float32; stats are returned as float64 arrays of length C.
    """
    flat = train_images.reshape(-1, train_images.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0  # guard against a constant band
    logger.info(f"TRAIN band stats — mean: {mean.tolist()}, std: {std.tolist()}")
    return mean, std


def compute_class_frequencies(labels: np.ndarray, n_classes: int, logger):
    """Class frequency across the TRAIN split, logged for visibility into
    class imbalance and returned for optional use as inverse-frequency
    class weights.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    freqs = counts / counts.sum()
    for c, f in enumerate(freqs):
        logger.info(f"TRAIN class {c} frequency: {f:.6f} ({int(counts[c])} samples)")
    return freqs


class ClassificationArrayDataset:
    """Thin torch Dataset over in-memory (N,H,W,C) images / (N,) labels."""

    def __init__(self, images, labels, mean, std, augment: str, seed: int, image_size=None):
        self.images = images
        self.labels = labels
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.augment = augment
        self.rng = random.Random(seed)
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def _augment(self, image):
        if self.augment == "none":
            return image
        if self.rng.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
        if self.augment == "strong":
            if self.rng.random() < 0.5:
                image = np.ascontiguousarray(image[::-1, :, :])
            k = self.rng.choice([0, 1, 2, 3])
            if k:
                image = np.ascontiguousarray(np.rot90(image, k, axes=(0, 1)))
            # Light Gaussian noise, scaled to each band's own std so it
            # stays proportionate whether the band is int16 reflectance
            # (m-eurosat) or already-small float32 SAR/optical (m-so2sat).
            # Drawn from self.rng (not the numpy global RNG) so augmentation
            # stays reproducible for a given seed.
            noise = np.array(
                [self.rng.gauss(0.0, 0.02) for _ in range(image.shape[-1])], dtype=np.float32
            )
            image = image + noise * self.std
        return image

    def __getitem__(self, idx):
        import torch
        import torch.nn.functional as F

        image = self.images[idx]
        label = self.labels[idx]
        image = self._augment(image)

        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        tensor = torch.from_numpy(image.copy()).float()

        if self.image_size is not None:
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        return tensor, torch.tensor(label, dtype=torch.long)


def build_model(backbone: str, pretrained: bool, in_chans: int, n_classes: int):
    import timm

    return timm.create_model(backbone, pretrained=pretrained, in_chans=in_chans, num_classes=n_classes)


def build_loss(n_classes: int, class_weights, label_smoothing: float, device):
    import torch

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return torch.nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)


def build_optimizer(name: str, params, lr: float, weight_decay: float):
    import torch

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"optimizer must be 'adam', 'adamw', or 'sgd', got {name!r}")


def build_scheduler(name: str, optimizer, epochs: int, warmup_epochs: int):
    """Returns None for "none", else a per-epoch scheduler (call .step()
    once per epoch, after the epoch's training+validation).

    "cosine" ramps the LR linearly from 10% to 100% over `warmup_epochs`,
    then cosine-decays to ~0 over the remaining epochs. warmup_epochs=0
    skips straight to cosine decay from epoch 0.
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    if name == "none":
        return None
    if name != "cosine":
        raise ValueError(f"scheduler must be 'none' or 'cosine', got {name!r}")

    warmup_epochs = min(warmup_epochs, max(epochs - 1, 0))
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def predict_with_tta(model, images, device, amp_enabled, autocast):
    """Average softmax probabilities over the identity image plus 3 flip/
    rotation variants, then undo nothing (predictions are per-image class
    scores, orientation-invariant by construction) and return the averaged
    probabilities. Only used at validation (and only when cfg["tta"] is
    true) — never changes what the model is trained on.
    """
    import torch

    variants = [
        images,
        torch.flip(images, dims=[-1]),
        torch.rot90(images, k=1, dims=[-2, -1]),
        torch.rot90(images, k=2, dims=[-2, -1]),
    ]
    probs_sum = None
    for variant in variants:
        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(variant)
        probs = torch.softmax(logits.float(), dim=1)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / len(variants)


def compute_accuracy_metrics(pred: "np.ndarray", target: "np.ndarray", n_classes: int):
    """pred/target are (N,) int arrays. Returns (overall_accuracy, per_class_accuracy
    list). A class absent from the target in this split gets None (no
    denominator to compute recall-style per-class accuracy against).
    """
    overall = float((pred == target).mean())

    per_class = []
    for c in range(n_classes):
        mask = target == c
        if mask.sum() == 0:
            per_class.append(None)
        else:
            per_class.append(float((pred[mask] == target[mask]).mean()))
    return overall, per_class


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

    logger = logging.getLogger(f"train_classification.{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stderr_handler)

    logger.info(f"Starting classification training run {run_id} for dataset={dataset!r}")
    logger.info(f"Config: {json.dumps(cfg)}")

    set_seed(cfg["seed"])

    import torch
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg["amp"]) and device.type == "cuda"
    if cfg["amp"] and device.type != "cuda":
        logger.info("AMP requested but no CUDA device is available — running in full precision.")
    logger.info(f"Device: {device}, AMP enabled: {amp_enabled}")

    task_dir = find_task_dir(dataset)
    import geobench

    task_specs = geobench.load_task_specs(task_dir)
    all_band_names = [b.name for b in task_specs.bands_info]
    band_names = cfg["bands"] if cfg["bands"] else all_band_names
    n_bands = len(band_names)
    n_classes = task_specs.label_type.n_classes
    logger.info(f"Bands ({n_bands}/{len(all_band_names)} used): {band_names}")
    logger.info(f"Image size: {task_specs.patch_size}, classes: {n_classes} ({task_specs.label_type.class_names})")

    if dataset == "m-eurosat" and cfg["bands"]:
        logger.warning(
            "m-eurosat: geo-bench-1.0 stores incorrect band names/wavelengths for most "
            "channels (pixel data is unaffected, but selecting by name returns the wrong "
            "channel — see https://github.com/ServiceNow/geo-bench/issues/28). "
            "A non-default `bands` config was given for m-eurosat; consider using all "
            "bands (bands=null) instead."
        )

    train_images, train_labels = build_dataset_arrays(
        task_specs, "train", band_names, cfg["max_train_samples"], logger
    )
    val_images, val_labels = build_dataset_arrays(
        task_specs, "validation", band_names, cfg["max_val_samples"], logger
    )

    mean, std = compute_train_band_stats(train_images, logger)
    class_freqs = compute_class_frequencies(train_labels, n_classes, logger)

    class_weights = cfg["class_weights"]
    logger.info(f"Class weights used: {class_weights}")

    train_dataset = ClassificationArrayDataset(
        train_images, train_labels, mean, std, cfg["augmentation"], cfg["seed"], cfg["image_size"]
    )
    val_dataset = ClassificationArrayDataset(
        val_images, val_labels, mean, std, "none", cfg["seed"], cfg["image_size"]
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    try:
        model = build_model(cfg["backbone"], cfg["pretrained"], n_bands, n_classes)
        model = model.to(device)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if not is_oom_error(e):
            raise
        logger.error("Out of memory while building/moving the model to device.")
        print(
            "OUT OF MEMORY while constructing the model. Try a smaller --batch_size or a lighter backbone.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {cfg['backbone']} (pretrained={cfg['pretrained']}), {n_params:,} parameters")

    loss_fn = build_loss(n_classes, class_weights, cfg["label_smoothing"], device)
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["weight_decay"])
    scheduler = build_scheduler(cfg["scheduler"], optimizer, cfg["epochs"], cfg["warmup_epochs"])
    scaler = GradScaler(device="cuda", enabled=amp_enabled)

    best_val_acc = -1.0
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
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                n_batches += 1
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if not is_oom_error(e):
                raise
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
            for images, labels in val_loader:
                images = images.to(device)
                try:
                    if cfg["tta"]:
                        probs = predict_with_tta(model, images, device, amp_enabled, autocast)
                        preds = probs.argmax(dim=1).cpu().numpy()
                    else:
                        with autocast(device_type=device.type, enabled=amp_enabled):
                            logits = model(images)
                        preds = logits.argmax(dim=1).cpu().numpy()
                except (torch.OutOfMemoryError, RuntimeError) as e:
                    if not is_oom_error(e):
                        raise
                    logger.error("Out of memory during validation.")
                    print(
                        f"OUT OF MEMORY during validation (batch_size={cfg['batch_size']}). "
                        "Try a smaller --batch_size.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                all_preds.append(preds)
                all_targets.append(labels.numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        accuracy, per_class_accuracy = compute_accuracy_metrics(all_preds, all_targets, n_classes)

        if scheduler is not None:
            scheduler.step()

        epoch_seconds = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch + 1}/{cfg['epochs']} — train_loss={avg_train_loss:.4f} "
            f"val_accuracy={accuracy:.4f} per_class_accuracy={per_class_accuracy} "
            f"time={epoch_seconds:.1f}s (split=validation)"
        )

        report_progress(
            run_id=run_id,
            dataset=dataset,
            task_type="classification",
            model=cfg["backbone"],
            optimizer=cfg["optimizer"],
            lr=cfg["lr"],
            scheduler=cfg["scheduler"],
            epoch=epoch + 1,
            epochs=cfg["epochs"],
            train_loss=avg_train_loss,
            val_metric=accuracy,
            metric_name="val_accuracy",
            best_val_metric=max(best_val_acc, accuracy),
            elapsed_seconds=time.time() - training_start,
        )

        if accuracy > best_val_acc:
            best_val_acc = accuracy
            best_metrics = {
                "epoch": epoch + 1,
                "val_accuracy": accuracy,
                "val_per_class_accuracy": per_class_accuracy,
                "train_loss": avg_train_loss,
            }
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info(f"New best val_accuracy={accuracy:.4f} — checkpoint saved to {best_checkpoint_path}")

        elapsed = time.time() - training_start
        if time_budget_seconds is not None and elapsed > time_budget_seconds:
            logger.info(f"Time budget of {cfg['time_budget_minutes']} minutes reached — stopping early.")
            break

    total_seconds = time.time() - training_start
    gpu_minutes = total_seconds / 60 if device.type == "cuda" else 0.0

    final_val_metric = best_val_acc
    metrics = {
        "run_id": run_id,
        "dataset": dataset,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "n_params": n_params,
        "n_bands": n_bands,
        "band_names": band_names,
        "n_classes": n_classes,
        "class_frequencies": class_freqs.tolist(),
        "best": best_metrics,
        "final_val_metric": final_val_metric,
        "split_used": "validation",
        "total_train_seconds": total_seconds,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(f"Run finished in {total_seconds:.1f}s. Best val_accuracy={best_val_acc:.4f}")
    logger.info(f"FINAL_VAL_METRIC: {final_val_metric}")

    print(json.dumps({
        "val_metric": final_val_metric,
        "gpu_minutes": gpu_minutes,
        "rationale": f"best val_accuracy={final_val_metric:.4f} at epoch {best_metrics['epoch']} (run_dir={run_dir})",
    }))


if __name__ == "__main__":
    main()
