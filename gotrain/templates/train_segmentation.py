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
if str(REPO_ROOT) not in sys.path:
    # Needed to `import gotrain.status` below -- this script runs as a
    # subprocess (sys.executable + this file's path, not `-m`), so Python
    # only puts this file's own directory on sys.path by default.
    sys.path.insert(0, str(REPO_ROOT))

from gotrain import reproducibility  # noqa: E402 -- needs sys.path fix above


def report_progress(**fields) -> None:
    """Best-effort live per-epoch status update for the dashboard. Never
    allowed to fail a training run -- if gotrain.status can't be imported
    or the write fails for any reason, training continues unaffected.
    Mirrors gotrain/templates/train_classification.py's report_progress.
    """
    try:
        from gotrain.status import set_current_run

        set_current_run(fields)
    except Exception:
        pass
    try:
        from gotrain.logging_utils import log_live_metric

        log_live_metric(
            run_id=fields.get("run_id"),
            dataset=fields.get("dataset"),
            epoch=fields.get("epoch"),
            train_loss=fields.get("train_loss"),
            val_metric=fields.get("val_metric"),
            elapsed_seconds=fields.get("elapsed_seconds"),
            task_type=fields.get("task_type"),
            metric_name=fields.get("metric_name"),
        )
    except Exception:
        pass


DEFAULTS = {
    "encoder": "resnet34",
    "pretrained": True,
    "architecture": "unet",
    "lr": 1e-3,
    "batch_size": 8,
    "epochs": 2,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "scheduler": "none",
    "warmup_epochs": 0,
    "loss": "ce",
    "focal_alpha": None,
    "focal_gamma": 2.0,
    "class_weights": None,
    "augmentation": "none",
    "bands": None,
    "amp": True,
    "time_budget_minutes": None,
    "seed": 0,
    "max_train_samples": None,
    "max_val_samples": None,
    "fine_tuning_strategy": "full",
    "manifest_id": None,
    # Set by gotrain.tools.training.launch_training from the agent's actual
    # per-run subprocess timeout -- see train_classification.py's identical
    # field for the full rationale. Not a user-facing knob (no
    # config_schema entry).
    "_timeout_seconds": None,
}

BENCHMARK_DIRS = ["classification_v1.0", "segmentation_v1.0"]

# Loading/time-budget helpers shared with train_classification.py; see
# gotrain/data_loading.py for why each exists.
from gotrain import data_loading  # noqa: E402
from gotrain.data_loading import (  # noqa: E402,F401 -- re-exported for callers/tests
    LOADING_DEADLINE_CHECK_EVERY,
    LOADING_PHASE_BUDGET_FRACTION,
    TRAINING_CUTOFF_FRACTION,
    DataLoadingBudgetExceeded,
)


def is_oom_error(exc: BaseException) -> bool:
    """True for a CUDA/MPS OutOfMemoryError, and also for plain CPU
    allocation failures — torch raises a generic RuntimeError for those
    (e.g. "DefaultCPUAllocator: not enough memory"), not OutOfMemoryError,
    so catching only torch.OutOfMemoryError misses CPU-only OOMs entirely.
    Mirrors gotrain/templates/train_classification.py's is_oom_error.
    """
    import torch

    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "not enough memory" in message


def safe_to_drop_last(n_samples: int, batch_size: int) -> bool:
    """True if DataLoader(..., drop_last=True) is safe for TRAIN: dropping
    a trailing partial batch is only safe when at least one full batch
    remains. Without this, a training set smaller than batch_size would
    have drop_last silently skip every batch -- zero gradient updates per
    epoch, no error, no training -- which is worse than the
    batch-of-exactly-1 BatchNorm crash drop_last is meant to prevent. Pure
    arithmetic, no torch, unit-testable without the ML stack installed.
    """
    return n_samples >= batch_size


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


def build_dataset_arrays(task_specs, split: str, band_names, max_samples, logger, excluded_ids=None, loading_deadline=None):
    """Load every sample of `split` into memory as (N, H, W, C) uint8 images
    and (N, H, W) uint8 masks. GEO-Bench segmentation tasks here are small
    enough (a few thousand 320x320x3 or 500x500x3 samples) that this is
    simpler and fast enough for a hackathon training loop than writing a
    lazy on-disk Dataset; it also makes computing TRAIN-only normalization
    stats a single pass instead of two.

    `excluded_ids` (a DataIQ manifest's excluded_sample_ids, see
    gotrain/dataiq/manifest.py) skips those indices entirely. Mirrors
    train_classification.py's identical guard: a non-empty `excluded_ids`
    for any split other than "train" is refused outright, so a manifest
    can never touch validation even by mistake.

    `loading_deadline` (a time.time()-scale deadline, see
    LOADING_PHASE_BUDGET_FRACTION's comment) raises DataLoadingBudgetExceeded
    if loading is still running past it -- some segmentation datasets have
    much larger patches than the hackathon's original two (m-nz-cattle
    500x500, m-pv4ger-seg 320x320) with no sample cap by default, and a
    slow load here was previously only bounded by the parent process's
    hard, unrecorded subprocess kill.
    """
    if excluded_ids and split != "train":
        raise ValueError(
            f"excluded_ids was given for split={split!r} -- DataIQ manifests may only "
            "ever filter the TRAIN split."
        )
    excluded_ids = excluded_ids or set()

    geobench_split = {"train": "train", "validation": "valid"}[split]
    ds = task_specs.get_dataset(split=geobench_split)

    n = len(ds)
    if max_samples is not None:
        n = min(n, max_samples)

    indices = [i for i in range(n) if i not in excluded_ids]
    n_excluded_in_range = n - len(indices)
    if not indices:
        raise ValueError(f"No {split} samples left to load (n={n}, {n_excluded_in_range} excluded).")

    # Preallocated (not list + np.stack) so peak memory is one copy, not
    # two. Images keep their native dtype (uint8/int16) except float64,
    # which pack_sample narrows to float32.
    images = masks = None
    started_at = time.time()
    for k, i in enumerate(indices):
        data_loading.check_loading_budget(split, k, len(indices), started_at, loading_deadline)
        sample = ds[i]
        arr = data_loading.pack_sample(sample, band_names)
        mask = np.asarray(sample.label.data)
        if images is None:
            images = np.empty((len(indices),) + arr.shape, dtype=arr.dtype)
            masks = np.empty((len(indices),) + mask.shape, dtype=mask.dtype)
        images[k] = arr
        masks[k] = mask

    exclusion_note = f", {n_excluded_in_range} excluded by manifest" if n_excluded_in_range else ""
    logger.info(
        f"Loaded {split} split: {images.shape[0]} samples, images {images.shape}, "
        f"masks {masks.shape}{exclusion_note}"
    )
    return images, masks


def prepare_batch(images: np.ndarray, mean, std):
    """(N,H,W,C) numpy batch -> normalized NCHW float tensor, exactly as
    SegmentationArrayDataset prepares a training input. Used per batch by
    StressLab, final_eval and the DataIQ loss scan."""
    import torch

    x = (images.astype(np.float32) - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2))))


def compute_train_band_stats(train_images: np.ndarray, logger):
    """Per-band mean/std computed from TRAIN images only. train_images is
    (N, H, W, C) uint8; stats are returned as float64 arrays of length C.
    """
    mean, std = data_loading.band_stats(train_images)
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


def pad_to_multiple(height: int, width: int, multiple: int = 32) -> tuple:
    """Bottom/right padding needed to bring (height, width) up to the next
    multiple of `multiple` -- (0, 0) when already aligned. Pure arithmetic,
    no torch, so it's unit-testable without the ML stack installed.
    """
    return (-height) % multiple, (-width) % multiple


def build_model(architecture: str, encoder: str, pretrained: bool, in_channels: int, n_classes: int):
    """segmentation_models_pytorch's encoders downsample by 32x (5
    stride-2 stages) before the decoder upsamples back with skip
    connections. An input whose H/W isn't a multiple of 32 (e.g.
    m-nz-cattle's 500x500: 500 % 32 == 20) makes a decoder stage's
    upsampled feature map mismatch its skip connection's size, raising a
    RuntimeError on concatenation. m-pv4ger-seg's 320x320 is already a
    clean multiple of 32 and is unaffected.

    Padding/cropping is applied here, inside the model's own forward,
    rather than in the training loop, so every caller gets it for free:
    this template's train/val loops, gotrain/tools/final_eval.py,
    gotrain/rollback.py's inference sanity check, and
    gotrain/dataiq/loss_scan.py all call this same build_model.
    """
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp

    encoder_weights = "imagenet" if pretrained else None
    kwargs = dict(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=n_classes,
    )
    if architecture == "unet":
        model = smp.Unet(**kwargs)
    elif architecture == "fpn":
        model = smp.FPN(**kwargs)
    else:
        raise ValueError(f"architecture must be 'unet' or 'fpn', got {architecture!r}")

    original_forward = model.forward

    def padded_forward(x):
        h, w = x.shape[-2:]
        pad_h, pad_w = pad_to_multiple(h, w)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return original_forward(x)[..., :h, :w]

    model.forward = padded_forward
    return model


def build_loss(loss_name: str, n_classes: int, class_weights, focal_alpha, focal_gamma, device):
    """Dice and focal both help with class imbalance (pv4ger-seg is ~75/25,
    nz-cattle is sparser still): Dice directly optimizes the overlap metric
    instead of per-pixel cross-entropy, and focal down-weights easy
    (already-confident) pixels so the loss focuses on the hard/minority
    ones. class_weights (inverse frequency) is a third, complementary way
    to the same end and can be combined with ce or ce+dice.
    """
    import torch
    import segmentation_models_pytorch as smp

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    ce = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    focal = smp.losses.FocalLoss(mode="multiclass", alpha=focal_alpha, gamma=focal_gamma)

    if loss_name == "ce":
        return ce
    if loss_name == "dice":
        return dice
    if loss_name == "ce+dice":
        def combined_ce_dice(logits, target):
            return ce(logits, target) + dice(logits, target)
        return combined_ce_dice
    if loss_name == "focal":
        return focal
    if loss_name == "ce+focal":
        def combined_ce_focal(logits, target):
            return ce(logits, target) + focal(logits, target)
        return combined_ce_focal
    raise ValueError(
        f"loss must be 'ce', 'dice', 'ce+dice', 'focal', or 'ce+focal', got {loss_name!r}"
    )


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
    """Mirrors train_classification.py's build_scheduler: "none" (default)
    or "cosine" (LinearLR warmup -> CosineAnnealingLR decay, combined via
    SequentialLR). Call .step() once per epoch.
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
    # The parent's hard subprocess timeout counts from (roughly) here, so
    # every internal deadline is measured from this moment too.
    process_start = time.time()

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
    n_classes = task_specs.label_type.n_classes
    logger.info(f"Bands ({len(band_names)}/{len(all_band_names)} used): {band_names}")
    logger.info(f"Image size: {task_specs.patch_size}, classes: {n_classes} ({task_specs.label_type.class_names})")

    excluded_ids = set()
    if cfg["manifest_id"]:
        from gotrain.dataiq.manifest import get_excluded_ids

        excluded_ids = get_excluded_ids(dataset, cfg["manifest_id"])
        logger.info(f"DataIQ manifest {cfg['manifest_id']!r}: excluding {len(excluded_ids)} TRAIN sample(s)")

    # See LOADING_PHASE_BUDGET_FRACTION's comment: a hard deadline for
    # loading BOTH splits combined, derived from this run's actual
    # subprocess timeout, so a large split can't alone consume the whole
    # budget and get silently SIGKILLed with no checkpoint and no reason.
    loading_deadline = train_cutoff = None
    if cfg["_timeout_seconds"]:
        loading_deadline = process_start + cfg["_timeout_seconds"] * LOADING_PHASE_BUDGET_FRACTION
        train_cutoff = process_start + cfg["_timeout_seconds"] * TRAINING_CUTOFF_FRACTION
        logger.info(
            f"Time budget: loading until +{cfg['_timeout_seconds'] * LOADING_PHASE_BUDGET_FRACTION:.0f}s, "
            f"training until +{cfg['_timeout_seconds'] * TRAINING_CUTOFF_FRACTION:.0f}s, "
            f"of this run's {cfg['_timeout_seconds']:.0f}s timeout."
        )

    try:
        train_images, train_masks = build_dataset_arrays(
            task_specs, "train", band_names, cfg["max_train_samples"], logger,
            excluded_ids=excluded_ids, loading_deadline=loading_deadline,
        )
        val_images, val_masks = build_dataset_arrays(
            task_specs, "validation", band_names, cfg["max_val_samples"], logger,
            loading_deadline=loading_deadline,
        )
    except DataLoadingBudgetExceeded as e:
        logger.error(str(e))
        print(f"DATA LOADING TIMEOUT: {e}", file=sys.stderr)
        sys.exit(1)

    # Channel count comes from the packed data, not len(band_names): one
    # band name can expand to many channels (m-NeonTree's "Neon" band is
    # 369 hyperspectral channels).
    n_bands = train_images.shape[-1]
    logger.info(f"Input channels: {n_bands}")

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

    # drop_last=True on TRAIN only: if the sample count leaves a last
    # batch of exactly 1, BatchNorm raises "Expected more than 1 value
    # per channel when training" -- a real, previously-unguarded crash.
    # Guarded to never drop the ONLY batch an epoch would otherwise have
    # (a training set smaller than batch_size) -- that would silently
    # train on nothing every epoch instead of crashing, which is worse.
    # Never applied to validation (below): every held-out sample must
    # still be scored, and model.eval()'s BatchNorm uses running stats,
    # so a batch of 1 there is harmless anyway.
    drop_last_train = safe_to_drop_last(len(train_dataset), cfg["batch_size"])
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, drop_last=drop_last_train
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    fine_tuning_strategy = cfg["fine_tuning_strategy"]
    effective_pretrained = cfg["pretrained"] and fine_tuning_strategy != "from_scratch"

    try:
        model = build_model(cfg["architecture"], cfg["encoder"], effective_pretrained, n_bands, n_classes)
        model = model.to(device)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if not is_oom_error(e):
            raise
        logger.error("Out of memory while building/moving the model to device.")
        print("OUT OF MEMORY while constructing the model. Try a smaller --batch_size or a lighter encoder.", file=sys.stderr)
        sys.exit(1)

    # Recursive self-improvement -- see train_classification.py's
    # identical block for the full rationale: if launch_training resolved
    # a parent checkpoint (config["warm_start_from_parent"]), load its
    # learned weights now, before fine-tuning strategy freezes anything.
    warm_start_checkpoint_path = cfg.get("_warm_start_checkpoint_path")
    warm_started_from = None
    if warm_start_checkpoint_path:
        try:
            state_dict = torch.load(warm_start_checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.error(f"Warm-start failed to load parent checkpoint {warm_start_checkpoint_path}: {e}")
            print(
                f"WARM-START FAILED: could not load parent checkpoint into this architecture: {e}. "
                "This usually means the architecture/encoder changed between parent and child run.",
                file=sys.stderr,
            )
            sys.exit(1)
        warm_started_from = {
            "parent_run_id": cfg.get("_warm_start_parent_run_id"),
            "parent_version": cfg.get("_warm_start_parent_version"),
            "checkpoint_path": warm_start_checkpoint_path,
        }
        logger.info(f"Warm-started from parent run {warm_started_from['parent_run_id']} ({warm_started_from['parent_version']})")

    from gotrain.finetuning import apply_fine_tuning_strategy, trainable_parameters

    fine_tuning_summary = apply_fine_tuning_strategy(model, "smp", fine_tuning_strategy)
    n_params = fine_tuning_summary["total_parameters"]
    logger.info(
        f"Model: {cfg['architecture']}/{cfg['encoder']} (pretrained={effective_pretrained}), {n_params:,} parameters, "
        f"fine_tuning_strategy={fine_tuning_strategy!r} -- "
        f"{fine_tuning_summary['trainable_parameters']:,} trainable / "
        f"{fine_tuning_summary['frozen_parameters']:,} frozen"
    )

    # Defense in depth: gotrain/tools/training.py already rejects a
    # class_weights/n_classes length mismatch pre-flight when it can
    # resolve n_classes cheaply; this catches the case where it couldn't
    # (or this template was invoked directly, bypassing launch_training)
    # with a clear, logged message and a clean exit instead of a bare
    # RuntimeError surfacing from inside CrossEntropyLoss.
    from gotrain.config_schema import ConfigValidationError, validate_class_weights

    try:
        validate_class_weights("segmentation", class_weights, n_classes)
    except ConfigValidationError as e:
        logger.error(str(e))
        print(f"INVALID CONFIG: {e}", file=sys.stderr)
        sys.exit(1)

    loss_fn = build_loss(cfg["loss"], n_classes, class_weights, cfg["focal_alpha"], cfg["focal_gamma"], device)
    optimizer = build_optimizer(cfg["optimizer"], trainable_parameters(model), cfg["lr"], cfg["weight_decay"])
    scheduler = build_scheduler(cfg["scheduler"], optimizer, cfg["epochs"], cfg["warmup_epochs"])
    scaler = GradScaler(device="cuda", enabled=amp_enabled)

    best_val_iou = -1.0
    best_metrics = None
    best_checkpoint_path = run_dir / "best_model.pt"
    time_budget_seconds = cfg["time_budget_minutes"] * 60 if cfg["time_budget_minutes"] else None
    training_start = time.time()
    interrupted_reason = None

    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        try:
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                with autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(images)
                    loss = loss_fn(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                n_batches += 1
                # Soft time cutoff -- see train_classification.py.
                if train_cutoff is not None and time.time() > train_cutoff:
                    interrupted_reason = "time_budget"
                    break
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

        if n_batches == 0 and epoch == 0:
            logger.error("Time budget exhausted before a single training step completed.")
            print(
                "TIME BUDGET EXHAUSTED before any training step: loading left no time to train. "
                "Retry with a smaller max_train_samples/max_val_samples or a larger run budget.",
                file=sys.stderr,
            )
            sys.exit(1)
        if interrupted_reason == "time_budget":
            logger.info(f"Time budget cutoff reached during epoch {epoch + 1} after {n_batches} batch(es).")

        avg_train_loss = running_loss / max(n_batches, 1)

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                try:
                    with autocast(device_type=device.type, enabled=amp_enabled):
                        logits = model(images)
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
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(masks.numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        mean_iou, per_class_iou = compute_mean_iou(all_preds, all_targets, n_classes)

        # For binary tasks, report the foreground (class 1) IoU explicitly.
        foreground_iou = per_class_iou[1] if n_classes == 2 and len(per_class_iou) > 1 else None

        if scheduler is not None:
            scheduler.step()

        epoch_seconds = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch + 1}/{cfg['epochs']} — train_loss={avg_train_loss:.4f} "
            f"val_mean_iou={mean_iou:.4f} per_class_iou={per_class_iou} "
            f"foreground_iou={foreground_iou} time={epoch_seconds:.1f}s (split=validation)"
        )

        report_progress(
            run_id=run_id,
            dataset=dataset,
            task_type="segmentation",
            model=f"{cfg['architecture']}/{cfg['encoder']}",
            optimizer=cfg["optimizer"],
            lr=cfg["lr"],
            scheduler=None,
            epoch=epoch + 1,
            epochs=cfg["epochs"],
            train_loss=avg_train_loss,
            val_metric=mean_iou,
            metric_name="val_mean_iou",
            foreground_iou=foreground_iou,
            best_val_metric=max(best_val_iou, mean_iou),
            elapsed_seconds=time.time() - training_start,
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

        if interrupted_reason == "time_budget":
            break
        elapsed = time.time() - training_start
        if time_budget_seconds is not None and elapsed > time_budget_seconds:
            logger.info(f"Time budget of {cfg['time_budget_minutes']} minutes reached — stopping early.")
            interrupted_reason = "time_budget"
            break
        # Predictive: don't start an epoch that can't finish before the cutoff.
        if train_cutoff is not None and epoch + 1 < cfg["epochs"] and time.time() + epoch_seconds > train_cutoff:
            logger.info("Next epoch would overrun the run's time budget — stopping with the best checkpoint so far.")
            interrupted_reason = "time_budget"
            break

        # Engineer control plane (gotrain.control): see
        # train_classification.py's identical check for the full
        # rationale. Checked once per epoch, best-effort, never allowed
        # to fail a real training run.
        try:
            from gotrain.control import should_stop_training

            interrupted_reason = should_stop_training()
        except Exception:
            interrupted_reason = None
        if interrupted_reason:
            logger.info(f"Stopping early: {interrupted_reason} (engineer control).")
            break

    total_seconds = time.time() - training_start
    gpu_minutes = total_seconds / 60 if device.type == "cuda" else 0.0

    final_val_metric = best_val_iou
    metrics = {
        "run_id": run_id,
        "dataset": dataset,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "n_params": n_params,
        "n_bands": n_bands,
        # Saved so StressLab / final_eval / loss-scan normalize exactly as
        # training did without re-loading the whole train split.
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "band_names": band_names,
        "n_classes": n_classes,
        "dependency_versions": reproducibility.capture_dependency_versions(),
        "warm_started_from": warm_started_from,
        "class_pixel_frequencies": class_freqs.tolist(),
        "best": best_metrics,
        "final_val_metric": final_val_metric,
        "split_used": "validation",
        "total_train_seconds": total_seconds,
        "interrupted": interrupted_reason is not None,
        "interrupted_reason": interrupted_reason,
        "fine_tuning": fine_tuning_summary,
        "manifest_id": cfg["manifest_id"],
        "n_excluded_by_manifest": len(excluded_ids),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(f"Run finished in {total_seconds:.1f}s. Best val_mean_iou={best_val_iou:.4f}")
    logger.info(f"FINAL_VAL_METRIC: {final_val_metric}")

    rationale = f"best val_mean_iou={final_val_metric:.4f} at epoch {best_metrics['epoch']} (run_dir={run_dir})"
    if interrupted_reason:
        rationale += f" -- stopped early: {interrupted_reason}"

    print(json.dumps({
        "val_metric": final_val_metric,
        "gpu_minutes": gpu_minutes,
        "rationale": rationale,
        "run_dir": str(run_dir),
        "checkpoint_path": str(best_checkpoint_path),
        "extra_metrics": {
            "foreground_iou": best_metrics.get("val_foreground_iou"),
            "interrupted": interrupted_reason is not None,
            "interrupted_reason": interrupted_reason,
            "fine_tuning_strategy": fine_tuning_summary["fine_tuning_strategy"],
            "trainable_parameters": fine_tuning_summary["trainable_parameters"],
            "frozen_parameters": fine_tuning_summary["frozen_parameters"],
            "manifest_id": cfg["manifest_id"],
            "n_excluded_by_manifest": len(excluded_ids),
        },
    }))


if __name__ == "__main__":
    main()
