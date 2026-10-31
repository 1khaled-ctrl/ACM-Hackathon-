# Training template interface

This is the contract every `gotrain/templates/train_*.py` script follows,
derived from `fake_train.py` and from what
`gotrain/tools/training.py::launch_training` actually calls. Keep this
file in sync when that contract changes — don't let the trainers drift
apart from each other or from `gotrain/config_schema.py` (the single
source of truth for which config values are valid per `task_type`).

## CLI

```
python train_<task>.py --dataset <name> --config '<json object>'
```

`launch_training` invokes exactly this (`sys.executable`, the template
path, `--dataset`, `--config`) and nothing else — no run-directory or
run-id argument is passed in. A template generates its own run id/directory.

## Config validation (before any subprocess runs)

`launch_training` calls `gotrain.config_schema.validate_config(task_type,
config)` **before** spawning the training subprocess. An unsupported
value for a closed-choice field (optimizer, scheduler, loss,
backbone/encoder, architecture, augmentation) is rejected immediately —
no wasted compute, and the rejection does **not** count against the
agent's `--max-runs` budget (see `gotrain/agent.py`'s
`LimitsAndLoggingHooks`). The error names the field, the bad value, and
the valid values. When a trainer adds a new value, add it to
`gotrain/config_schema.py`'s `CLASSIFICATION_VALID_VALUES` /
`SEGMENTATION_VALID_VALUES` in the same change, or the new value will be
rejected before the trainer ever sees it.

## Stdout contract (hard requirement)

`launch_training` does:

```python
result = json.loads(proc.stdout.strip().splitlines()[-1])
val_metric = result["val_metric"]
```

So:
- The **last line** written to stdout **must** be a single JSON object
  with at least a `"val_metric"` key (float). Also read if present:
  `"gpu_minutes"`, `"rationale"`, `"run_dir"` (the run's output directory,
  used to promote it into the model registry — see
  `gotrain/registry.py`), `"checkpoint_path"` (the best checkpoint file,
  also used for registry promotion).
- Everything else a template wants to print for a human (progress, a
  `FINAL_VAL_METRIC: <value>` summary line, warnings, etc.) must go to
  **stderr**, or to a log file — never to stdout before that last line.
  Both trainers log `FINAL_VAL_METRIC` exactly once, via the logger
  (`train.log` + stderr), never via a second stdout `print()`.
- A non-zero exit code is treated as a training failure by `launch_training`
  (it raises `RuntimeError` with the captured stderr). Exit non-zero on any
  unrecoverable error (including OOM — see `is_oom_error()` in both
  trainers, which also catches plain CPU allocator `RuntimeError`s, not
  just `torch.OutOfMemoryError`) after printing a clear message.

## Run directory and files

Each run writes to its own directory under `runs/<dataset>/<run_id>/`
(created by the template itself; `/runs/` is gitignored):

- `config.json` — the exact config dict used for the run (CLI `--config`,
  merged with the template's own defaults).
- `metrics.json` — final metrics (validation + any per-class/per-epoch
  detail worth keeping), plus `band_names`, `n_bands`, device, AMP status,
  and timing info.
- `train.log` — full human-readable training log (device, param count,
  per-epoch timing, class frequencies, per-epoch validation metrics, each
  tagged `(split=validation)`).
- a checkpoint file for the best epoch (`*.pt`, gitignored).

`gotrain/tools/training.py::launch_training` promotes `config.json`,
`metrics.json`, and the checkpoint into `models/<dataset>/<version>/`
(`gotrain/registry.py::register_run`) whenever the stdout JSON includes
`run_dir` — the resulting `version_id` (e.g. `"v1"`) is stored on the
ledger record. `fake_train.py` doesn't report a `run_dir`, so fake runs
are never registered.

## Splits

- Train on **train**. Evaluate every epoch on **validation**. Never open
  the **test** split — that happens exactly once, elsewhere
  (`gotrain/tools/final_eval.py`), never inside a training template.

## Classification config keys (`train_classification.py`)

```jsonc
{
  "dataset": "m-eurosat",          // GEO-Bench task name
  "backbone": "resnet18",          // "resnet18" | "resnet50" | "efficientnet_b0" | "convnext_tiny" (timm)
  "pretrained": true,              // imagenet weights for the backbone
  "optimizer": "adam",             // "adam" | "sgd" | "adamw"
  "scheduler": "none",             // "none" | "cosine" (LinearLR warmup -> CosineAnnealingLR)
  "warmup_epochs": 0,              // only used when scheduler is "cosine"
  "lr": 1e-3,
  "weight_decay": 0.0,
  "label_smoothing": 0.0,          // passed to CrossEntropyLoss
  "batch_size": 32,
  "epochs": 2,
  "augmentation": "none",          // "none" | "basic" | "strong" (flip+rotation+light noise)
  "image_size": null,              // int (resize to size x size) or null
  "bands": null,                   // null (= all bands) or a list of band names
  "class_weights": null,           // null, or a list[float] of length n_classes
  "amp": true,                     // torch.amp; no effect without a CUDA device
  "tta": false,                    // test-time augmentation at VALIDATION only
  "time_budget_minutes": null,     // soft wall-clock budget; stops early if hit
  "seed": 0,
  "max_train_samples": null,       // for smoke tests: cap on train set size
  "max_val_samples": null,
  "fine_tuning_strategy": "full",  // "from_scratch" | "linear_probe" | "partial" | "full" (gotrain/finetuning.py)
  "manifest_id": null              // null/"base" (= no exclusions) or a DataIQ manifest id (gotrain/dataiq/manifest.py)
}
```

**fine_tuning_strategy**: which parameters actually get gradients (see
`gotrain/finetuning.py`). `"from_scratch"` forces `pretrained=false`
regardless of the `pretrained` key; `"linear_probe"`/`"partial"` require
`pretrained=true` (rejected pre-flight otherwise, by
`gotrain/config_schema.py::validate_fine_tuning_strategy`) since freezing
most/all of a randomly-initialized backbone cannot learn. Recorded in
`metrics.json`'s `"fine_tuning"` block (total/trainable/frozen parameter
counts) and in the ledger's `extra_metrics`.

**manifest_id**: excludes a DataIQ manifest's `excluded_sample_ids` (TRAIN
split only -- see `gotrain/dataiq/manifest.py`) before computing TRAIN
band stats or building the training `Dataset`. Rejected pre-flight if the
manifest doesn't exist (`gotrain.dataiq.manifest.validate_manifest_exists`,
called from `gotrain/tools/training.py`). Never applied to the validation
split -- `build_dataset_arrays` refuses a non-empty `excluded_ids` for any
split other than `"train"`.

**m-eurosat band warning**: geo-bench-1.0 stores incorrect band names for
most of m-eurosat's channels (pixel data is unaffected; selecting bands
by name grabs the wrong channel). `train_classification.py` logs a
warning if `bands` is non-null for `dataset="m-eurosat"`; the system
prompt tells the agent not to do this.

## Segmentation config keys (`train_segmentation.py`)

```jsonc
{
  "dataset": "m-pv4ger-seg",       // GEO-Bench task name
  "encoder": "resnet34",           // "resnet34" | "resnet50" | "efficientnet-b3" (segmentation_models_pytorch)
  "pretrained": true,               // imagenet weights for the encoder
  "architecture": "unet",           // "unet" | "fpn"
  "optimizer": "adam",              // "adam" | "sgd" | "adamw"
  "scheduler": "none",              // "none" | "cosine" (LinearLR warmup -> CosineAnnealingLR)
  "warmup_epochs": 0,               // only used when scheduler is "cosine"
  "lr": 1e-3,
  "batch_size": 8,
  "epochs": 2,
  "weight_decay": 0.0,
  "loss": "ce",                     // "ce" | "dice" | "ce+dice" | "focal" | "ce+focal"
  "focal_alpha": null,              // only used by "focal"/"ce+focal"
  "focal_gamma": 2.0,               // only used by "focal"/"ce+focal"
  "class_weights": null,            // null, or a list[float] of length n_classes
  "augmentation": "none",           // "none" | "basic" | "strong" (flip + 90-degree rotation)
  "bands": null,                    // null (= all bands) or a list of band names
  "amp": true,                      // torch.amp; no effect without a CUDA device
  "time_budget_minutes": null,      // soft wall-clock budget; stops early if hit
  "seed": 0,
  "max_train_samples": null,        // for smoke tests: cap on train set size
  "max_val_samples": null,
  "fine_tuning_strategy": "full",   // "from_scratch" | "linear_probe" | "partial" | "full" (gotrain/finetuning.py)
  "manifest_id": null               // null/"base" (= no exclusions) or a DataIQ manifest id (gotrain/dataiq/manifest.py)
}
```

See the classification section above for `fine_tuning_strategy`/
`manifest_id`'s semantics — identical for segmentation, just applied to
`segmentation_head.*` instead of `fc.*`/`classifier.*`/`head.*` as the
backbone-vs-head split (`gotrain/finetuning.py`'s `"smp"` family).

Any key not given falls back to the default shown above — every key
added since the first version of this file is optional, so a config
written before that key existed still works unchanged.
