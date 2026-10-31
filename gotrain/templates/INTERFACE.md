# Training template interface

This is the contract every `gotrain/templates/train_*.py` script must follow,
derived from the current, tested `fake_train.py` and from what
`gotrain/tools/training.py::launch_training` actually calls today. Keep
this file in sync if that contract changes — don't let the two trainers
drift apart.

## CLI

```
python train_<task>.py --dataset <name> --config '<json object>'
```

`launch_training` invokes exactly this (`sys.executable`, the template
path, `--dataset`, `--config`) and nothing else — no run-directory or
run-id argument is passed in today. A template is responsible for
generating its own run directory if it wants one.

## Stdout contract (hard requirement)

`launch_training` does:

```python
result = json.loads(proc.stdout.strip().splitlines()[-1])
val_metric = result["val_metric"]
```

So:
- The **last line** written to stdout **must** be a single JSON object with
  at least a `"val_metric"` key (float). `"gpu_minutes"` and `"rationale"`
  are optional but read if present.
- Everything else a template wants to print for a human (progress, a
  `FINAL_VAL_METRIC: <value>` summary line, warnings, etc.) must go to
  **stderr**, or to a log file — never to stdout before that last line,
  since any earlier stdout line is ignored by `launch_training` but would
  break a human piping stdout only.
- A non-zero exit code is treated as a training failure by `launch_training`
  (it raises `RuntimeError` with the captured stderr). Exit non-zero on any
  unrecoverable error (including OOM) after printing a clear message.

## Run directory and files (not read by launch_training, but expected by convention)

Each run writes to its own directory under `runs/<dataset>/<run_id>/`
(created by the template itself; `runs/` is gitignored):

- `config.json` — the exact config dict used for the run (CLI `--config`,
  merged with any defaults the template applied).
- `metrics.json` — final metrics (validation + any per-class/per-epoch
  detail worth keeping), plus `band_names`, `n_bands`, timing, and device
  info.
- `train.log` — full human-readable training log (device, param count,
  per-epoch timing, class frequencies, per-epoch validation metrics).
- a checkpoint file for the best epoch (`*.pt`, gitignored).

## Splits

- Train on **train**. Evaluate every epoch on **validation**. Never open
  the **test** split — that happens exactly once, elsewhere
  (`gotrain/tools/final_eval.py`), never inside a training template.

## Segmentation-specific config keys (`train_segmentation.py`)

```jsonc
{
  "dataset": "m-pv4ger-seg",       // GEO-Bench task name
  "encoder": "resnet34",           // any segmentation_models_pytorch encoder
  "pretrained": true,               // imagenet weights for the encoder
  "architecture": "unet",           // "unet" | "fpn"
  "lr": 1e-3,
  "batch_size": 8,
  "epochs": 2,
  "optimizer": "adam",              // "adam" | "sgd"
  "weight_decay": 0.0,
  "loss": "ce",                     // "ce" | "dice" | "ce+dice"
  "class_weights": null,            // null, or a list[float] of length n_classes
  "augmentation": "none",           // "none" | "basic" | "strong"
  "bands": null,                    // null (= all bands) or a list of band names
  "time_budget_minutes": null,      // soft wall-clock budget; stops early if hit
  "seed": 0,
  "max_train_samples": null,        // for smoke tests: cap on train set size
  "max_val_samples": null
}
```

Any key not given falls back to the default shown above.
