# Example log set

A real (not fabricated) small session on `m-eurosat`: 2 training runs
(second derived from the first via `parent_id`, switching optimizer to
`adamw`), both run with small `max_train_samples`/`max_val_samples` so it
finished in seconds rather than minutes — same code path as a real run,
just capped for speed.

- `logs/` — the event stream, run ledger, and final status from this session.
- `runs/m-eurosat/<run_id>/` — each run's `config.json`, `metrics.json`, and
  `train.log` (device, param count, per-epoch validation metrics, class
  frequencies).
- `models/m-eurosat/v1/`, `v2/` — both runs promoted to the model registry.

Checkpoint files (`.pt`, ~90MB each) are excluded — they're exactly what
`.gitignore` keeps out of the real `runs/`/`models/` directories too; this
folder mirrors their shape, not their full weight.
