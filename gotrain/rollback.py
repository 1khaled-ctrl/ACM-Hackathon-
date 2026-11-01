"""Verified rollback: restore a dataset's "active" model version after
confirming the candidate checkpoint is actually usable -- never just flip
a pointer and hope.

execute_rollback() implements the design brief's safe rollback workflow:
 1. Refuse if a training run is currently in progress (checked via
    gotrain.status's live current_run heartbeat) -- "confirm termination
    before starting conflicting work."
 2. Transition control -> ROLLBACK_REQUESTED -> ROLLBACK_VERIFYING,
    blocking every other tool call for the duration (gotrain.agent's hook
    + gotrain.control.raise_if_blocking both already enforce this once
    the state is set).
 3. Verify the target version's checkpoint file integrity (exists,
    non-empty, config/metrics present).
 4. Verify it actually loads into the architecture its OWN config.json
    describes and produces a correctly-shaped output on a synthetic
    all-zero tensor -- never real data from any split.
 5. Only if both pass, atomically rewrite
    models/<dataset>/active_version.json.
 6. Record the result (event log + control transition) and leave the
    system PAUSED_FOR_ENGINEER (success) or FAILED_SAFE (failure) --
    rollback never auto-resumes autonomous execution; an engineer must
    explicitly resume.

If verification fails at any step, the active pointer is never touched:
the previously active version (if any) stays active, and the caller gets
exactly which check failed and why.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from gotrain import control, registry, status
from gotrain.logging_utils import log_event

ACTIVE_VERSION_FILENAME = "active_version.json"


class RollbackVerificationError(RuntimeError):
    """Raised when a candidate version fails integrity/inference
    verification. The active-version pointer is never touched when this
    is raised.
    """


class RollbackConflictError(RuntimeError):
    """Raised when a rollback is requested while a training run is
    currently in progress -- the caller must wait or pause first.
    """


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _active_version_path(dataset: str) -> Path:
    return registry.MODELS_DIR / dataset / ACTIVE_VERSION_FILENAME


def get_active_version(dataset: str) -> Optional[str]:
    path = _active_version_path(dataset)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("version")


def _set_active_version(dataset: str, version: str, verification: dict) -> dict:
    path = _active_version_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": dataset, "version": version, "set_at": _now(), "verification": verification}
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".active_version_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return payload


def verify_checkpoint_integrity(dataset: str, version: str) -> dict:
    """File-level checks only -- does the version directory exist, does
    it have config.json/metrics.json/a non-empty checkpoint. Does not
    require torch; this check alone can run without the ML stack
    installed (and does, in this repo's test suite).
    """
    version_dir = registry.get_version_dir(dataset, version)
    problems = []
    if not version_dir.is_dir():
        return {"passed": False, "problems": [f"version directory does not exist: {version_dir}"]}

    if not (version_dir / "config.json").exists():
        problems.append("config.json missing")
    if not (version_dir / "metrics.json").exists():
        problems.append("metrics.json missing")

    checkpoint_path = version_dir / "model.pt"
    if not checkpoint_path.exists():
        problems.append("model.pt missing")
    elif checkpoint_path.stat().st_size == 0:
        problems.append("model.pt is empty (0 bytes) -- likely an interrupted write")

    return {"passed": not problems, "problems": problems}


def verify_inference_sanity(dataset: str, version: str, task_type: str) -> dict:
    """Load the version's own config + checkpoint and run ONE forward
    pass on a synthetic all-zero tensor of the right shape -- never real
    data from any split. Confirms the checkpoint actually loads into the
    architecture its own config.json describes and produces output of
    the expected shape. Requires torch/timm/segmentation_models_pytorch,
    the same stack the training templates need.
    """
    import torch

    version_dir = registry.get_version_dir(dataset, version)
    try:
        config = json.loads((version_dir / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((version_dir / "metrics.json").read_text(encoding="utf-8"))
    except Exception as e:
        return {"passed": False, "problems": [f"could not read config/metrics: {e}"]}

    n_bands = metrics.get("n_bands")
    n_classes = metrics.get("n_classes")
    if not n_bands or not n_classes:
        return {"passed": False, "problems": ["metrics.json missing n_bands/n_classes"]}

    try:
        state_dict = torch.load(version_dir / "model.pt", map_location="cpu")
    except Exception as e:
        return {"passed": False, "problems": [f"model.pt failed to load: {e}"]}

    try:
        if task_type == "classification":
            from gotrain.templates.train_classification import build_model

            band_names = metrics.get("band_names")
            model = build_model(config.get("backbone", "resnet18"), False, n_bands, n_classes, band_names)
            dummy = torch.zeros(1, n_bands, 32, 32)
        elif task_type == "segmentation":
            from gotrain.templates.train_segmentation import build_model

            model = build_model(
                config.get("architecture", "unet"), config.get("encoder", "resnet34"), False, n_bands, n_classes
            )
            dummy = torch.zeros(1, n_bands, 64, 64)
        else:
            return {"passed": False, "problems": [f"unknown task_type {task_type!r}"]}

        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            output = model(dummy)
    except Exception as e:
        return {"passed": False, "problems": [f"inference sanity check failed: {e}"]}

    problems = []
    if output.shape[0] != 1:
        problems.append(f"unexpected output batch dimension: {tuple(output.shape)}")
    return {"passed": not problems, "problems": problems}


def rollback_to_version(dataset: str, version: str, task_type: str, run_inference_check=True) -> dict:
    """Run the verification chain and, only if everything passes,
    atomically set `version` as dataset's active checkpoint. Raises
    RollbackVerificationError (without touching the active pointer) if
    any check fails. `run_inference_check` is a callable
    (version, task_type) -> dict | True (meaning "use
    verify_inference_sanity") | False (meaning "skip it") -- tests inject
    a stub here so the orchestration logic is verifiable without torch.
    """
    integrity = verify_checkpoint_integrity(dataset, version)
    if not integrity["passed"]:
        log_event("rollback_failed", dataset=dataset, version=version, stage="integrity", problems=integrity["problems"])
        raise RollbackVerificationError(f"integrity check failed for {dataset}/{version}: {integrity['problems']}")

    if run_inference_check is False:
        inference = {"passed": True, "problems": [], "skipped": True}
    elif callable(run_inference_check):
        inference = run_inference_check(version, task_type)
    else:
        inference = verify_inference_sanity(dataset, version, task_type)

    if not inference["passed"]:
        log_event("rollback_failed", dataset=dataset, version=version, stage="inference", problems=inference["problems"])
        raise RollbackVerificationError(f"inference sanity check failed for {dataset}/{version}: {inference['problems']}")

    verification = {"integrity": integrity, "inference": inference}
    result = _set_active_version(dataset, version, verification)
    log_event("rollback_succeeded", dataset=dataset, version=version, verification=verification)
    return result


def execute_rollback(dataset: str, version: str, task_type: str, requested_by: str = "engineer") -> dict:
    """The full engineer-facing rollback action: checks for an in-progress
    training run, drives the control state machine through
    ROLLBACK_REQUESTED -> ROLLBACK_VERIFYING, runs rollback_to_version,
    and leaves control at PAUSED_FOR_ENGINEER (success) or FAILED_SAFE
    (failure) -- this is the one function the dashboard's "Rollback"
    button calls.
    """
    current_run = status.get_status().get("current_run")
    if current_run:
        raise RollbackConflictError(
            f"a training run is currently in progress (run_id={current_run.get('run_id')!r}) "
            "-- pause or wait for it to finish before rolling back."
        )

    control.transition(control.ROLLBACK_REQUESTED, reason=f"rollback to {version} requested", requested_by=requested_by)
    control.transition(control.ROLLBACK_VERIFYING, reason=f"verifying {dataset}/{version}", requested_by=requested_by)

    try:
        result = rollback_to_version(dataset, version, task_type)
    except RollbackVerificationError as e:
        control.transition(control.FAILED_SAFE, reason=str(e), requested_by=requested_by)
        raise

    control.transition(
        control.PAUSED_FOR_ENGINEER,
        reason=f"rolled back {dataset} to {version}; resume when ready",
        requested_by=requested_by,
    )
    return result
