"""Regression tests for the run-id / checkpoint mismatch found in the
pre-submission audit.

Before the fix, launch_training stored a fresh uuid as the ledger run_id
while the training template saved its checkpoint under its OWN uuid
(runs/<dataset>/<template_uuid>/best_model.pt). StressLab, DataIQ loss-scan
and final_eval looked up runs/<dataset>/<ledger_run_id>/best_model.pt, so
for every real run they raised FileNotFoundError (reproduced against a real
m-eurosat training run; see the end-to-end test at the bottom). final_eval's
failure was the worst case: it happens after the one-time test lock is
taken, so it would have burned the dataset's single test evaluation.
"""

import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import control, experiment_planner, ledger, logging_utils, registry, run_artifacts  # noqa: E402
from gotrain.tools import final_eval, training  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setattr(logging_utils, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", logs_dir / "events.jsonl")
    monkeypatch.setattr(logging_utils, "STATUS_PATH", logs_dir / "status.json")
    monkeypatch.setattr(ledger, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(ledger, "LEDGER_PATH", logs_dir / "ledger.jsonl")
    monkeypatch.setattr(final_eval, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(control, "CONTROL_PATH", logs_dir / "control.json")
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    runs = tmp_path / "runs"
    for module in (run_artifacts, final_eval, experiment_planner):
        monkeypatch.setattr(module, "RUNS_DIR", runs)
    return SimpleNamespace(tmp=tmp_path, logs=logs_dir, runs=runs)


def _write_template_run_dir(runs: Path, dataset: str, name: str) -> Path:
    run_dir = runs / dataset / name
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.pt").write_bytes(b"ckpt")
    (run_dir / "metrics.json").write_text(json.dumps({"best": {"epoch": 1}}), encoding="utf-8")
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def test_resolves_checkpoint_in_runs_dir_by_run_id(env):
    _write_template_run_dir(env.runs, "m-eurosat", "run-a")
    assert run_artifacts.checkpoint_path("m-eurosat", "run-a", env.runs).name == "best_model.pt"


def test_legacy_run_resolves_through_registry_version(env):
    """A ledger record written BEFORE the fix: its run_id never matched any
    runs/ directory, but its version_id points at the registry copy."""
    template_dir = _write_template_run_dir(env.runs, "m-eurosat", "template-uuid")
    version = registry.register_run("m-eurosat", "legacy-ledger-id", str(template_dir), str(template_dir / "best_model.pt"))
    ledger.add_run(dataset="m-eurosat", config={}, val_metric=0.8, run_id="legacy-ledger-id", version_id=version)

    path = run_artifacts.checkpoint_path("m-eurosat", "legacy-ledger-id", env.runs)
    assert path == registry.get_version_dir("m-eurosat", version) / "model.pt"
    assert final_eval.resolve_checkpoint("m-eurosat", "legacy-ledger-id") == path
    assert run_artifacts.metrics_path("m-eurosat", "legacy-ledger-id", env.runs).parent.name == version


def test_missing_checkpoint_raises_with_every_location_named(env):
    ledger.add_run(dataset="m-eurosat", config={}, val_metric=0.8, run_id="r1", version_id="v9")
    with pytest.raises(FileNotFoundError) as exc:
        run_artifacts.checkpoint_path("m-eurosat", "r1", env.runs)
    assert "best_model.pt" in str(exc.value) and "model.pt" in str(exc.value)


def test_zero_byte_checkpoint_is_not_accepted(env):
    run_dir = env.runs / "m-eurosat" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.pt").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        run_artifacts.checkpoint_path("m-eurosat", "r1", env.runs)


def test_corrupt_metrics_file_reads_as_missing(env):
    run_dir = env.runs / "m-eurosat" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text('{"best": {"epoch"', encoding="utf-8")
    assert experiment_planner.get_run_metrics_file("m-eurosat", "r1") is None


def test_run_final_eval_preflight_refuses_before_taking_the_lock(env):
    sys.path.insert(0, str(REPO_ROOT))
    import run_final_eval

    run = {"dataset": "m-eurosat", "run_id": "no-such-run", "task_type": "classification", "config": {}}
    with pytest.raises(FileNotFoundError):
        run_final_eval.preflight(run)
    assert not list(env.logs.glob("test_lock_*.json"))


# ---------------------------------------------------------------------------
# launch_training: id unification, exit codes, missing result line
# ---------------------------------------------------------------------------


def _fake_proc(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_ledger_run_id_equals_template_run_dir_name(env, monkeypatch):
    run_dir = _write_template_run_dir(env.runs, "m-eurosat", "template-run-123")
    out = json.dumps({"val_metric": 0.7, "run_dir": str(run_dir), "checkpoint_path": str(run_dir / "best_model.pt")})
    monkeypatch.setattr(training.subprocess, "run", lambda *a, **k: _fake_proc(stdout=out))
    monkeypatch.setattr(training.datasets_tools, "get_dataset_info", lambda d: {"label_info": {"n_classes": 10}})

    record = training.launch_training("m-eurosat", "classification", {})
    assert record["run_id"] == "template-run-123"
    assert run_artifacts.checkpoint_path("m-eurosat", record["run_id"], env.runs) == run_dir / "best_model.pt"


def test_killed_subprocess_records_exit_code_and_oom_hint(env, monkeypatch):
    monkeypatch.setattr(training.subprocess, "run", lambda *a, **k: _fake_proc(returncode=-9, stderr="Loaded train split: 3000 samples\n"))
    with pytest.raises(RuntimeError, match="out-of-memory"):
        training.launch_training("m-SA-crop-type", "segmentation", {})
    event = logging_utils.read_events(event="training_failed")[-1]
    assert event["returncode"] == -9
    assert "SIGKILL" in event["reason"]


def test_exit_zero_without_result_line_is_a_clear_failure(env, monkeypatch):
    monkeypatch.setattr(training.subprocess, "run", lambda *a, **k: _fake_proc(stdout=""))
    with pytest.raises(RuntimeError, match="did not print its result JSON"):
        training.launch_training("m-eurosat", "classification", {})
    assert logging_utils.read_events(event="training_failed")


@pytest.mark.parametrize("code,expected", [(-9, "out-of-memory"), (-15, "signal 15"), (1, "code 1")])
def test_describe_exit_code(code, expected):
    assert expected in training.describe_exit_code(code)


# ---------------------------------------------------------------------------
# Stop reason uses the exit-code reason and the informative stderr line
# ---------------------------------------------------------------------------


@pytest.fixture
def history_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GOTRAIN_LOGS_DIR", str(tmp_path))
    from gotrain import history

    importlib.reload(logging_utils)
    importlib.reload(ledger)
    importlib.reload(history)
    yield history
    monkeypatch.delenv("GOTRAIN_LOGS_DIR", raising=False)
    importlib.reload(logging_utils)
    importlib.reload(ledger)
    importlib.reload(history)


def test_stop_reason_prefers_failure_marker_over_trailing_progress_line(history_env):
    history = history_env
    rec = history.start_session(dataset="m-cashew-plant", max_runs=3, run_minutes=5)
    logging_utils.log_event(
        "training_failed", dataset="m-cashew-plant", returncode=1, reason="exited with code 1",
        stderr="Loaded train split\nOUT OF MEMORY during training (epoch 0, batch_size=8).\nsome trailing info\n",
    )
    reason = history.derive_stop_reason("m-cashew-plant", rec["started_at"])
    assert "exited with code 1" in reason
    assert "OUT OF MEMORY" in reason


def test_stop_reason_names_oom_kill(history_env):
    history = history_env
    rec = history.start_session(dataset="m-bigearthnet", max_runs=3, run_minutes=5)
    logging_utils.log_event(
        "training_failed", dataset="m-bigearthnet", returncode=-9,
        reason=training.describe_exit_code(-9),
        stderr="Loaded validation split: 1000 samples, images (1000, 120, 120, 12), labels (1000, 43)\n",
    )
    reason = history.derive_stop_reason("m-bigearthnet", rec["started_at"])
    assert "out-of-memory" in reason


# ---------------------------------------------------------------------------
# End to end with the REAL classification template (tiny, CPU)
# ---------------------------------------------------------------------------


def _have_local_eurosat() -> bool:
    try:
        import torch  # noqa: F401
        import timm  # noqa: F401
    except ImportError:
        return False
    return (REPO_ROOT / "data" / "geobench" / "classification_v1.0" / "m-eurosat" / "task_specs.pkl").exists()


@pytest.mark.skipif(not _have_local_eurosat(), reason="needs torch, timm and local m-eurosat")
def test_real_run_checkpoint_is_found_by_stresslab_lossscan_and_final_eval(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setenv("GOTRAIN_LOGS_DIR", str(logs_dir))  # keep the subprocess off the live logs/
    monkeypatch.setattr(logging_utils, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", logs_dir / "events.jsonl")
    monkeypatch.setattr(logging_utils, "STATUS_PATH", logs_dir / "status.json")
    monkeypatch.setattr(ledger, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(ledger, "LEDGER_PATH", logs_dir / "ledger.jsonl")
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")

    record = training.launch_training(
        "m-eurosat", "classification",
        {"pretrained": False, "epochs": 1, "batch_size": 4, "max_train_samples": 8, "max_val_samples": 8, "amp": False},
    )
    template_dir = REPO_ROOT / "runs" / "m-eurosat" / record["run_id"]
    try:
        assert template_dir.is_dir(), "ledger run_id must name the template's run directory"

        from gotrain.dataiq import loss_scan
        from gotrain.stresslab import runner as stresslab_runner

        for resolve in (stresslab_runner._checkpoint, loss_scan._checkpoint, final_eval.resolve_checkpoint):
            assert resolve("m-eurosat", record["run_id"]).exists()

        import torch
        from gotrain.templates import train_classification as tc

        model = tc.build_model("resnet18", False, 13, 10)
        model.load_state_dict(torch.load(final_eval.resolve_checkpoint("m-eurosat", record["run_id"]), map_location="cpu"))
    finally:
        import shutil

        shutil.rmtree(template_dir, ignore_errors=True)
