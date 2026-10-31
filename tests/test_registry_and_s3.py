"""Tests for the model registry (gotrain/registry.py) and the S3 tools
(gotrain/tools/s3.py). No test here requires real AWS access or network —
S3 tests only exercise the graceful-failure path (no bucket configured),
which is the only path exercisable without live AWS credentials.
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gotrain import registry
from gotrain.tools import s3


@pytest.fixture
def isolated_models_dir(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    monkeypatch.setattr(registry, "MODELS_DIR", models_dir)
    return models_dir


@pytest.fixture
def sample_run_dir(tmp_path):
    run_dir = tmp_path / "runs" / "m-eurosat" / "some-run-id"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"lr": 0.001}), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"final_val_metric": 0.9}), encoding="utf-8")
    checkpoint = run_dir / "best_model.pt"
    checkpoint.write_bytes(b"fake checkpoint bytes")
    return run_dir, checkpoint


def test_register_run_creates_v1_then_v2(isolated_models_dir, sample_run_dir):
    run_dir, checkpoint = sample_run_dir

    v1 = registry.register_run("m-eurosat", "run-1", str(run_dir), str(checkpoint))
    assert v1 == "v1"

    v2 = registry.register_run("m-eurosat", "run-2", str(run_dir), str(checkpoint))
    assert v2 == "v2"

    assert registry.list_versions("m-eurosat") == ["v1", "v2"]


def test_register_run_copies_config_metrics_and_checkpoint(isolated_models_dir, sample_run_dir):
    run_dir, checkpoint = sample_run_dir
    version = registry.register_run("m-eurosat", "run-1", str(run_dir), str(checkpoint))

    version_dir = registry.get_version_dir("m-eurosat", version)
    assert (version_dir / "config.json").read_text(encoding="utf-8") == (run_dir / "config.json").read_text(encoding="utf-8")
    assert (version_dir / "metrics.json").exists()
    assert (version_dir / "model.pt").read_bytes() == b"fake checkpoint bytes"

    manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run-1"
    assert manifest["dataset"] == "m-eurosat"
    assert manifest["checkpoint"] == "model.pt"


def test_register_run_skips_missing_checkpoint_gracefully(isolated_models_dir, sample_run_dir):
    run_dir, _ = sample_run_dir
    version = registry.register_run("m-eurosat", "run-1", str(run_dir), checkpoint_path=None)

    version_dir = registry.get_version_dir("m-eurosat", version)
    assert not (version_dir / "model.pt").exists()
    manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint"] is None


def test_versions_are_independent_per_dataset(isolated_models_dir, sample_run_dir):
    run_dir, checkpoint = sample_run_dir
    v1_eurosat = registry.register_run("m-eurosat", "run-1", str(run_dir), str(checkpoint))
    v1_so2sat = registry.register_run("m-so2sat", "run-2", str(run_dir), str(checkpoint))

    assert v1_eurosat == "v1"
    assert v1_so2sat == "v1"  # independent counters per dataset


def test_upload_to_s3_skips_gracefully_without_bucket(monkeypatch):
    monkeypatch.delenv(s3.BUCKET_ENV_VAR, raising=False)
    result = s3.upload_to_s3("m-eurosat", "v1")
    assert result["status"] == "skipped"
    assert s3.BUCKET_ENV_VAR in result["reason"]


def test_download_from_s3_skips_gracefully_without_bucket(monkeypatch):
    monkeypatch.delenv(s3.BUCKET_ENV_VAR, raising=False)
    result = s3.download_from_s3("m-eurosat", "v1")
    assert result["status"] == "skipped"


def test_upload_to_s3_fails_gracefully_not_an_exception(monkeypatch):
    """With a bucket configured but no real AWS access, upload must return
    a structured error, never raise — the agent calls this as a tool and
    needs a result it can read, not an unhandled exception.
    """
    monkeypatch.setenv(s3.BUCKET_ENV_VAR, "a-bucket-that-does-not-exist-anywhere")
    result = s3.upload_to_s3("m-eurosat", "v1")
    assert result["status"] == "error"
    assert "reason" in result
