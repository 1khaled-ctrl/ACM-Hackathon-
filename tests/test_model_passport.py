"""Tests for gotrain/model_passport.py -- pure file-based assembly over
already-recorded ledger/registry/approval/geoguard/dataiq data. No torch
or geobench data needed: everything a passport reads is plain JSON this
project already writes elsewhere.
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import approval, ledger, logging_utils, model_passport, registry
from gotrain.dataiq import decision as dataiq_decision
from gotrain.dataiq import manifest as dataiq_manifest
from gotrain.geoguard import report as geoguard_report


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    models_dir = tmp_path / "models"

    monkeypatch.setattr(logging_utils, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", logs_dir / "events.jsonl")
    monkeypatch.setattr(logging_utils, "STATUS_PATH", logs_dir / "status.json")
    monkeypatch.setattr(ledger, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(ledger, "LEDGER_PATH", logs_dir / "ledger.jsonl")
    monkeypatch.setattr(registry, "MODELS_DIR", models_dir)
    monkeypatch.setattr(approval, "APPROVALS_STATE_PATH", logs_dir / "approvals_state.json")
    monkeypatch.setattr(approval, "APPROVALS_LOG_PATH", logs_dir / "approvals.jsonl")
    monkeypatch.setattr(dataiq_manifest, "MANIFESTS_DIR", logs_dir / "dataiq" / "manifests")
    monkeypatch.setattr(geoguard_report, "GEOGUARD_DIR", logs_dir / "geoguard")
    monkeypatch.setattr(model_passport, "EVENTS_PATH", logs_dir / "events.jsonl")
    monkeypatch.setattr(model_passport, "PASSPORTS_DIR", logs_dir / "model_passports")
    return {"logs_dir": logs_dir, "models_dir": models_dir}


def test_generate_passport_requires_a_submitted_run(isolated_env):
    ledger.add_run("m-eurosat", {"backbone": "resnet18"}, val_metric=0.9, task_type="classification")
    with pytest.raises(ValueError):
        model_passport.generate_passport("m-eurosat")


def test_generate_passport_defaults_to_submitted_run(isolated_env):
    record = ledger.add_run("m-eurosat", {"backbone": "resnet18", "pretrained": True}, val_metric=0.9, task_type="classification")
    ledger.update_run(record["run_id"], submitted_final=True)

    passport = model_passport.generate_passport("m-eurosat")
    assert passport["model_id"] == record["run_id"]
    assert passport["architecture"] == "resnet18"
    assert passport["validation_metrics"]["val_metric"] == 0.9
    assert passport["engineer_approval"]["submitted_final"] is True
    assert passport["robustness_results"] is None  # never fabricated


def test_generate_passport_for_explicit_run_id(isolated_env):
    record = ledger.add_run("m-eurosat", {"backbone": "resnet18"}, val_metric=0.9, task_type="classification")
    passport = model_passport.generate_passport("m-eurosat", run_id=record["run_id"])
    assert passport["model_id"] == record["run_id"]


def test_generate_passport_rejects_unknown_run_id(isolated_env):
    with pytest.raises(ValueError):
        model_passport.generate_passport("m-eurosat", run_id="does-not-exist")


def test_checkpoint_sha256_is_computed_from_registered_version(isolated_env):
    version = registry.register_run(
        dataset="m-eurosat", run_id="r1",
        run_dir=str(_make_run_dir(isolated_env["models_dir"].parent)),
        checkpoint_path=str(_make_checkpoint(isolated_env["models_dir"].parent)),
    )
    record = ledger.add_run(
        "m-eurosat", {"backbone": "resnet18"}, val_metric=0.9, task_type="classification",
        run_id="r1", version_id=version,
    )
    ledger.update_run(record["run_id"], submitted_final=True)

    passport = model_passport.generate_passport("m-eurosat")
    assert passport["checkpoint_version"] == version
    assert passport["checkpoint_sha256"] is not None
    assert len(passport["checkpoint_sha256"]) == 64  # sha256 hex digest length


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "m-eurosat" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    return run_dir


def _make_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "runs" / "m-eurosat" / "r1" / "best_model.pt"
    checkpoint.write_bytes(b"fake checkpoint bytes")
    return checkpoint


def test_data_provenance_reflects_manifest_and_approved_actions(isolated_env):
    proposal = dataiq_decision.propose_action(
        "m-eurosat", sample_id=42, action="EXCLUDE_FROM_TRAINING_CANDIDATE",
        evidence={"loss": 5.0}, severity="medium", reason="suspected mislabel",
    )
    approval.decide(proposal["key"], approve=True, decided_by="engineer", decision_reason="confirmed via manual review")
    manifest = dataiq_manifest.create_manifest("m-eurosat", excluded_sample_ids=[42], reason="approved exclusion")

    record = ledger.add_run(
        "m-eurosat", {"backbone": "resnet18", "manifest_id": manifest["manifest_id"]},
        val_metric=0.92, task_type="classification",
    )
    ledger.update_run(record["run_id"], submitted_final=True)

    passport = model_passport.generate_passport("m-eurosat")
    provenance = passport["data_provenance"]
    assert provenance["training_manifest_id"] == manifest["manifest_id"]
    assert provenance["excluded_sample_count"] == 1
    assert provenance["excluded_sample_ids"] == [42]
    assert len(provenance["approved_data_actions"]) == 1
    assert provenance["approved_data_actions"][0]["sample_id"] == 42


def test_data_provenance_defaults_to_base_manifest_when_none_used(isolated_env):
    record = ledger.add_run("m-eurosat", {"backbone": "resnet18"}, val_metric=0.9, task_type="classification")
    ledger.update_run(record["run_id"], submitted_final=True)
    passport = model_passport.generate_passport("m-eurosat")
    assert passport["data_provenance"]["excluded_sample_count"] == 0


def test_known_limitations_flags_missing_test_result(isolated_env):
    record = ledger.add_run("m-eurosat", {"backbone": "resnet18"}, val_metric=0.9, task_type="classification")
    ledger.update_run(record["run_id"], submitted_final=True)
    passport = model_passport.generate_passport("m-eurosat")
    assert any("test-split results" in lim.lower() for lim in passport["known_limitations"])


def test_known_limitations_flags_small_validation_set_for_nz_cattle(isolated_env):
    record = ledger.add_run("m-nz-cattle", {"architecture": "unet"}, val_metric=0.5, task_type="segmentation")
    ledger.update_run(record["run_id"], submitted_final=True)
    passport = model_passport.generate_passport("m-nz-cattle")
    assert any("validation set is very small" in lim.lower() for lim in passport["known_limitations"])


def test_write_and_read_passport_round_trip(isolated_env):
    record = ledger.add_run("m-eurosat", {"backbone": "resnet18"}, val_metric=0.9, task_type="classification")
    ledger.update_run(record["run_id"], submitted_final=True)
    passport = model_passport.generate_passport("m-eurosat")

    model_passport.write_passport("m-eurosat", passport)
    loaded = model_passport.read_passport("m-eurosat")
    assert loaded == passport
    assert model_passport.read_passport("m-so2sat") is None


def test_format_passport_markdown_includes_key_sections(isolated_env):
    record = ledger.add_run("m-eurosat", {"backbone": "resnet18"}, val_metric=0.9, task_type="classification")
    ledger.update_run(record["run_id"], submitted_final=True)
    passport = model_passport.generate_passport("m-eurosat")

    text = model_passport.format_passport_markdown(passport)
    assert "# Model Passport" in text
    assert "## Data Provenance and Quality" in text
    assert "## Known Limitations" in text
    assert "## Robustness Results" in text


def test_format_passport_html_renders_self_contained_document(isolated_env):
    record = ledger.add_run("m-eurosat", {"backbone": "resnet18"}, val_metric=0.9, task_type="classification")
    ledger.update_run(record["run_id"], submitted_final=True)
    passport = model_passport.generate_passport("m-eurosat")

    html = model_passport.format_passport_html(passport)
    assert "<!doctype html>" in html.lower()
    assert "m-eurosat Passport" in html
    assert record["run_id"] in html
    assert "Data Provenance and Quality" in html
    assert "Known Limitations" in html
    # theme-aware tokens present for all three viewer states
    assert ":root {" in html
    assert "prefers-color-scheme: dark" in html
    assert 'data-theme="dark"' in html


def test_format_passport_html_never_fabricates_missing_fields(isolated_env):
    # A passport generated before GeoGuard ever ran for this dataset has
    # no sensor/band info -- must render honestly, not invent a value.
    record = ledger.add_run("m-eurosat", {"backbone": "resnet18"}, val_metric=0.9, task_type="classification")
    ledger.update_run(record["run_id"], submitted_final=True)
    passport = model_passport.generate_passport("m-eurosat")
    assert passport["sensor"] is None  # no geoguard report in this isolated env

    html = model_passport.format_passport_html(passport)
    assert "StressLab not implemented" in html  # robustness honestly absent, not invented
    assert "Not yet evaluated" in html  # no test result, since none was run
