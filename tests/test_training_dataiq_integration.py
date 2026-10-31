"""Tests for gotrain/tools/training.py::launch_training's integration
with fine-tuning-strategy validation and DataIQ manifests -- both checked
before any subprocess is spawned, so these are safe without torch/GPU/data.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import approval, control, ledger, logging_utils
from gotrain.config_schema import ConfigValidationError
from gotrain.dataiq import decision, manifest
from gotrain.tools import final_eval, training


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setattr(logging_utils, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", logs_dir / "events.jsonl")
    monkeypatch.setattr(logging_utils, "STATUS_PATH", logs_dir / "status.json")
    monkeypatch.setattr(ledger, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(ledger, "LEDGER_PATH", logs_dir / "ledger.jsonl")
    monkeypatch.setattr(final_eval, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(control, "CONTROL_PATH", logs_dir / "control.json")
    monkeypatch.setattr(approval, "APPROVALS_STATE_PATH", logs_dir / "approvals_state.json")
    monkeypatch.setattr(approval, "APPROVALS_LOG_PATH", logs_dir / "approvals.jsonl")
    monkeypatch.setattr(manifest, "MANIFESTS_DIR", logs_dir / "dataiq" / "manifests")
    return logs_dir


# ---------------------------------------------------------------------------
# fine_tuning_strategy pre-flight validation
# ---------------------------------------------------------------------------


def test_launch_training_rejects_linear_probe_without_pretrained(isolated_logs):
    with pytest.raises(ConfigValidationError):
        training.launch_training(
            "m-eurosat", "classification",
            {"backbone": "resnet18", "pretrained": False, "fine_tuning_strategy": "linear_probe"},
        )
    assert training.query_history("m-eurosat") == []


def test_launch_training_rejects_partial_without_pretrained(isolated_logs):
    with pytest.raises(ConfigValidationError):
        training.launch_training(
            "m-eurosat", "classification",
            {"backbone": "resnet18", "pretrained": False, "fine_tuning_strategy": "partial"},
        )


def test_launch_training_allows_from_scratch_without_pretrained(isolated_logs):
    # from_scratch + pretrained=False is the expected, sensible combination.
    try:
        training.launch_training(
            "m-eurosat", "classification",
            {"backbone": "resnet18", "pretrained": False, "fine_tuning_strategy": "from_scratch"},
        )
    except ConfigValidationError:
        pytest.fail("from_scratch with pretrained=False must not be rejected")
    except Exception:
        pass  # any failure past validation is expected (no torch/data here)


def test_launch_training_allows_full_with_pretrained(isolated_logs):
    record = training.launch_training(
        "m-eurosat", "fake", {"pretrained": True, "fine_tuning_strategy": "full"},
    )
    assert record["val_metric"] is not None


def test_launch_training_rejects_unknown_fine_tuning_strategy(isolated_logs):
    with pytest.raises(ConfigValidationError):
        training.launch_training("m-eurosat", "classification", {"fine_tuning_strategy": "half_baked"})


# ---------------------------------------------------------------------------
# manifest_id pre-flight validation
# ---------------------------------------------------------------------------


def test_launch_training_rejects_unknown_manifest_id(isolated_logs):
    with pytest.raises(manifest.ManifestError):
        training.launch_training("m-eurosat", "fake", {"manifest_id": "does-not-exist"})
    assert training.query_history("m-eurosat") == []


def test_launch_training_accepts_none_and_base_manifest(isolated_logs):
    r1 = training.launch_training("m-eurosat", "fake", {"manifest_id": None})
    assert r1["val_metric"] is not None
    r2 = training.launch_training(
        "m-eurosat", "fake", {"manifest_id": "base"}, parent_id=r1["run_id"],
        hypothesis={"change": "explicit base manifest", "why": "sanity check", "expected_direction": "no_change"},
    )
    assert r2["val_metric"] is not None


def test_launch_training_accepts_a_real_approved_manifest(isolated_logs):
    proposal = decision.propose_action(
        "m-eurosat", sample_id=99, action="EXCLUDE_FROM_TRAINING_CANDIDATE",
        evidence={"loss": 5.1}, severity="medium", reason="suspected mislabel",
    )
    approval.decide(proposal["key"], approve=True, decided_by="engineer")
    created = manifest.create_manifest("m-eurosat", excluded_sample_ids=[99], reason="approved exclusion")

    baseline = training.launch_training("m-eurosat", "fake", {"manifest_id": None})
    experiment = training.launch_training(
        "m-eurosat", "fake", {"manifest_id": created["manifest_id"]}, parent_id=baseline["run_id"],
        hypothesis={
            "change": "excluded sample 99 (suspected mislabel) via manifest",
            "why": "high training loss + high-confidence wrong prediction suggested a label problem",
            "expected_direction": "increase",
        },
    )
    assert experiment["config"]["manifest_id"] == created["manifest_id"]
    # the controlled comparison is visible in the ledger as a one-variable config diff
    assert experiment["config_diff"] == {"manifest_id": {"from": None, "to": created["manifest_id"]}}
