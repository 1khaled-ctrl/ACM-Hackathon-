"""Tests for DataIQ's versioned, reversible training-data manifests
(gotrain/dataiq/manifest.py) and their interaction with the data-quality
decision queue (gotrain/dataiq/decision.py). Pure file-based logic -- no
torch/geobench I/O, no actual data is ever touched.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import approval, logging_utils
from gotrain.dataiq import decision, manifest


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setattr(logging_utils, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", logs_dir / "events.jsonl")
    monkeypatch.setattr(logging_utils, "STATUS_PATH", logs_dir / "status.json")
    monkeypatch.setattr(approval, "APPROVALS_STATE_PATH", logs_dir / "approvals_state.json")
    monkeypatch.setattr(approval, "APPROVALS_LOG_PATH", logs_dir / "approvals.jsonl")
    monkeypatch.setattr(manifest, "MANIFESTS_DIR", logs_dir / "dataiq" / "manifests")
    return logs_dir


# ---------------------------------------------------------------------------
# Base manifest / existence checks
# ---------------------------------------------------------------------------


def test_base_manifest_has_no_exclusions(isolated_logs):
    base = manifest.get_manifest("m-eurosat", manifest.BASE_MANIFEST_ID)
    assert base["excluded_sample_ids"] == []
    assert manifest.get_excluded_ids("m-eurosat", manifest.BASE_MANIFEST_ID) == set()


def test_validate_manifest_exists_allows_none_and_base(isolated_logs):
    manifest.validate_manifest_exists("m-eurosat", None)
    manifest.validate_manifest_exists("m-eurosat", "base")  # must not raise


def test_validate_manifest_exists_rejects_unknown_id(isolated_logs):
    with pytest.raises(manifest.ManifestError):
        manifest.validate_manifest_exists("m-eurosat", "does-not-exist")


# ---------------------------------------------------------------------------
# create_manifest requires prior approval
# ---------------------------------------------------------------------------


def test_create_manifest_rejects_unapproved_exclusion(isolated_logs):
    with pytest.raises(manifest.ManifestApprovalError):
        manifest.create_manifest("m-eurosat", excluded_sample_ids=[42], reason="looks mislabeled")


def test_create_manifest_rejects_pending_exclusion(isolated_logs):
    decision.propose_action(
        "m-eurosat", sample_id=42, action="EXCLUDE_FROM_TRAINING_CANDIDATE",
        evidence={}, severity="medium", reason="suspected label noise",
    )
    with pytest.raises(manifest.ManifestApprovalError):
        manifest.create_manifest("m-eurosat", excluded_sample_ids=[42], reason="looks mislabeled")


def test_create_manifest_succeeds_after_approval(isolated_logs):
    proposal = decision.propose_action(
        "m-eurosat", sample_id=42, action="EXCLUDE_FROM_TRAINING_CANDIDATE",
        evidence={}, severity="medium", reason="suspected label noise",
    )
    approval.decide(proposal["key"], approve=True, decided_by="engineer")

    created = manifest.create_manifest("m-eurosat", excluded_sample_ids=[42], reason="approved exclusion")
    assert created["excluded_sample_ids"] == [42]
    assert created["newly_excluded_sample_ids"] == [42]
    assert created["base_manifest_id"] == manifest.BASE_MANIFEST_ID

    assert manifest.get_excluded_ids("m-eurosat", created["manifest_id"]) == {42}
    manifest.validate_manifest_exists("m-eurosat", created["manifest_id"])  # must not raise


def test_create_manifest_succeeds_for_auto_approved_quarantine(isolated_logs):
    # QUARANTINE_INVALID_SAMPLE is auto-approved under policy -- no human decision needed.
    decision.propose_action(
        "m-eurosat", sample_id=7, action="QUARANTINE_INVALID_SAMPLE",
        evidence={"problems": ["1 NaN pixel"]}, severity="high", reason="objectively invalid",
    )
    created = manifest.create_manifest("m-eurosat", excluded_sample_ids=[7], reason="auto-quarantine")
    assert created["excluded_sample_ids"] == [7]


def test_create_manifest_rejects_rejected_exclusion(isolated_logs):
    proposal = decision.propose_action(
        "m-eurosat", sample_id=42, action="EXCLUDE_FROM_TRAINING_CANDIDATE",
        evidence={}, severity="medium", reason="suspected label noise",
    )
    approval.decide(proposal["key"], approve=False, decided_by="engineer", decision_reason="looks like a legitimate hard example")
    with pytest.raises(manifest.ManifestApprovalError):
        manifest.create_manifest("m-eurosat", excluded_sample_ids=[42], reason="trying anyway")


# ---------------------------------------------------------------------------
# Manifest composition / lineage
# ---------------------------------------------------------------------------


def test_manifests_compose_on_top_of_a_base_manifest(isolated_logs):
    for sample_id in (1, 2, 3):
        decision.propose_action(
            "m-eurosat", sample_id=sample_id, action="QUARANTINE_INVALID_SAMPLE",
            evidence={}, severity="high", reason="corrupted",
        )
    first = manifest.create_manifest("m-eurosat", excluded_sample_ids=[1, 2], reason="first batch of quarantines")

    second = manifest.create_manifest(
        "m-eurosat", excluded_sample_ids=[3], reason="one more", base_manifest_id=first["manifest_id"],
    )
    assert second["base_manifest_id"] == first["manifest_id"]
    assert second["excluded_sample_ids"] == [1, 2, 3]
    assert second["newly_excluded_sample_ids"] == [3]


def test_list_manifests_includes_base_first(isolated_logs):
    decision.propose_action(
        "m-eurosat", sample_id=1, action="QUARANTINE_INVALID_SAMPLE", evidence={}, severity="high", reason="x",
    )
    manifest.create_manifest("m-eurosat", excluded_sample_ids=[1], reason="x")
    manifests = manifest.list_manifests("m-eurosat")
    assert manifests[0]["manifest_id"] == manifest.BASE_MANIFEST_ID
    assert len(manifests) == 2


def test_manifests_are_independent_per_dataset(isolated_logs):
    decision.propose_action(
        "m-eurosat", sample_id=1, action="QUARANTINE_INVALID_SAMPLE", evidence={}, severity="high", reason="x",
    )
    created = manifest.create_manifest("m-eurosat", excluded_sample_ids=[1], reason="x")
    assert manifest.list_manifests("m-so2sat") == [manifest.get_manifest("m-so2sat", manifest.BASE_MANIFEST_ID)]
    with pytest.raises(manifest.ManifestError):
        manifest.validate_manifest_exists("m-so2sat", created["manifest_id"])


# ---------------------------------------------------------------------------
# Never touches validation/test -- structural guarantee, not just a test
# ---------------------------------------------------------------------------


def test_manifest_public_functions_take_no_split_argument():
    """DataIQ manifests only ever reference TRAIN-split indices -- no
    public function in this module accepts a split name, so there is no
    way to even ask it to act on validation/test data.
    """
    import inspect

    public_functions = [
        manifest.get_manifest, manifest.validate_manifest_exists, manifest.list_manifests,
        manifest.get_excluded_ids, manifest.create_manifest,
    ]
    for fn in public_functions:
        params = set(inspect.signature(fn).parameters)
        assert "split" not in params, f"{fn.__name__} must not accept a split argument"
