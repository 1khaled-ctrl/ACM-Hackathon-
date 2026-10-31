"""Tests for gotrain/tools/adaptation.py -- the "Plan B" record: when a
proposal is rejected/blocked by governance, the agent is expected to
explicitly log what it's trying instead and why, rather than silently
moving on. Pure gotrain.logging_utils.log_event usage -- no torch,
strands, or geobench needed.
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import logging_utils
from gotrain.tools import adaptation


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setattr(logging_utils, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", logs_dir / "events.jsonl")
    monkeypatch.setattr(logging_utils, "STATUS_PATH", logs_dir / "status.json")
    return logs_dir


def test_record_adaptation_writes_an_event(isolated_logs):
    record = adaptation.record_adaptation(
        dataset="m-eurosat",
        rejected_proposal="backbone=resnet50, rejected for GPU budget limits",
        new_plan="fell back to the default backbone (resnet18) and increased augmentation instead",
        rationale="resnet50 is blocked; augmentation is an ungated change that still targets the same overfitting signal",
    )
    assert record["dataset"] == "m-eurosat"
    assert "resnet50" in record["rejected_proposal"]
    assert "recorded_at" in record

    events = [
        json.loads(line)
        for line in (isolated_logs / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    adaptation_events = [e for e in events if e["event"] == "adaptation_recorded"]
    assert len(adaptation_events) == 1
    assert adaptation_events[0]["dataset"] == "m-eurosat"
    assert adaptation_events[0]["new_plan"] == record["new_plan"]


def test_record_adaptation_updates_status_last_event(isolated_logs):
    adaptation.record_adaptation(
        dataset="m-pv4ger-seg",
        rejected_proposal="encoder=efficientnet-b3, still pending",
        new_plan="tried class_weights instead",
        rationale="class_weights is ungated and addresses the same imbalance signal",
    )
    status = logging_utils.read_status()
    assert "m-pv4ger-seg" in status["last_event"]
    assert "Adapted after rejection" in status["last_event"]


def test_record_adaptation_is_independent_per_call(isolated_logs):
    adaptation.record_adaptation("m-eurosat", "x rejected", "plan A", "reason A")
    adaptation.record_adaptation("m-eurosat", "y rejected", "plan B", "reason B")

    events = [
        json.loads(line)
        for line in (isolated_logs / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    adaptation_events = [e for e in events if e["event"] == "adaptation_recorded"]
    assert len(adaptation_events) == 2
    assert [e["new_plan"] for e in adaptation_events] == ["plan A", "plan B"]
