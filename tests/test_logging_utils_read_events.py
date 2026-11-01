"""Tests for gotrain.logging_utils.read_events -- the event-stream reader
added alongside log_event, which previously had no corresponding reader.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import logging_utils


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_utils, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(logging_utils, "EVENTS_PATH", tmp_path / "events.jsonl")
    return tmp_path


def test_read_events_returns_empty_list_when_no_file_exists(isolated_logs):
    assert logging_utils.read_events() == []


def test_read_events_filters_by_event_type_and_dataset(isolated_logs):
    logging_utils.log_event("training_started", dataset="m-eurosat")
    logging_utils.log_event("training_failed", dataset="m-eurosat")
    logging_utils.log_event("training_started", dataset="m-so2sat")

    all_events = logging_utils.read_events()
    assert len(all_events) == 3

    eurosat_only = logging_utils.read_events(dataset="m-eurosat")
    assert len(eurosat_only) == 2

    started_only = logging_utils.read_events(event="training_started")
    assert len(started_only) == 2

    eurosat_started = logging_utils.read_events(event="training_started", dataset="m-eurosat")
    assert len(eurosat_started) == 1
