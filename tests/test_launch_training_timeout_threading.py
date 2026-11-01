"""Confirms launch_training actually threads timeout_seconds into the
subprocess's config as cfg["_timeout_seconds"] for classification/
segmentation task types -- the wiring that lets
train_classification.py/train_segmentation.py give their own data-loading
loop a deadline (see tests/test_training_loading_deadline.py for the
deadline-checking logic itself). Mocks subprocess.run so no real training
happens; only the constructed command line is inspected.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain import ledger, logging_utils
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
    return logs_dir


def _mock_subprocess_run(returncode=0, stdout_json=None):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = json.dumps(stdout_json or {"val_metric": 0.5, "rationale": "mock"})
    result.stderr = ""
    return result


def test_timeout_seconds_is_threaded_into_classification_config(isolated_logs, monkeypatch):
    monkeypatch.setattr(
        training.datasets_tools, "get_dataset_info",
        lambda dataset: {"label_info": {"n_classes": 10}},
    )
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _mock_subprocess_run()

    monkeypatch.setattr(training.subprocess, "run", fake_run)

    training.launch_training("m-eurosat", "classification", {"lr": 0.001}, timeout_seconds=300)

    # cmd is [python, template_path, "--dataset", ds, "--config", json_str]
    config_json = json.loads(captured["cmd"][-1])
    assert config_json["_timeout_seconds"] == 300
    # The real OS-level kill must still be armed too -- this fix is additive,
    # not a replacement for the hard timeout.
    assert captured["kwargs"]["timeout"] == 300


def test_timeout_seconds_is_threaded_into_segmentation_config(isolated_logs, monkeypatch):
    monkeypatch.setattr(
        training.datasets_tools, "get_dataset_info",
        lambda dataset: {"label_info": {"n_classes": 2}},
    )
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _mock_subprocess_run()

    monkeypatch.setattr(training.subprocess, "run", fake_run)

    training.launch_training("m-nz-cattle", "segmentation", {"lr": 0.001}, timeout_seconds=120)

    config_json = json.loads(captured["cmd"][-1])
    assert config_json["_timeout_seconds"] == 120


def test_no_timeout_seconds_given_means_no_deadline_field_forced(isolated_logs, monkeypatch):
    """When launch_training is called without a timeout (e.g. a manual/
    direct call, not through the agent's hook which always injects one),
    _timeout_seconds must not be forced to some fabricated value -- the
    template's own DEFAULTS["_timeout_seconds"]=None takes over, same as
    before this fix (unbounded loading).
    """
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _mock_subprocess_run()

    monkeypatch.setattr(training.subprocess, "run", fake_run)

    training.launch_training("m-eurosat", "fake", {"lr": 0.001})

    # task_type="fake" never gets _timeout_seconds injected at all (it has
    # no real loading loop to budget) -- confirms the gate in
    # launch_training is task_type-scoped, not unconditional.
    config_json = json.loads(captured["cmd"][-1])
    assert "_timeout_seconds" not in config_json
