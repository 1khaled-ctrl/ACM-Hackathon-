"""Suite-wide isolation from the real logs/ and models/.

Some tests run the real training templates as subprocesses (wherever torch
and the local GEO-Bench data exist). Those subprocesses read
GOTRAIN_LOGS_DIR at import, and launch_training registers their checkpoints
in gotrain.registry.MODELS_DIR -- so without this, a pytest run wrote phantom
progress into the live dashboard's logs/status.json + live_metrics.jsonl and
added junk versions to models/<dataset>/ (the rollback list). Tests that
need specific paths still override these with their own fixtures.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_subprocess_logs_and_registry(tmp_path_factory, monkeypatch):
    from gotrain import registry

    root = tmp_path_factory.mktemp("isolated")
    monkeypatch.setenv("GOTRAIN_LOGS_DIR", str(root / "logs"))
    monkeypatch.setattr(registry, "MODELS_DIR", root / "models")
