"""Tests for dashboard/api/server.py::read_telemetry -- specifically the
GPU utilization field added for the NEXUS telemetry panel (CPU/RAM/GPU
util %/VRAM used-of-total, GPU model name demoted to a secondary label).

torch.cuda.utilization() is reused as-is (no new dependency -- it shells
out to nvidia-smi internally); these tests mock torch.cuda rather than
requiring real CUDA hardware, since this machine has none. The real
no-GPU path (torch.cuda.is_available() False) is exercised for real,
unmocked, since that's what actually runs here.
"""

import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def server():
    sys.modules.pop("dashboard.api.server", None)
    from dashboard.api import server as server_mod

    yield server_mod

    sys.modules.pop("dashboard.api.server", None)


def test_no_cuda_device_reports_gpu_none(server):
    """The real path on a machine with no GPU: must degrade to
    gpu=None, same as before this change, not a half-filled dict.
    """
    telemetry = server.read_telemetry()
    assert "cpu_percent" in telemetry
    assert "ram_percent" in telemetry
    # This machine genuinely has no CUDA device, so this exercises the
    # real (unmocked) fallback path.
    assert telemetry["gpu"] is None


def test_gpu_present_reports_utilization_and_vram(server, monkeypatch):
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.mem_get_info.return_value = (4_300_000_000, 16_000_000_000)
    fake_torch.cuda.get_device_name.return_value = "Tesla T4"
    fake_torch.cuda.utilization.return_value = 68

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    telemetry = server.read_telemetry()

    gpu = telemetry["gpu"]
    assert gpu["name"] == "Tesla T4"
    assert gpu["util_percent"] == 68
    assert gpu["util_unavailable"] is False
    assert gpu["mem_used_gb"] == pytest.approx(11.7, abs=0.1)
    assert gpu["mem_total_gb"] == pytest.approx(16.0, abs=0.1)


def test_gpu_present_but_torch_utilization_fails_falls_back_to_nvidia_smi(server, monkeypatch):
    """torch.cuda.utilization() shells out to nvidia-smi itself and can
    fail independently of CUDA memory queries (e.g. that specific
    subprocess spawn pattern blocked in a managed environment, even
    though CUDA memory queries and a direct nvidia-smi call both work
    fine). Must fall back to calling nvidia-smi directly and succeed,
    rather than giving up and reporting unavailable.
    """
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.mem_get_info.return_value = (4_300_000_000, 16_000_000_000)
    fake_torch.cuda.get_device_name.return_value = "Tesla T4"
    fake_torch.cuda.utilization.side_effect = RuntimeError("nvidia-smi not found")

    fake_completed = mock.MagicMock(stdout="73\n")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: fake_completed)

    telemetry = server.read_telemetry()
    gpu = telemetry["gpu"]
    assert gpu is not None
    assert gpu["name"] == "Tesla T4"
    assert gpu["util_percent"] == 73
    assert gpu["util_unavailable"] is False
    assert gpu["mem_used_gb"] == pytest.approx(11.7, abs=0.1)


def test_gpu_present_but_both_utilization_methods_fail_reports_unavailable(server, monkeypatch):
    """Both torch.cuda.utilization() AND the nvidia-smi fallback fail on
    a GPU that otherwise IS present (VRAM/name resolved) -- util_percent
    stays None, but util_unavailable must be True so the frontend can
    show N/A ("couldn't be read") instead of the generic '--' used for
    "nothing to measure" (no GPU at all). Memory numbers must still be
    reported -- the whole gpu dict is never dropped just because
    utilization specifically is unreadable.
    """
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.mem_get_info.return_value = (4_300_000_000, 16_000_000_000)
    fake_torch.cuda.get_device_name.return_value = "Tesla T4"
    fake_torch.cuda.utilization.side_effect = RuntimeError("nvidia-smi not found")

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        server.subprocess, "run",
        mock.Mock(side_effect=FileNotFoundError("nvidia-smi: command not found")),
    )

    telemetry = server.read_telemetry()
    gpu = telemetry["gpu"]
    assert gpu is not None
    assert gpu["name"] == "Tesla T4"
    assert gpu["util_percent"] is None
    assert gpu["util_unavailable"] is True
    assert gpu["mem_used_gb"] == pytest.approx(11.7, abs=0.1)


def test_nvidia_smi_fallback_parses_real_csv_output_format(server, monkeypatch):
    """nvidia-smi's actual --format=csv,noheader,nounits output includes
    a trailing newline and no units -- confirm the parser handles the
    real shape, not just a bare pre-stripped number.
    """
    fake_completed = mock.MagicMock(stdout=" 42 \n")
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: fake_completed)
    assert server._gpu_utilization_via_nvidia_smi() == 42


def test_nvidia_smi_fallback_returns_none_on_malformed_output(server, monkeypatch):
    fake_completed = mock.MagicMock(stdout="not-a-number\n")
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: fake_completed)
    assert server._gpu_utilization_via_nvidia_smi() is None


def test_torch_import_failure_does_not_break_cpu_ram_reporting(server, monkeypatch):
    """If torch itself can't be imported for any reason, CPU/RAM (from
    psutil, a separate try/except) must still be reported -- telemetry
    degrades field-by-field, not all-or-nothing.
    """
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated: torch not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    telemetry = server.read_telemetry()

    assert telemetry["gpu"] is None
    assert telemetry["cpu_percent"] is not None
    assert telemetry["ram_percent"] is not None
