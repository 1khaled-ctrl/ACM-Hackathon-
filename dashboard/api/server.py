"""GoTrain NEXUS -- FastAPI bridge between the Python agent and the React UI.

Single-server design: this process serves BOTH the built React frontend
(dashboard/frontend/dist) and the JSON/WebSocket API, on one port. That
keeps the SageMaker Studio proxy URL simple (one /proxy/<port>/ prefix)
and means only one process has to be kept alive on the lab.

It is a *bridge*, not a second brain. Every piece of state it exposes is
read from the existing GoTrain log files, and every control it offers
calls straight into gotrain.control -- the same state machine that
gotrain/agent.py's hook and gotrain/tools/training.py already enforce.
There is deliberately no second control mechanism here.

Test-split safety: this module never imports gotrain.tools.final_eval and
never reads logs/test_lock_*.json. The only evaluation values it can show
are whatever the ledger/live-metrics already recorded from VALIDATION.
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Same default as run_gotrain.py -- without this, geobench falls back to its
# own GEO_BENCH_DIR and /api/datasets comes back empty even though the data
# is sitting in <repo>/data/geobench. setdefault, so an explicit
# GEO_BENCH_DIR in the environment still wins (e.g. on the lab).
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gotrain import control, ledger
from gotrain import history as history_mod
from gotrain.logging_utils import EVENTS_PATH, LIVE_METRICS_PATH, LOGS_DIR
from gotrain.status import get_status

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
AGENT_PID_PATH = LOGS_DIR / "agent.pid"
AGENT_LOG_PATH = LOGS_DIR / "agent.log"

app = FastAPI(title="GoTrain NEXUS")


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Log readers. Every number the UI shows originates here.
# ---------------------------------------------------------------------------

def read_jsonl(path: Path, tail: Optional[int] = None) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A training subprocess may be mid-write on the last line.
                continue
    return rows[-tail:] if tail else rows


def compute_eta_seconds(current_run: dict) -> Optional[float]:
    epoch = current_run.get("epoch")
    epochs = current_run.get("epochs")
    elapsed = current_run.get("elapsed_seconds")
    if not epoch or not epochs or elapsed is None or epoch <= 0:
        return None
    return (elapsed / epoch) * max(epochs - epoch, 0)


def read_telemetry() -> dict:
    """Host CPU/RAM (and GPU when torch sees one). Best-effort: the UI
    degrades to '--' rather than erroring if anything is unavailable.
    """
    out = {"cpu_percent": None, "ram_percent": None, "gpu": None}
    try:
        import psutil

        out["cpu_percent"] = psutil.cpu_percent(interval=None)
        out["ram_percent"] = psutil.virtual_memory().percent
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            out["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "mem_used_gb": round((total_b - free_b) / 1e9, 2),
                "mem_total_gb": round(total_b / 1e9, 2),
            }
    except Exception:
        pass
    return out


def agent_process_alive() -> bool:
    if not AGENT_PID_PATH.exists():
        return False
    try:
        pid = int(AGENT_PID_PATH.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:
        return False


def reconcile_open_session() -> None:
    """Close out a mission-history session whose process is gone.

    Nothing watches for the agent subprocess exiting -- run_gotrain.py
    can't reliably close its own record (it may be killed outright), and
    there is no supervisor. build_snapshot() already runs about once a
    second for the WebSocket, so this piggybacks on that as the place
    where "the process that owned this session is no longer alive" is
    noticed. Best-effort: history bookkeeping must never be able to
    break the live snapshot the dashboard depends on.
    """
    try:
        if agent_process_alive():
            return
        session = history_mod.get_open_session()
        if not session:
            return
        # Infer how it ended from the control plane: COMPLETED means the
        # agent submitted a final model; FAILED_SAFE means a verification
        # step failed; anything else means the process went away without
        # finishing (killed via Stop, crashed on startup, etc.).
        state = control.get_state()
        if state == control.COMPLETED:
            status = history_mod.COMPLETED
        elif state == control.FAILED_SAFE:
            status = history_mod.FAILED
        else:
            status = history_mod.STOPPED
        history_mod.finish_session(session["session_id"], status=status, ended_in_state=state)
    except Exception:
        pass


def build_snapshot() -> dict:
    reconcile_open_session()
    status_data = get_status()
    current_run = status_data.get("current_run") or {}
    try:
        history = ledger.get_history()
    except Exception:
        history = []
    live_rows = read_jsonl(LIVE_METRICS_PATH)
    run_id = current_run.get("run_id")
    run_rows = [r for r in live_rows if r.get("run_id") == run_id] if run_id else []

    return {
        "control_state": control.get_state(),
        "control": control.get_control(),
        "agent_running": agent_process_alive(),
        "current_run": current_run,
        "eta_seconds": compute_eta_seconds(current_run),
        "status": status_data,
        "history": history,
        "live_metrics": run_rows or live_rows[-200:],
        "events": read_jsonl(EVENTS_PATH, tail=40)[::-1],
        "telemetry": read_telemetry(),
        "logs_dir": str(LOGS_DIR),
    }


# ---------------------------------------------------------------------------
# REST API.
# ---------------------------------------------------------------------------

@app.get("/api/snapshot")
def api_snapshot() -> dict:
    return build_snapshot()


@app.get("/api/datasets")
def api_datasets() -> dict:
    """Locally available GEO-Bench datasets, with metadata. Never opens
    the test split -- get_dataset_info reports it as 'locked'.
    """
    try:
        from gotrain.tools.datasets import get_dataset_info, list_datasets

        names = list_datasets()
    except Exception as exc:
        return {"datasets": [], "error": str(exc)}

    out = []
    for name in names:
        try:
            out.append(get_dataset_info(name))
        except Exception as exc:
            out.append({"name": name, "error": str(exc)})
    return {"datasets": out}


class TransitionRequest(BaseModel):
    action: str


# Only engineer-initiated moves are exposed. Resume goes through
# RESUME_REQUESTED (never straight to RUNNING_AUTONOMOUSLY) so the agent
# loop performs the final transition itself, exactly as control.py intends.
ACTION_MAP = {
    "pause": (control.PAUSE_REQUESTED, "engineer clicked pause (NEXUS)"),
    "manual": (control.MANUAL_CONTROL, "engineer took manual control (NEXUS)"),
    "resume": (control.RESUME_REQUESTED, "engineer clicked resume (NEXUS)"),
    "cancel_pause": (control.RUNNING_AUTONOMOUSLY, "engineer cancelled pause (NEXUS)"),
    "acknowledge_failure": (control.PAUSED_FOR_ENGINEER, "engineer acknowledged failure (NEXUS)"),
}


@app.post("/api/control")
def api_control(req: TransitionRequest) -> dict:
    if req.action not in ACTION_MAP:
        raise HTTPException(status_code=400, detail=f"unknown action {req.action!r}")
    new_state, reason = ACTION_MAP[req.action]
    try:
        data = control.transition(new_state, reason=reason, requested_by="engineer")
    except control.IllegalTransitionError as exc:
        # A stale UI can race the agent; surface it as a clean 409 rather
        # than a 500 so the frontend can just re-sync and re-render.
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "control": data}


# ---------------------------------------------------------------------------
# Manual-control actions: approvals + rollback. Thin passthroughs to
# gotrain.approval / gotrain.rollback -- the same functions the old
# Streamlit dashboard called directly. No new business logic here.
# ---------------------------------------------------------------------------

@app.get("/api/approvals")
def api_approvals(status: Optional[str] = None) -> dict:
    from gotrain import approval

    return {"proposals": approval.list_proposals(status=status)}


class ApprovalDecisionRequest(BaseModel):
    key: str
    approve: bool
    decision_reason: Optional[str] = None


@app.post("/api/approvals/decide")
def api_approvals_decide(req: ApprovalDecisionRequest) -> dict:
    from gotrain import approval

    try:
        record = approval.decide(
            req.key, req.approve, decided_by="engineer", decision_reason=req.decision_reason
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True, "proposal": record}


@app.get("/api/rollback/versions")
def api_rollback_versions(dataset: str) -> dict:
    from gotrain import registry
    from gotrain.rollback import get_active_version

    try:
        versions = registry.list_versions(dataset)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"dataset": dataset, "active_version": get_active_version(dataset), "versions": versions}


class RollbackRequest(BaseModel):
    dataset: str
    version: str
    task_type: str


@app.post("/api/rollback/execute")
def api_rollback_execute(req: RollbackRequest) -> dict:
    from gotrain.rollback import RollbackConflictError, RollbackVerificationError, execute_rollback

    try:
        result = execute_rollback(req.dataset, req.version, req.task_type, requested_by="engineer")
    except RollbackConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except RollbackVerificationError as exc:
        # execute_rollback already drove control to FAILED_SAFE; surface
        # the reason but this is not a "retry the request" style error.
        raise HTTPException(status_code=422, detail=str(exc))
    return {"ok": True, "result": result}


class StartRequest(BaseModel):
    dataset: str
    max_runs: int = 6
    run_minutes: float = 5
    max_tool_calls: int = 40


@app.post("/api/agent/start")
def api_agent_start(req: StartRequest) -> dict:
    """Launch run_gotrain.py as a detached subprocess.

    Guards: refuses if an agent process is already alive (so a double
    click can't start two agents racing on the same ledger), and refuses
    an unknown dataset name rather than passing arbitrary strings to the
    subprocess.
    """
    if agent_process_alive():
        raise HTTPException(status_code=409, detail="An agent process is already running.")

    try:
        from gotrain.tools.datasets import list_datasets

        available = list_datasets()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"cannot list datasets: {exc}")
    if req.dataset not in available:
        raise HTTPException(status_code=400, detail=f"unknown dataset {req.dataset!r}; available: {available}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_gotrain.py"),
        "--dataset", req.dataset,
        "--max-runs", str(req.max_runs),
        "--run-minutes", str(req.run_minutes),
        "--max-tool-calls", str(req.max_tool_calls),
    ]
    logfile = open(AGENT_LOG_PATH, "a", encoding="utf-8")
    popen_kwargs = {"stdout": logfile, "stderr": subprocess.STDOUT, "cwd": str(REPO_ROOT)}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(cmd, **popen_kwargs)
    AGENT_PID_PATH.write_text(str(proc.pid), encoding="utf-8")

    # Mark the moment this launch was requested. run_gotrain.py's own
    # startup (import Strands, build the Bedrock model, first tool call)
    # takes several seconds -- during that window events.jsonl's most
    # recent entries are still whatever the PREVIOUS session left behind
    # (which can include "final_model_submitted" if it completed). Without
    # this marker the UI has no way to tell "stale tail of an old session"
    # apart from "the new session already did something", which reads as
    # the agent finishing instantly. The frontend greys out/labels events
    # older than launched_at as "before this launch".
    from gotrain.logging_utils import write_status

    # Also clear current_run here, not just rely on gotrain/tools/training.py
    # clearing it when a launch_training subprocess exits -- if the agent
    # itself dies before ever reaching launch_training (e.g. it can't
    # construct the Bedrock model), that clearing code never runs, and a
    # NEW session would otherwise inherit a PREVIOUS session's leftover
    # "epoch 7/10" numbers with nothing to say they're stale.
    # Open a mission-history record for this launch (gotrain/history.py).
    # Written here rather than inside run_gotrain.py so that a subprocess
    # which dies during startup still leaves a visible record of having
    # been attempted, instead of vanishing with no trace in the UI.
    session = history_mod.start_session(
        dataset=req.dataset,
        max_runs=req.max_runs,
        run_minutes=req.run_minutes,
        max_tool_calls=req.max_tool_calls,
        pid=proc.pid,
    )

    write_status(
        launch={
            "dataset": req.dataset,
            "pid": proc.pid,
            "launched_at": _now_iso(),
            "session_id": session["session_id"],
        },
        current_run=None,
    )

    return {"ok": True, "pid": proc.pid, "dataset": req.dataset, "cmd": cmd}


@app.post("/api/agent/stop")
def api_agent_stop() -> dict:
    """Stop the agent process.

    Prefer pausing via the control plane for a graceful stop -- this is
    the hard kill for when a process needs to actually go away.
    """
    if not AGENT_PID_PATH.exists():
        raise HTTPException(status_code=404, detail="No agent pid file; nothing to stop.")
    try:
        pid = int(AGENT_PID_PATH.read_text(encoding="utf-8").strip())
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"unreadable pid file: {exc}")

    try:
        import psutil

        if psutil.pid_exists(pid):
            proc = psutil.Process(pid)
            for child in proc.children(recursive=True):
                child.terminate()
            proc.terminate()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to stop pid {pid}: {exc}")

    try:
        AGENT_PID_PATH.unlink()
    except OSError:
        pass
    return {"ok": True, "stopped_pid": pid}


@app.get("/api/history")
def api_history(limit: int = 50) -> dict:
    """Every mission launched from the dashboard, newest first, with
    summary metrics derived from the ledger (see gotrain/history.py).
    """
    return {"sessions": history_mod.list_sessions(limit=limit)}


@app.get("/api/history/{session_id}")
def api_history_detail(session_id: str) -> dict:
    """One mission in full: summary plus the runs, per-epoch curves and
    events that fall inside its time window.
    """
    record = history_mod.load_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    return record


@app.get("/api/agent/log")
def api_agent_log(tail: int = 200) -> dict:
    if not AGENT_LOG_PATH.exists():
        return {"lines": []}
    with open(AGENT_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return {"lines": [ln.rstrip("\n") for ln in lines[-tail:]]}


# ---------------------------------------------------------------------------
# WebSocket: push a snapshot ~1/s so the UI never polls.
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(build_snapshot())
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
    except Exception:
        # Never let one bad frame take the server down; the client
        # reconnects on close.
        return


# ---------------------------------------------------------------------------
# Serve the built React app. Mounted last so /api and /ws win.
# ---------------------------------------------------------------------------

if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/")
    def serve_index():
        return FileResponse(str(FRONTEND_DIST / "index.html"))

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        candidate = FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
else:
    @app.get("/")
    def no_frontend():
        return JSONResponse(
            status_code=503,
            content={
                "error": "frontend not built",
                "fix": "cd dashboard/frontend && npm install && npm run build",
            },
        )


def main() -> None:
    import uvicorn

    port = int(os.environ.get("NEXUS_PORT", "8501"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
