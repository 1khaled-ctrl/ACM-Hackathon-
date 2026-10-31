"""GoTrain NEXUS -- a live, auto-refreshing view of the real GoTrain run state.

Reads only existing log files under gotrain.logging_utils.LOGS_DIR:
  logs/status.json        -- current_run snapshot (written every epoch)
  logs/control.json       -- engineer control-plane state
  logs/events.jsonl       -- full event stream
  logs/ledger.jsonl       -- one row per completed training run
  logs/live_metrics.jsonl -- one row per epoch, every run (new, append-only)

No separate control mechanism: every button here calls straight into
gotrain.control (the same module gotrain/agent.py's hook and
gotrain/tools/training.py both check before allowing any agent action).
This file does not read or write anything on the test split -- it has no
import of gotrain.tools.final_eval at all.

Run it with:  streamlit run dashboard/app.py
Works from VS Code (local logs/) or from JupyterLab on the lab (same
relative logs/ dir, or point GOTRAIN_LOGS_DIR at a different location) --
see gotrain/logging_utils.py, which this file never bypasses.
"""

import json
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gotrain import control, ledger
from gotrain.logging_utils import EVENTS_PATH, LIVE_METRICS_PATH, LOGS_DIR
from gotrain.status import get_status

REFRESH_SECONDS = 1

st.set_page_config(page_title="GoTrain NEXUS", page_icon="\U0001F6F0", layout="wide")


# ---------------------------------------------------------------------------
# Data loading -- every value on this page comes from one of these.
# ---------------------------------------------------------------------------

def load_jsonl(path: Path, max_lines: int = None) -> list:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if max_lines is not None:
        records = records[-max_lines:]
    return records


def load_history() -> list:
    try:
        return ledger.get_history()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# ETA.
# ---------------------------------------------------------------------------

def compute_eta_seconds(current_run: dict) -> "float | None":
    if not current_run:
        return None
    epoch = current_run.get("epoch")
    epochs = current_run.get("epochs")
    elapsed = current_run.get("elapsed_seconds")
    if not epoch or not epochs or elapsed is None or epoch <= 0:
        return None
    seconds_per_epoch = elapsed / epoch
    remaining_epochs = max(epochs - epoch, 0)
    return seconds_per_epoch * remaining_epochs


def format_duration(seconds: "float | None") -> str:
    if seconds is None:
        return "--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Control panel -- reuses gotrain.control exactly as the main dashboard does.
# ---------------------------------------------------------------------------

CONTROL_STATE_LABELS = {
    control.RUNNING_AUTONOMOUSLY: ("Running Autonomously", "#1a7f37"),
    control.PAUSE_REQUESTED: ("Pause Requested", "#9a6700"),
    control.PAUSED_FOR_ENGINEER: ("Paused", "#0969da"),
    control.MANUAL_CONTROL: ("Manual Control", "#9a6700"),
    control.RESUME_REQUESTED: ("Resuming...", "#0969da"),
    control.ROLLBACK_REQUESTED: ("Rollback Requested", "#9a6700"),
    control.ROLLBACK_VERIFYING: ("Verifying Rollback", "#9a6700"),
    control.FAILED_SAFE: ("Failed Safe", "#cf222e"),
    control.COMPLETED: ("Completed", "#1a7f37"),
}


def render_controls() -> None:
    state = control.get_state()
    label, color = CONTROL_STATE_LABELS.get(state, (state, "#57606a"))

    st.markdown(
        f"**Agent state:** "
        f"<span style='color:{color};font-weight:700'>{label}</span>",
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    if state == control.RUNNING_AUTONOMOUSLY:
        if cols[0].button("Pause", key="nx_pause", use_container_width=True):
            control.transition(control.PAUSE_REQUESTED, reason="engineer clicked pause (NEXUS)", requested_by="engineer")
            st.rerun()
        if cols[1].button("Take Manual Control", key="nx_manual", use_container_width=True):
            control.transition(control.MANUAL_CONTROL, reason="engineer took manual control (NEXUS)", requested_by="engineer")
            st.rerun()
    elif state == control.PAUSE_REQUESTED:
        if cols[0].button("Cancel pause request", key="nx_cancel_pause", use_container_width=True):
            control.transition(control.RUNNING_AUTONOMOUSLY, reason="engineer cancelled pause (NEXUS)", requested_by="engineer")
            st.rerun()
    elif state == control.PAUSED_FOR_ENGINEER:
        if cols[0].button("Resume", key="nx_resume", use_container_width=True):
            control.transition(control.RESUME_REQUESTED, reason="engineer clicked resume (NEXUS)", requested_by="engineer")
            st.rerun()
        if cols[1].button("Take Manual Control", key="nx_manual2", use_container_width=True):
            control.transition(control.MANUAL_CONTROL, reason="engineer took manual control (NEXUS)", requested_by="engineer")
            st.rerun()
    elif state == control.MANUAL_CONTROL:
        if cols[0].button("Resume Autonomous", key="nx_resume2", use_container_width=True):
            control.transition(control.RESUME_REQUESTED, reason="engineer resumed from manual control (NEXUS)", requested_by="engineer")
            st.rerun()
    elif state == control.FAILED_SAFE:
        if cols[0].button("Acknowledge failure -> Pause", key="nx_ack_fail", use_container_width=True):
            control.transition(control.PAUSED_FOR_ENGINEER, reason="engineer acknowledged failure (NEXUS)", requested_by="engineer")
            st.rerun()
    elif state in (control.ROLLBACK_REQUESTED, control.ROLLBACK_VERIFYING):
        cols[0].info("Rollback in progress...")
    elif state == control.COMPLETED:
        cols[0].success("Final model submitted -- nothing left to control.")


# ---------------------------------------------------------------------------
# Page.
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("GoTrain NEXUS")
    st.caption(f"Reading live from {LOGS_DIR}")

    status_data = get_status()
    current_run = status_data.get("current_run") or {}

    render_controls()
    st.divider()

    # --- Headline metrics -------------------------------------------------
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Dataset", current_run.get("dataset", "--"))
    c2.metric("Model", current_run.get("model", "--"))
    epoch = current_run.get("epoch")
    epochs = current_run.get("epochs")
    c3.metric("Epoch", f"{epoch}/{epochs}" if epoch and epochs else "--")
    train_loss = current_run.get("train_loss")
    c4.metric("Train Loss", f"{train_loss:.4f}" if isinstance(train_loss, (int, float)) else "--")
    val_metric = current_run.get("val_metric")
    metric_name = current_run.get("metric_name", "val_metric")
    c5.metric(metric_name, f"{val_metric:.4f}" if isinstance(val_metric, (int, float)) else "--")
    elapsed = current_run.get("elapsed_seconds")
    c6.metric("Elapsed", format_duration(elapsed))

    eta = compute_eta_seconds(current_run)
    st.caption(f"ETA to end of current run: **{format_duration(eta)}**" if current_run else "No run currently active.")

    st.divider()

    # --- Live charts --------------------------------------------------------
    live_rows = load_jsonl(LIVE_METRICS_PATH)
    run_id = current_run.get("run_id")

    chart_rows = [r for r in live_rows if r.get("run_id") == run_id] if run_id else live_rows[-200:]

    if chart_rows:
        df = pd.DataFrame(chart_rows)
        left, right = st.columns(2)

        with left:
            st.subheader("Training loss vs epoch")
            if "train_loss" in df and df["train_loss"].notna().any():
                loss_chart = (
                    alt.Chart(df)
                    .mark_line(point=True)
                    .encode(x="epoch:Q", y="train_loss:Q", tooltip=["run_id", "epoch", "train_loss"])
                    .properties(height=280)
                )
                st.altair_chart(loss_chart, use_container_width=True)
            else:
                st.info("No training-loss history yet.")

        with right:
            st.subheader("Validation metric vs epoch")
            if "val_metric" in df and df["val_metric"].notna().any():
                val_chart = (
                    alt.Chart(df)
                    .mark_line(point=True, color="#1a7f37")
                    .encode(x="epoch:Q", y="val_metric:Q", tooltip=["run_id", "epoch", "val_metric"])
                    .properties(height=280)
                )
                st.altair_chart(val_chart, use_container_width=True)
            else:
                st.info("No validation-metric history yet.")
    else:
        st.info("No per-epoch history yet -- logs/live_metrics.jsonl has no rows for the current run.")

    st.divider()

    # --- Recent events -------------------------------------------------------
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Recent agent events")
        events = load_jsonl(EVENTS_PATH, max_lines=25)
        if events:
            ev_df = pd.DataFrame(events[::-1])
            cols_order = [c for c in ["timestamp", "event"] if c in ev_df.columns]
            cols_order += [c for c in ev_df.columns if c not in cols_order]
            st.dataframe(ev_df[cols_order], use_container_width=True, height=360, hide_index=True)
        else:
            st.info("No events logged yet.")

    with right:
        st.subheader("Completed runs")
        history = load_history()
        if history:
            hist_df = pd.DataFrame(history)
            keep = [c for c in ["run_id", "dataset", "val_metric", "status"] if c in hist_df.columns]
            st.dataframe(hist_df[keep].tail(15) if keep else hist_df.tail(15), use_container_width=True, height=360, hide_index=True)
        else:
            st.info("No completed runs recorded in the ledger yet.")

    time.sleep(REFRESH_SECONDS)
    st.rerun()


if __name__ == "__main__":
    main()
