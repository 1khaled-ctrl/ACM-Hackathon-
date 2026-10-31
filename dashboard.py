"""Read-only Streamlit dashboard over logs/status.json, logs/events.jsonl,
and logs/ledger.jsonl. Shows what the agent is doing right now, the run
history/lineage per dataset, and the raw event stream. Never reads or
displays test-split data — only the ledger/status/event metadata that the
rest of this project already keeps test-split-free.

Run with: streamlit run dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

LOGS_DIR = Path(__file__).resolve().parent / "logs"
STATUS_PATH = LOGS_DIR / "status.json"
EVENTS_PATH = LOGS_DIR / "events.jsonl"
LEDGER_PATH = LOGS_DIR / "ledger.jsonl"

st.set_page_config(page_title="GoTrain Dashboard", layout="wide")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


st.title("GoTrain Dashboard")

status = load_json(STATUS_PATH)
events = load_jsonl(EVENTS_PATH)
runs = load_jsonl(LEDGER_PATH)

st.header("Current status")
if status:
    cols = st.columns(4)
    cols[0].metric("Stage", status.get("stage", "-"))
    cols[1].metric("Action", status.get("action", "-"))
    cols[2].metric("Where", status.get("where", "-"))
    cols[3].metric("Iteration", status.get("iteration", "-"))

    best = status.get("best_so_far")
    if best:
        st.success(
            f"Best so far: run `{best['run_id']}` on {best['dataset']} "
            f"— val_metric = {best['val_metric']}"
        )
    if status.get("last_event"):
        st.caption(f"Last event: {status['last_event']}")
    st.caption(f"Updated at: {status.get('updated_at', '-')}")
else:
    st.info("No status.json yet — nothing has run.")

st.header("Run history")
if runs:
    df = pd.DataFrame(runs)
    datasets = sorted(df["dataset"].unique())
    selected = st.selectbox("Dataset", ["All"] + datasets)
    view = df if selected == "All" else df[df["dataset"] == selected]

    display_cols = [
        "run_id", "parent_id", "dataset", "val_metric", "delta_vs_parent",
        "decision", "split_used", "gpu_minutes", "timestamp",
    ]
    st.dataframe(view[display_cols], use_container_width=True)

    if selected != "All" and not view.empty:
        chart_df = view[["timestamp", "val_metric"]].dropna()
        if not chart_df.empty:
            st.line_chart(chart_df.set_index("timestamp")["val_metric"])

    with st.expander("Inspect one run (config diff + rationale)"):
        run_id = st.selectbox("run_id", view["run_id"].tolist())
        if run_id:
            record = next(r for r in runs if r["run_id"] == run_id)
            st.json(record)
else:
    st.info("No runs recorded yet.")

st.header("Event stream")
if events:
    st.dataframe(pd.DataFrame(events), use_container_width=True)
else:
    st.info("No events recorded yet.")
