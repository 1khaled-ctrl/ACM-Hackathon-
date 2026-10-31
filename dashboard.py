"""GoTrain Dashboard — read-only visualization of the agent's experiments.

Shows what the agent is doing, its current/best experiments, how validation
score has improved across runs, and its logged decisions. Never launches
training, never changes a config, never triggers the final test evaluation,
and never displays a test-split score (final_eval.py's output is written to
logs/events.jsonl with split="test" only once a human runs run_final_eval.py
by hand — this dashboard doesn't special-case or surface that split at all).

Reuses the existing backend exactly as-is:
  - gotrain.status / gotrain.ledger for status.json and ledger.jsonl
  - logs/events.jsonl read directly (same technique the previous version of
    this file used) for the raw tool-call/training-lifecycle event stream.

The only backend change made to support this dashboard is that the training
templates now report live per-epoch progress into status.json's
"current_run" field (see gotrain/status.py::set_current_run) — that's
already-computed-every-epoch information the templates simply didn't expose
anywhere before. Everything else here is purely additive reading.

Run with: streamlit run dashboard.py
"""

import html
import json
import time
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import streamlit as st

from gotrain import ledger
from gotrain import status as status_mod
from gotrain.logging_utils import EVENTS_PATH

REFRESH_SECONDS = 4

DATASETS = [
    {"name": "m-eurosat", "task_type": "classification"},
    {"name": "m-so2sat", "task_type": "classification"},
    {"name": "m-pv4ger-seg", "task_type": "segmentation"},
    {"name": "m-nz-cattle", "task_type": "segmentation"},
]
DATASET_TASK_TYPE = {d["name"]: d["task_type"] for d in DATASETS}

WORKFLOW_STEPS = ["Analyze Dataset", "Choose Model", "Train", "Validate", "Compare", "Decide"]

COLOR = {
    "blue": "#3987e5",
    "green": "#0ca30c",
    "red": "#e66767",
    "muted": "#898781",
    "grid": "#2c2c2a",
}


# ---------------------------------------------------------------------------
# Data loading — read-only, tolerant of missing/partially-written files.
# ---------------------------------------------------------------------------


def load_status() -> dict:
    try:
        return status_mod.get_status()
    except Exception:
        return {}


def load_history() -> list:
    try:
        return ledger.get_history()
    except Exception:
        return []


def load_events() -> list:
    if not EVENTS_PATH.exists():
        return []
    events = []
    with open(EVENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # Tolerate a line still being written concurrently by a
                # training subprocess — skip it rather than crash the page.
                continue
    return events


# ---------------------------------------------------------------------------
# Formatting helpers.
# ---------------------------------------------------------------------------


def esc(x) -> str:
    return html.escape(str(x)) if x is not None else ""


def pct(value, decimals=1) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value * 100:.{decimals}f}%"


def fmt_float(value, decimals=4) -> str:
    return f"{value:.{decimals}f}" if isinstance(value, (int, float)) else "—"


def fmt_elapsed(seconds) -> str:
    if not isinstance(seconds, (int, float)) or seconds < 0:
        return "—"
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def metric_label(task_type) -> str:
    return "Mean IoU" if task_type == "segmentation" else "Validation Accuracy"


def model_name(record: dict) -> str:
    cfg = record.get("config") or {}
    if cfg.get("backbone"):
        return cfg["backbone"]
    if cfg.get("architecture") and cfg.get("encoder"):
        return f"{cfg['architecture']}/{cfg['encoder']}"
    return cfg.get("architecture") or cfg.get("encoder") or "—"


def summarize_config(record: dict) -> str:
    diff = record.get("config_diff") or {}
    if diff:
        field, change = next(iter(diff.items()))
        return f"{field}: {change.get('from')} → {change.get('to')}"
    cfg = record.get("config") or {}
    parts = [f"{k}={cfg[k]}" for k in ("backbone", "architecture", "encoder", "lr") if cfg.get(k) is not None]
    return ", ".join(parts) if parts else "default config"


def time_ago_seconds(iso_ts) -> float:
    if not iso_ts:
        return float("inf")
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return float("inf")


def card(title: str, inner_html: str, extra_cls: str = "") -> str:
    return f'<div class="gt-card {extra_cls}"><div class="gt-card-title">{esc(title)}</div>{inner_html}</div>'


def kv_row(label: str, value_html: str) -> str:
    return f'<div class="gt-kv"><span>{esc(label)}</span><b>{value_html}</b></div>'


def empty(message: str) -> str:
    return f'<div class="gt-empty">{esc(message)}</div>'


DECISION_PILL = {"kept": "gt-pill-kept", "discarded": "gt-pill-discarded", "pending": "gt-pill-pending"}


def decision_pill(decision: str) -> str:
    cls = DECISION_PILL.get(decision, "gt-pill-pending")
    return f'<span class="gt-pill {cls}">{esc(decision.capitalize())}</span>'


# ---------------------------------------------------------------------------
# Derived state — inferred from the real event stream, never fabricated.
# ---------------------------------------------------------------------------


def is_training_open(events: list) -> bool:
    """True if the most recent training_started has no training_finished /
    training_failed / training_timed_out after it yet."""
    last_start, last_end = None, None
    for e in events:
        et = e.get("event")
        if et == "training_started":
            last_start = e.get("timestamp")
        elif et in ("training_finished", "training_failed", "training_timed_out"):
            last_end = e.get("timestamp")
    if last_start is None:
        return False
    return last_end is None or last_start > last_end


def infer_agent_status(events: list, training_open: bool) -> str:
    if training_open:
        return "RUNNING"
    if not events:
        return "IDLE"
    last = events[-1]
    et = last.get("event")
    if et == "final_model_submitted":
        return "COMPLETED"
    if et == "training_failed":
        return "ERROR"
    if et == "tool_call" and last.get("status") == "error":
        result_text = (last.get("result_truncated") or "").lower()
        if "budget exhausted" not in result_text:
            return "ERROR"
    return "IDLE"


def infer_step_index(events: list, training_open: bool) -> int:
    """Best-effort current workflow-step index, inferred from the real
    event sequence (never a separate "I am now in phase X" signal — the
    agent doesn't log one). Returns len(WORKFLOW_STEPS) as a sentinel for
    "a final model has been submitted, nothing left to do".
    """
    if training_open:
        return 2  # Train
    for e in reversed(events):
        et = e.get("event")
        if et == "final_model_submitted":
            return len(WORKFLOW_STEPS)
        if et == "decision_recorded":
            return 5  # Decide
        if et == "training_finished":
            return 3  # Validate
        if et == "tool_call" and e.get("tool") == "compare_runs" and e.get("status") == "success":
            return 4  # Compare
        if et in ("training_failed", "training_timed_out"):
            return 2
        if et == "tool_call" and e.get("tool") == "query_history":
            return 1  # Choose Model
        if et == "tool_call" and e.get("tool") == "get_dataset_info":
            return 0  # Analyze
        if et == "agent_initialized":
            return 0
    return 0


def format_event(e: dict):
    """One (HH:MM:SS, friendly text) tuple for an events.jsonl record, or
    None to skip it. Only formats semantic events plus the "exploration"
    tool calls that have no dedicated semantic event of their own — skips
    the generic tool_call entry for launch_training/record_decision/
    submit_final_model since those already have a richer dedicated event,
    to avoid showing the same action twice.
    """
    ts = e.get("timestamp", "")
    short_ts = ts[11:19] if len(ts) >= 19 else ts
    et = e.get("event")

    if et == "agent_initialized":
        return short_ts, f"Agent initialized for {esc(e.get('dataset'))} (max {esc(e.get('max_runs'))} runs)"
    if et == "training_started":
        cfg = e.get("config") or {}
        model = cfg.get("backbone") or cfg.get("architecture") or "a model"
        return short_ts, f"Launching {esc(model)} on {esc(e.get('dataset'))}"
    if et == "training_finished":
        vm = e.get("val_metric")
        suffix = f" — validation score {pct(vm)}" if isinstance(vm, (int, float)) else ""
        return short_ts, f"Training completed{suffix}"
    if et == "training_failed":
        return short_ts, f"Training failed for {esc(e.get('dataset'))}"
    if et == "training_timed_out":
        return short_ts, f"Training timed out for {esc(e.get('dataset'))}"
    if et == "decision_recorded":
        rationale = e.get("rationale") or ""
        return short_ts, f"Decision: {esc(e.get('decision'))} — {esc(rationale)}"
    if et == "final_model_submitted":
        return short_ts, f"Submitted final model (run #{esc(str(e.get('run_id'))[:8])})"
    if et == "tool_call":
        tool = e.get("tool")
        args = e.get("arguments") or {}
        mapping = {
            "list_datasets": "Listed available datasets",
            "get_dataset_info": f"Inspected dataset {esc(args.get('dataset', '?'))}",
            "query_history": f"Reviewed experiment history for {esc(args.get('dataset', '?'))}",
            "compare_runs": "Compared two experiments",
            "get_run_status": "Checked a run's status",
            "get_run_metrics": "Checked a run's metrics",
        }
        return (short_ts, mapping[tool]) if tool in mapping else None
    return None


# ---------------------------------------------------------------------------
# Section renderers.
# ---------------------------------------------------------------------------


def render_sidebar(status_data: dict, training_open: bool) -> str:
    with st.sidebar:
        st.markdown('<div class="gt-logo">🛰️&nbsp; GoTrain</div>', unsafe_allow_html=True)
        view = st.radio(
            "Navigate", ["Dashboard", "Experiments", "Models", "Agent Logs"], label_visibility="collapsed"
        )
        st.markdown("<hr class='gt-sidebar-hr'/>", unsafe_allow_html=True)

        agent_recent = time_ago_seconds(status_data.get("updated_at")) < 60
        gpu_color = "var(--blue)" if training_open else "var(--muted)"
        agent_color = "var(--blue)" if agent_recent else "var(--muted)"
        st.markdown(
            f"""
            <div class="gt-sidebar-status">
              <div class="gt-kv"><span>System</span><b style="color:var(--green)">Online</b></div>
              <div class="gt-kv"><span>GPU</span><b style="color:{gpu_color}">{'Busy' if training_open else 'Available'}</b></div>
              <div class="gt-kv"><span>Agent</span><b style="color:{agent_color}">{'Running' if agent_recent else 'Idle'}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return view


def render_header(status_data: dict, dataset: str, task_type: str, agent_status: str, max_runs) -> None:
    dot_cls = {
        "RUNNING": "gt-dot-running",
        "IDLE": "gt-dot-idle",
        "COMPLETED": "gt-dot-completed",
        "ERROR": "gt-dot-error",
    }[agent_status]
    iteration = status_data.get("iteration")
    if iteration is not None and max_runs:
        iteration_text = f"{iteration} / {max_runs}"
    elif iteration is not None:
        iteration_text = str(iteration)
    else:
        iteration_text = "—"
    action = status_data.get("action") or "Idle"

    st.markdown(
        f"""
        <div class="gt-header">
          <div class="gt-header-top">
            <div class="gt-header-title">GoTrain Agent</div>
            <div class="gt-header-status"><span class="gt-dot {dot_cls}"></span>{esc(agent_status)}</div>
          </div>
          <div class="gt-header-grid">
            <div><div class="gt-header-label">Dataset</div><div class="gt-header-value">{esc(dataset or "—")}</div></div>
            <div><div class="gt-header-label">Task</div><div class="gt-header-value">{esc(task_type.capitalize() if task_type else "—")}</div></div>
            <div><div class="gt-header-label">Iteration</div><div class="gt-header-value">{esc(iteration_text)}</div></div>
          </div>
          <div class="gt-header-action">Currently: {esc(action)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_stepper(current_index: int) -> None:
    completed_all = current_index >= len(WORKFLOW_STEPS)
    chips = []
    for i, label in enumerate(WORKFLOW_STEPS):
        if completed_all or i < current_index:
            cls, icon = "gt-step-done", "✓"
        elif i == current_index:
            cls, icon = "gt-step-current", "●"
        else:
            cls, icon = "gt-step-upcoming", "○"
        chips.append(f'<div class="gt-step {cls}"><span class="gt-step-icon">{icon}</span>{esc(label)}</div>')
        if i < len(WORKFLOW_STEPS) - 1:
            chips.append('<div class="gt-step-line"></div>')
    suffix = ' <span class="gt-pill gt-pill-kept">Final model submitted</span>' if completed_all else ""
    st.markdown(f'<div class="gt-stepper">{"".join(chips)}{suffix}</div>', unsafe_allow_html=True)


def render_test_lock_banner() -> None:
    st.markdown(
        """
        <div class="gt-lock-banner">
          🔒 <b>Test Set Locked</b> — GoTrain optimizes using validation data only.
          Final test evaluation is human-triggered and can only run once per dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_current_experiment(live_run: dict, ds_history: list, dataset: str) -> None:
    if live_run:
        run_no = len(ds_history) + 1
        epoch, epochs = live_run.get("epoch"), live_run.get("epochs")
        progress_pct = int(100 * epoch / epochs) if epoch and epochs else 0
        metric_name = (live_run.get("metric_name") or "")
        metric_disp = "Mean IoU (so far)" if "iou" in metric_name else "Validation Score (so far)"
        inner = "".join(
            [
                kv_row("Run", f"#{run_no:03d}"),
                kv_row("Model", esc(live_run.get("model", "—"))),
                kv_row("Learning Rate", esc(live_run.get("lr", "—"))),
                kv_row("Optimizer", esc(str(live_run.get("optimizer", "—")).upper())),
                kv_row("Scheduler", esc(live_run.get("scheduler") or "none")),
                kv_row("Epoch", f"{esc(epoch) if epoch else '—'} / {esc(epochs) if epochs else '—'}"),
                f'<div class="gt-progress-track"><div class="gt-progress-fill" style="width:{progress_pct}%;"></div></div>',
                kv_row("Train Loss", esc(fmt_float(live_run.get("train_loss")))),
                kv_row(metric_disp, pct(live_run.get("val_metric"))),
                kv_row("Elapsed", esc(fmt_elapsed(live_run.get("elapsed_seconds")))),
            ]
        )
        st.markdown(card("Current Experiment", inner, "gt-card-active"), unsafe_allow_html=True)
        return

    if not ds_history:
        st.markdown(card("Current Experiment", empty("No experiments yet")), unsafe_allow_html=True)
        return

    last = sorted(ds_history, key=lambda r: r["timestamp"])[-1]
    cfg = last.get("config") or {}
    inner = "".join(
        [
            kv_row("Run", f"#{len(ds_history):03d}"),
            kv_row("Model", esc(model_name(last))),
            kv_row("Learning Rate", esc(cfg.get("lr", "—"))),
            kv_row(metric_label(DATASET_TASK_TYPE.get(dataset, "classification")), pct(last.get("val_metric"))),
            kv_row("Decision", decision_pill(last.get("decision") or "pending")),
        ]
    )
    st.markdown(card("Last Experiment — agent idle", inner), unsafe_allow_html=True)


def render_best_model(ds_history: list, dataset: str, task_type: str) -> None:
    numeric = [r for r in ds_history if isinstance(r.get("val_metric"), (int, float))]
    if not numeric:
        st.markdown(card("🏆 Best Model", empty("No experiments yet")), unsafe_allow_html=True)
        return

    best = max(numeric, key=lambda r: r["val_metric"])
    chronological = sorted(numeric, key=lambda r: r["timestamp"])
    first = chronological[0]
    run_no = chronological.index(best) + 1
    # version_id (e.g. "v3") comes from the real model registry
    # (gotrain/registry.py) when this run's checkpoint was promoted there.
    # Runs with no run_dir to promote (e.g. task_type="fake") never get
    # one -- fall back to the lineage-chain length as a rough proxy.
    version = best.get("version_id")
    if not version:
        try:
            version = f"v{len(ledger.get_lineage(best['run_id']))}"
        except Exception:
            version = f"v{run_no}"

    if best["run_id"] == first["run_id"]:
        improvement_text = "First experiment"
    else:
        improvement_text = f"{(best['val_metric'] - first['val_metric']) * 100:+.1f} pts vs. first run"

    rows = [kv_row(metric_label(task_type), pct(best["val_metric"]))]
    extra = best.get("extra_metrics") or {}
    if task_type == "segmentation" and isinstance(extra.get("foreground_iou"), (int, float)):
        rows.append(kv_row("Foreground IoU", pct(extra["foreground_iou"])))
    rows += [
        kv_row("Run", f"#{run_no:03d}"),
        kv_row("Improvement", esc(improvement_text)),
        kv_row("Version", esc(version)),
        kv_row("Dataset", esc(dataset)),
    ]
    inner = f'<div class="gt-hero">{esc(model_name(best))}</div>' + "".join(rows)
    st.markdown(card("🏆 Best Model", inner, "gt-card-best"), unsafe_allow_html=True)


def render_improvement_chart(ds_history: list, best_run_id, task_type: str) -> None:
    numeric = [r for r in ds_history if isinstance(r.get("val_metric"), (int, float))]
    if not numeric:
        st.markdown(card("Experiment Improvement", empty("No experiments yet")), unsafe_allow_html=True)
        return

    rows = []
    for i, r in enumerate(sorted(numeric, key=lambda x: x["timestamp"]), start=1):
        rows.append(
            {
                "run_number": i,
                "label": f"Run {i}",
                "score": r["val_metric"] * 100,
                "model": model_name(r),
                "is_best": r["run_id"] == best_run_id,
            }
        )
    df = pd.DataFrame(rows)

    st.markdown('<div class="gt-card"><div class="gt-card-title">Experiment Improvement</div>', unsafe_allow_html=True)

    base = alt.Chart(df).encode(
        x=alt.X("run_number:O", title="Experiment", axis=alt.Axis(labelColor=COLOR["muted"], titleColor=COLOR["muted"], domainColor=COLOR["grid"], tickColor=COLOR["grid"])),
        y=alt.Y(
            "score:Q",
            title=metric_label(task_type),
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(labelColor=COLOR["muted"], titleColor=COLOR["muted"], gridColor=COLOR["grid"], domainColor=COLOR["grid"]),
        ),
        tooltip=[
            alt.Tooltip("label:N", title="Run"),
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("score:Q", title=metric_label(task_type), format=".1f"),
        ],
    )
    line = base.mark_line(color=COLOR["blue"], strokeWidth=2, interpolate="monotone")
    points = base.mark_point(filled=True, size=90, color=COLOR["blue"])
    best_point = base.transform_filter(alt.datum.is_best).mark_point(
        filled=True, size=240, color=COLOR["green"], stroke="#ffffff", strokeWidth=1.5
    )
    best_label = base.transform_filter(alt.datum.is_best).mark_text(
        dy=-18, color=COLOR["green"], fontWeight="bold", fontSize=11
    ).encode(text=alt.value("BEST"))

    chart = (
        (line + points + best_point + best_label)
        .properties(height=260)
        .configure_view(strokeWidth=0)
        .configure(background="transparent")
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_agent_decision(ds_history: list, live_run: dict, events: list) -> None:
    if not ds_history:
        st.markdown(card("Agent Decision", empty("No decisions recorded yet")), unsafe_allow_html=True)
        return

    last = sorted(ds_history, key=lambda r: r["timestamp"])[-1]
    delta = last.get("delta_vs_parent")

    if last.get("parent_id") is None:
        observation = "This is the first experiment recorded for this dataset — no prior run to compare against."
    elif isinstance(delta, (int, float)) and delta > 0:
        observation = f"Validation improved by {delta * 100:.1f} points after this change."
    elif isinstance(delta, (int, float)) and delta < 0:
        observation = f"Validation dropped by {abs(delta) * 100:.1f} points after this change."
    else:
        observation = "Validation score stayed roughly flat after this change."

    decision = last.get("decision") or "pending"
    reason = last.get("rationale") or "No rationale was recorded for this run."

    if live_run:
        next_action = f"Running experiment #{len(ds_history) + 1:03d} — training {esc(live_run.get('model', 'a new model'))}."
    else:
        next_action = "Waiting for the agent's next move."
        for e in reversed(events):
            formatted = format_event(e)
            if formatted:
                next_action = formatted[1]
                break

    inner = "".join(
        [
            kv_row("Previous experiment", f"{esc(model_name(last))} — {pct(last.get('val_metric'))}"),
            f'<div class="gt-decision-block"><div class="gt-decision-label">Observation</div><div>{esc(observation)}</div></div>',
            f'<div class="gt-decision-block"><div class="gt-decision-label">Decision</div><div>{decision_pill(decision)}</div></div>',
            f'<div class="gt-decision-block"><div class="gt-decision-label">Reason</div><div>{esc(reason)}</div></div>',
            f'<div class="gt-decision-block"><div class="gt-decision-label">Next action</div><div>{next_action}</div></div>',
        ]
    )
    st.markdown(card("Agent Decision", inner), unsafe_allow_html=True)


def render_lineage(best_run_id) -> None:
    if not best_run_id:
        st.markdown(card("Model Lineage", empty("No experiments yet")), unsafe_allow_html=True)
        return
    try:
        chain = ledger.get_lineage(best_run_id)
    except Exception:
        chain = []
    if not chain:
        st.markdown(card("Model Lineage", empty("No experiments yet")), unsafe_allow_html=True)
        return

    nodes = []
    for i, r in enumerate(chain, start=1):
        is_best = r["run_id"] == best_run_id
        star = " ⭐ BEST" if is_best else ""
        node_cls = "gt-lineage-node gt-lineage-best" if is_best else "gt-lineage-node"
        nodes.append(
            f'<div class="{node_cls}"><div class="gt-lineage-version">v{i}</div>'
            f'<div class="gt-lineage-model">{esc(model_name(r))}</div>'
            f'<div class="gt-lineage-score">{pct(r.get("val_metric"))}{star}</div></div>'
        )
        if i < len(chain):
            nodes.append('<div class="gt-lineage-arrow">↓</div>')
    st.markdown(card("Model Lineage", f'<div class="gt-lineage">{"".join(nodes)}</div>'), unsafe_allow_html=True)


def render_history_table(ds_history: list, best_run_id, live_run: dict, dataset: str) -> None:
    ordered = sorted(ds_history, key=lambda r: r["timestamp"])
    rows_html = []
    for i, r in enumerate(ordered, start=1):
        is_best = r["run_id"] == best_run_id
        decision = r.get("decision") or "pending"
        delta = r.get("delta_vs_parent")
        if r.get("parent_id") is None:
            change = "Baseline"
        elif isinstance(delta, (int, float)):
            change = f"{delta * 100:+.1f} pts"
        else:
            change = "—"
        star = " 🏆" if is_best else ""
        rows_html.append(
            f'<tr class="{"gt-row-best" if is_best else ""}">'
            f"<td>#{i:03d}{star}</td>"
            f"<td>{esc(model_name(r))}</td>"
            f"<td>{esc(summarize_config(r))}</td>"
            f"<td>{pct(r.get('val_metric'))}</td>"
            f"<td>{esc(change)}</td>"
            f"<td>{decision_pill(decision)}</td>"
            f'<td><span class="gt-pill gt-pill-complete">Complete</span></td>'
            "</tr>"
        )
    if live_run:
        epoch, epochs = live_run.get("epoch"), live_run.get("epochs")
        progress = f"epoch {epoch}/{epochs}" if epoch and epochs else "starting…"
        rows_html.append(
            '<tr class="gt-row-running">'
            f"<td>#{len(ordered) + 1:03d}</td>"
            f"<td>{esc(live_run.get('model', '—'))}</td>"
            f"<td>{esc(progress)}</td>"
            "<td>—</td><td>—</td><td>—</td>"
            '<td><span class="gt-pill gt-pill-running">Running</span></td>'
            "</tr>"
        )

    if not rows_html:
        st.markdown(card(f"Experiment History — {dataset}", empty("No experiments yet")), unsafe_allow_html=True)
        return

    table = (
        '<div class="gt-table-wrap"><table class="gt-table">'
        "<thead><tr><th>Run</th><th>Model</th><th>Configuration</th><th>Score</th>"
        "<th>Change</th><th>Decision</th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table></div>"
    )
    st.markdown(card(f"Experiment History — {dataset}", table), unsafe_allow_html=True)


def render_datasets_overview(all_history: list, live_dataset) -> None:
    cards = []
    for d in DATASETS:
        name, task_type = d["name"], d["task_type"]
        ds_history = [r for r in all_history if r["dataset"] == name and isinstance(r.get("val_metric"), (int, float))]
        best = max(ds_history, key=lambda r: r["val_metric"], default=None)
        submitted = any(r.get("submitted_final") for r in all_history if r["dataset"] == name)

        if submitted:
            status_label, status_cls = "Complete", "gt-pill-complete"
        elif ds_history:
            status_label, status_cls = "Optimizing", "gt-pill-pending"
        else:
            status_label, status_cls = "Waiting", "gt-pill-waiting"

        metric_short = metric_label(task_type).replace("Validation ", "")
        metric_text = f"{metric_short}: {pct(best['val_metric']) if best else '—'}"
        live_marker = ' <span class="gt-live-marker">● live</span>' if name == live_dataset else ""

        cards.append(
            f'<div class="gt-ds-card"><div class="gt-ds-name">{esc(name)}{live_marker}</div>'
            f'<div class="gt-ds-task">{esc(task_type.capitalize())}</div>'
            f'<div class="gt-ds-metric">{esc(metric_text)}</div>'
            f'<span class="gt-pill {status_cls}">{esc(status_label)}</span></div>'
        )
    st.markdown(card("GEO-Bench Datasets", f'<div class="gt-ds-grid">{"".join(cards)}</div>'), unsafe_allow_html=True)


def render_activity_feed(events: list, limit: int = 20) -> None:
    items = []
    for e in reversed(events):
        formatted = format_event(e)
        if formatted:
            items.append(formatted)
        if len(items) >= limit:
            break

    if items:
        rows = "".join(
            f'<div class="gt-feed-row"><span class="gt-feed-ts">{esc(ts)}</span><span class="gt-feed-text">{text}</span></div>'
            for ts, text in items
        )
    else:
        rows = empty("No agent activity yet")
    st.markdown(card("Live Agent Activity", f'<div class="gt-feed">{rows}</div>'), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Style.
# ---------------------------------------------------------------------------

CSS = """
<style>
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

:root {
  --surface: #1a1a19;
  --surface-2: #20201e;
  --border: rgba(255,255,255,0.10);
  --text: #ffffff;
  --text-2: #c3c2b7;
  --muted: #898781;
  --grid: #2c2c2a;
  --blue: #3987e5;
  --green: #0ca30c;
  --red: #e66767;
}

@keyframes gt-pulse {
  0%   { box-shadow: 0 0 0 0 rgba(57,135,229,0.55); }
  70%  { box-shadow: 0 0 0 7px rgba(57,135,229,0); }
  100% { box-shadow: 0 0 0 0 rgba(57,135,229,0); }
}

.gt-logo { font-size: 1.15rem; font-weight: 700; color: var(--text); margin-bottom: 4px; }
.gt-sidebar-hr { border-color: var(--border); margin: 8px 0 14px; }
.gt-sidebar-status .gt-kv { font-size: 0.85rem; }

.gt-card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
  padding: 18px 20px; margin-bottom: 16px; }
.gt-card-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); font-weight: 700; margin-bottom: 10px; }
.gt-card-active { border-left: 4px solid var(--blue); }
.gt-card-best { border-left: 4px solid var(--green); }

.gt-kv { display: flex; justify-content: space-between; gap: 10px; padding: 4px 0;
  font-size: 0.88rem; color: var(--text-2); border-bottom: 1px solid rgba(255,255,255,0.04); }
.gt-kv:last-child { border-bottom: none; }
.gt-kv b { color: var(--text); font-weight: 600; text-align: right; }
.gt-hero { font-size: 1.5rem; font-weight: 700; color: var(--text); margin-bottom: 8px; }
.gt-empty { color: var(--muted); font-size: 0.88rem; padding: 18px 0; text-align: center; }

.gt-progress-track { height: 8px; border-radius: 6px; background: var(--grid); overflow: hidden; margin: 8px 0; }
.gt-progress-fill { height: 100%; border-radius: 6px; background: var(--blue); transition: width 0.4s ease; }

.gt-header { background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
  padding: 22px 26px; margin-bottom: 14px; }
.gt-header-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 8px; }
.gt-header-title { font-size: 1.65rem; font-weight: 700; color: var(--text); }
.gt-header-status { font-size: 0.95rem; font-weight: 700; letter-spacing: 0.04em; display: flex; align-items: center; color: var(--text); }
.gt-header-grid { display: flex; gap: 48px; margin-bottom: 12px; flex-wrap: wrap; }
.gt-header-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
.gt-header-value { font-size: 1.1rem; font-weight: 600; color: var(--text); }
.gt-header-action { color: var(--text-2); font-size: 0.92rem; }

.gt-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 8px; }
.gt-dot-running { background: var(--blue); animation: gt-pulse 1.6s infinite; }
.gt-dot-idle { background: var(--muted); }
.gt-dot-completed { background: var(--green); }
.gt-dot-error { background: var(--red); }

.gt-stepper { display: flex; align-items: center; gap: 4px; overflow-x: auto; padding: 2px 2px 14px; flex-wrap: wrap; }
.gt-step { display: flex; align-items: center; gap: 6px; font-size: 0.82rem; white-space: nowrap;
  padding: 6px 12px; border-radius: 999px; border: 1px solid var(--border); font-weight: 500; }
.gt-step-icon { font-size: 0.8rem; }
.gt-step-done { color: var(--green); border-color: rgba(12,163,12,0.35); }
.gt-step-current { color: var(--blue); border-color: rgba(57,135,229,0.55); background: rgba(57,135,229,0.10); font-weight: 700; }
.gt-step-current .gt-step-icon { animation: gt-pulse 1.6s infinite; border-radius: 50%; }
.gt-step-upcoming { color: var(--muted); }
.gt-step-line { flex: 0 0 18px; height: 1px; background: var(--grid); }

.gt-lock-banner { background: rgba(255,255,255,0.035); border: 1px solid var(--border); border-radius: 12px;
  padding: 10px 16px; font-size: 0.83rem; color: var(--text-2); margin-bottom: 18px; }

.gt-pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.72rem; font-weight: 700; white-space: nowrap; }
.gt-pill-kept, .gt-pill-complete { background: rgba(12,163,12,0.16); color: var(--green); }
.gt-pill-discarded { background: rgba(255,255,255,0.08); color: var(--text-2); }
.gt-pill-pending, .gt-pill-running { background: rgba(57,135,229,0.16); color: var(--blue); }
.gt-pill-waiting { background: rgba(255,255,255,0.06); color: var(--muted); }

.gt-decision-block { margin-top: 12px; }
.gt-decision-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 3px; }
.gt-decision-block > div:last-child { font-size: 0.9rem; color: var(--text-2); }

.gt-lineage { display: flex; flex-direction: column; align-items: center; gap: 2px; }
.gt-lineage-node { width: 100%; max-width: 280px; background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 10px; padding: 10px 16px; text-align: center; }
.gt-lineage-best { border-color: rgba(12,163,12,0.5); background: rgba(12,163,12,0.08); }
.gt-lineage-version { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
.gt-lineage-model { font-weight: 700; color: var(--text); margin: 2px 0; }
.gt-lineage-score { color: var(--text-2); font-size: 0.85rem; }
.gt-lineage-arrow { color: var(--muted); font-size: 1rem; line-height: 1.4; }

.gt-table-wrap { overflow-x: auto; }
.gt-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.gt-table th { text-align: left; color: var(--muted); font-weight: 700; font-size: 0.68rem;
  text-transform: uppercase; letter-spacing: 0.04em; padding: 6px 10px; border-bottom: 1px solid var(--grid); }
.gt-table td { padding: 8px 10px; border-bottom: 1px solid var(--grid); color: var(--text-2); white-space: nowrap; }
.gt-row-best td { color: var(--text); background: rgba(12,163,12,0.07); }
.gt-row-running td { color: var(--text); background: rgba(57,135,229,0.09); }

.gt-ds-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.gt-ds-card { background: var(--surface-2); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; }
.gt-ds-name { font-weight: 700; font-size: 0.92rem; color: var(--text); }
.gt-ds-task { color: var(--muted); font-size: 0.75rem; margin: 2px 0 8px; }
.gt-ds-metric { font-size: 0.82rem; color: var(--text-2); margin-bottom: 8px; }
.gt-live-marker { color: var(--blue); font-size: 0.68rem; font-weight: 700; }

.gt-feed { max-height: 380px; overflow-y: auto; display: flex; flex-direction: column; gap: 7px; }
.gt-feed-row { font-size: 0.83rem; color: var(--text-2); display: flex; gap: 10px; }
.gt-feed-ts { color: var(--muted); font-variant-numeric: tabular-nums; flex: 0 0 60px; }
</style>
"""


# ---------------------------------------------------------------------------
# Page assembly.
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="GoTrain", layout="wide", page_icon="🛰️")
    st.markdown(CSS, unsafe_allow_html=True)

    status_data = load_status()
    all_history = load_history()
    events = load_events()

    live_dataset = status_data.get("where")
    live_task_type = DATASET_TASK_TYPE.get(live_dataset)
    training_open = bool(status_data.get("current_run")) or is_training_open(events)
    agent_status = infer_agent_status(events, training_open)
    current_run = status_data.get("current_run")

    max_runs = next(
        (e.get("max_runs") for e in reversed(events) if e.get("event") == "agent_initialized" and e.get("dataset") == live_dataset),
        None,
    )

    view = render_sidebar(status_data, training_open)

    render_header(status_data, live_dataset, live_task_type, agent_status, max_runs)
    render_workflow_stepper(infer_step_index(events, training_open))
    render_test_lock_banner()

    dataset_names = [d["name"] for d in DATASETS]
    default_index = dataset_names.index(live_dataset) if live_dataset in dataset_names else 0
    selected_dataset = st.selectbox("Dataset details", dataset_names, index=default_index, key="selected_dataset")
    selected_task_type = DATASET_TASK_TYPE[selected_dataset]

    ds_history = [r for r in all_history if r["dataset"] == selected_dataset]
    best = ledger.get_best(selected_dataset)
    best_run_id = best["run_id"] if best else None
    live_run = current_run if current_run and current_run.get("dataset") == selected_dataset else None

    left, right = st.columns([2, 1], gap="large")

    with right:
        render_best_model(ds_history, selected_dataset, selected_task_type)

    with left:
        render_current_experiment(live_run, ds_history, selected_dataset)

    if view in ("Dashboard", "Experiments"):
        with left:
            render_improvement_chart(ds_history, best_run_id, selected_task_type)
        with right:
            render_agent_decision(ds_history, live_run, events)

    if view in ("Dashboard", "Models"):
        with left:
            render_lineage(best_run_id)

    if view in ("Dashboard", "Experiments"):
        render_history_table(ds_history, best_run_id, live_run, selected_dataset)

    if view in ("Dashboard", "Models"):
        render_datasets_overview(all_history, live_dataset)

    if view in ("Dashboard", "Agent Logs"):
        feed_limit = 40 if view == "Agent Logs" else 20
        with right:
            render_activity_feed(events, limit=feed_limit)

    time.sleep(REFRESH_SECONDS)
    st.rerun()


if __name__ == "__main__":
    main()
