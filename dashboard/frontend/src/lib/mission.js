/**
 * Shared, pure (no React, no network) formatting over one mission's
 * scoped data -- the exact shape gotrain/history.py's load_session()
 * returns, and what the live snapshot's `mission` field carries for
 * whichever session is current. Used by BOTH the live dashboard panels
 * and History Details, so "what counts as this mission's data" and "how
 * an event reads" can never drift between the two views.
 *
 * Nothing here invents data: every field read below is something
 * gotrain/ledger.py or gotrain/logging_utils.py already records. This
 * module only decides how to group, dedupe and phrase it.
 */

// ---------------------------------------------------------------------
// Event humanization for the Live Event Feed.
// ---------------------------------------------------------------------

/** One line per event TYPE this backend actually logs (see
 *  gotrain/logging_utils.py log_event call sites + gotrain/approval.py
 *  _append_audit) -- not a guess at a vocabulary, the real one. Each
 *  entry returns {text, tone}; tone drives the dot color in the feed. */
const EVENT_FORMATTERS = {
  agent_initialized: (e) => ({
    text: `Mission started on ${e.dataset} (${e.max_runs} runs x ${e.run_minutes}min, ${e.max_tool_calls} tool calls)`,
    tone: 'sky',
  }),
  geoguard_started: (e) => ({ text: `GeoGuard scanning ${e.dataset}...`, tone: 'slate' }),
  geoguard_finished: (e) => ({
    text: `GeoGuard scan complete${e.n_findings != null ? ` -- ${e.n_findings} finding(s)` : ''}`,
    tone: 'slate',
  }),
  dataiq_structural_scan_started: () => ({ text: 'DataIQ structural scan running...', tone: 'slate' }),
  dataiq_structural_scan_finished: (e) => ({
    text: `DataIQ structural scan complete${e.n_invalid ? ` -- ${e.n_invalid} invalid sample(s) flagged` : ' -- no issues found'}`,
    tone: 'slate',
  }),
  dataiq_loss_scan_started: () => ({ text: 'DataIQ loss scan running...', tone: 'slate' }),
  dataiq_loss_scan_finished: (e) => ({
    text: `DataIQ loss scan complete${e.n_flagged ? ` -- ${e.n_flagged} suspicious sample(s)` : ''}`,
    tone: 'slate',
  }),
  stresslab_started: () => ({ text: 'StressLab robustness check running...', tone: 'slate' }),
  stresslab_finished: (e) => ({
    text: `StressLab complete${e.verdict ? ` -- ${e.verdict}` : ''}`,
    tone: e.verdict && e.verdict.includes('fragile') ? 'amber' : 'slate',
  }),
  model_compatibility_check_started: () => ({ text: 'Checking pretrained-model compatibility...', tone: 'slate' }),
  model_compatibility_check_finished: (e) => ({
    text: `Compatibility check: ${e.compatible ? 'compatible' : 'not compatible'}`,
    tone: e.compatible ? 'slate' : 'amber',
  }),
  training_started: (e) => ({
    text: `Training started -- ${modelLabel(e.config) || e.task_type || 'run'}`,
    tone: 'sky',
  }),
  training_finished: (e) => ({ text: 'Training finished, evaluating on validation...', tone: 'sky' }),
  training_failed: (e) => ({ text: `Training FAILED${e.dataset ? ` (${e.dataset})` : ''}`, tone: 'red' }),
  training_timed_out: (e) => ({
    text: `Training timed out after ${e.timeout_seconds ? Math.round(e.timeout_seconds / 60) + 'min' : 'its budget'}`,
    tone: 'red',
  }),
  evaluation: (e) => ({
    text: `Evaluated on ${e.split || 'validation'} split${e.metrics ? '' : ''}`,
    tone: e.split === 'test' ? 'red' : 'slate',
  }),
  decision_recorded: (e) => ({
    text: `Run ${shortId(e.run_id)} marked ${e.decision || '?'}${e.rationale ? ` -- ${truncate(e.rationale, 70)}` : ''}`,
    tone: e.decision === 'kept' ? 'green' : 'slate',
  }),
  adaptation_recorded: (e) => ({
    text: `Adapted after rejection: ${truncate(e.new_plan, 80) || 'see detail'}`,
    tone: 'amber',
  }),
  model_passport_generated: () => ({ text: 'Model passport generated', tone: 'slate' }),
  model_passport_generation_failed: () => ({ text: 'Model passport generation failed (non-fatal)', tone: 'amber' }),
  final_model_submitted: (e) => ({
    text: `Final model submitted -- run ${shortId(e.run_id)}`,
    tone: 'green',
  }),
  rollback_succeeded: (e) => ({ text: `Rolled back ${e.dataset} to ${e.version}`, tone: 'sky' }),
  rollback_failed: (e) => ({
    text: `Rollback FAILED (${e.stage || 'verification'})`,
    tone: 'red',
  }),
  approval_requested: (e) => ({
    text: `Approval requested: ${e.field}=${e.value} -- ${truncate(e.reason, 60)}`,
    tone: 'amber',
  }),
  approval_decided: (e) => ({
    text: `${e.status === 'approved' ? 'Approved' : 'Rejected'}: ${e.field}=${e.value}`,
    tone: e.status === 'approved' ? 'green' : 'red',
  }),
  control_transition: (e) => {
    if (e.requested_by !== 'engineer') return null // agent's own ack (e.g. pause acknowledged) is noise here
    const labels = {
      PAUSE_REQUESTED: 'Engineer paused the mission',
      RUNNING_AUTONOMOUSLY: 'Engineer resumed the mission',
      MANUAL_CONTROL: 'Engineer took manual control',
      RESUME_REQUESTED: 'Engineer requested resume',
      ROLLBACK_REQUESTED: 'Engineer requested a rollback',
    }
    return { text: labels[e.to_state] || `Engineer action: ${e.to_state}`, tone: 'amber' }
  },
}

function shortId(id) {
  return id ? String(id).slice(0, 8) : '?'
}
function truncate(s, n) {
  if (!s) return ''
  return s.length > n ? s.slice(0, n) + '...' : s
}
function modelLabel(config) {
  if (!config) return null
  return config.backbone || config.encoder || config.architecture || null
}

/** Turn one tool_call event into a feed line -- these are high-volume
 *  and low-signal by default, so only a few tool names get a dedicated
 *  phrasing; everything else is suppressed (returns null) to keep the
 *  feed readable rather than a raw trace. */
function formatToolCall(e) {
  const tool = e.tool
  if (tool === 'launch_training') return null // training_started/finished already cover this
  if (tool === 'submit_final_model') return null // final_model_submitted already covers this
  if (tool === 'query_history') return null // too frequent, no new information for a human
  if (e.status === 'error') {
    return { text: `${tool} refused: ${truncate(e.result_truncated || e.exception, 90)}`, tone: 'amber' }
  }
  return null
}

/**
 * One raw event -> {key, text, tone, timestamp} or null to suppress it.
 * null is a deliberate, common outcome -- most tool_call rows and the
 * agent's own pause-acknowledgement transitions are noise, not signal.
 */
export function humanizeEvent(e) {
  if (!e || !e.event) return null
  let formatted
  if (e.event === 'tool_call') {
    formatted = formatToolCall(e)
  } else {
    const fn = EVENT_FORMATTERS[e.event]
    formatted = fn ? fn(e) : { text: e.event.replaceAll('_', ' '), tone: 'slate' }
  }
  if (!formatted) return null
  return {
    key: `${e.timestamp}-${e.event}-${e.run_id || e.key || ''}`,
    timestamp: e.timestamp,
    ...formatted,
  }
}

/**
 * A mission's full raw event list -> deduped, humanized feed entries,
 * newest first. Dedup is on the formatted key (not just timestamp),
 * which also collapses two raw events that happen to render identically
 * a moment apart.
 */
export function buildEventFeed(events) {
  const seen = new Set()
  const out = []
  for (const e of events || []) {
    const h = humanizeEvent(e)
    if (!h || seen.has(h.key)) continue
    seen.add(h.key)
    out.push(h)
  }
  return out.reverse()
}

// ---------------------------------------------------------------------
// Decision Timeline + Run Comparison: both read the same `runs` array
// (gotrain/ledger.py records, already session-scoped by the backend).
// ---------------------------------------------------------------------

/** Was this run actually confirmed as diminishing-returns / early-stop
 *  by the system prompt's stopping rule (two consecutive runs each
 *  improving <0.3 over their parent)? Approximate, read-only check over
 *  what's already in the ledger -- the agent's own record_decision
 *  rationale is the authoritative account; this only flags the pattern
 *  for a UI badge. */
export function isLikelyDiminishingReturns(run, parentRun) {
  if (!parentRun) return false
  const delta = run.delta_vs_parent
  return typeof delta === 'number' && delta >= 0 && delta < 0.3 &&
    typeof parentRun.delta_vs_parent === 'number' && parentRun.delta_vs_parent >= 0 && parentRun.delta_vs_parent < 0.3
}

/** Build a lookup of runs that failed or timed out but never made it
 *  into the ledger (training_failed/training_timed_out are events, not
 *  ledger rows -- see gotrain/tools/training.py: a non-zero exit never
 *  calls ledger.add_run). Needed so the Decision Timeline and Run
 *  Comparison can show "attempt #N failed" instead of silently having a
 *  gap where a run should be. */
export function extractFailedAttempts(events) {
  return (events || [])
    .filter((e) => e.event === 'training_failed' || e.event === 'training_timed_out')
    .map((e) => ({
      timestamp: e.timestamp,
      kind: e.event,
      dataset: e.dataset,
      task_type: e.task_type,
      timeout_seconds: e.timeout_seconds,
    }))
}

/**
 * `current_run` (from logs/status.json, written per-epoch by
 * report_progress in the training templates) has a DIFFERENT shape from
 * a ledger record: a flat `model` string instead of `config.backbone`,
 * no config_diff/hypothesis/decision (those only exist once the run is
 * recorded in the ledger, after it finishes). Normalizing it into a
 * ledger-row-compatible shape here means the comparison table and
 * timeline can treat "the run in progress" the same as any finished row
 * instead of needing special-cased field reads scattered through the UI.
 */
export function normalizeLiveRun(currentRun, mission) {
  if (!currentRun || !currentRun.run_id) return null
  // Guard against the exact cross-mission bug this whole module exists
  // to fix: logs/status.json's current_run can lag behind (or outlive) a
  // mission boundary -- e.g. a stale snapshot still naming a run from a
  // PREVIOUS mission's dataset after a new one has started, or after the
  // agent process has already exited. Never show a live row whose
  // dataset doesn't match the mission being displayed.
  if (mission && currentRun.dataset && currentRun.dataset !== mission.dataset) return null
  return {
    run_id: currentRun.run_id,
    dataset: currentRun.dataset,
    task_type: currentRun.task_type,
    parent_id: null,
    config: { backbone: currentRun.model, optimizer: currentRun.optimizer, lr: currentRun.lr, scheduler: currentRun.scheduler },
    config_diff: null,
    val_metric: currentRun.val_metric,
    delta_vs_parent: null,
    train_loss: currentRun.train_loss,
    elapsed_seconds: currentRun.elapsed_seconds,
    decision: null,
    rationale: null,
    hypothesis: null,
    hypothesis_outcome: null,
    timestamp: null,
  }
}

/**
 * Final train_loss and elapsed_seconds per run_id, read from
 * live_metrics.jsonl's last epoch row for that run -- NEITHER is a
 * ledger field (gotrain/ledger.py's add_run stores val_metric,
 * delta_vs_parent, gpu_minutes -- never per-epoch train_loss or
 * elapsed_seconds; those only exist per-epoch via report_progress ->
 * log_live_metric). Looking these up here, rather than assuming they
 * sit on the run record, is what keeps Run Comparison's "Train Loss" and
 * "Duration" columns honest instead of always blank.
 */
export function finalEpochStatsByRun(liveMetrics) {
  const byRun = new Map()
  for (const m of liveMetrics || []) {
    if (!m.run_id) continue
    const existing = byRun.get(m.run_id)
    if (!existing || (m.epoch ?? 0) >= (existing.epoch ?? 0)) {
      byRun.set(m.run_id, { epoch: m.epoch, train_loss: m.train_loss, elapsed_seconds: m.elapsed_seconds })
    }
  }
  return byRun
}

/**
 * Merge real ledger runs with failed/timed-out attempts into one
 * chronological timeline, each tagged with its own `kind` so the UI can
 * render success vs. failure rows differently without two separate
 * lists falling out of sync with each other.
 */
export function buildTimeline(mission) {
  if (!mission) return []
  const runs = (mission.runs || []).map((r, i) => ({
    kind: 'run',
    timestamp: r.timestamp,
    index: i,
    data: r,
  }))
  const failed = extractFailedAttempts(mission.events).map((f) => ({
    kind: f.kind,
    timestamp: f.timestamp,
    data: f,
  }))
  return [...runs, ...failed].sort((a, b) => (a.timestamp || '').localeCompare(b.timestamp || ''))
}
