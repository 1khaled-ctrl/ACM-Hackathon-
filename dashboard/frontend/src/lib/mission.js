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
 *  entry returns {text, raw, tone}:
 *    - text: a short, plain-English line a non-technical judge can read
 *      at a glance -- no tool/module names (GeoGuard, DataIQ), no jargon
 *      (split, config_diff, hypothesis_outcome).
 *    - raw: the original technical phrasing, kept verbatim for the
 *      expand/detail view -- this is what an engineer actually wants,
 *      and it must never be lost just because `text` got friendlier.
 *    - tone: drives the dot color in the feed.
 *  Backend event SEMANTICS are untouched by any of this -- these
 *  formatters only choose how to display a field the backend already
 *  logs, never add/remove/rename what gets logged. */
const EVENT_FORMATTERS = {
  agent_initialized: (e) => ({
    text: `Mission started on ${e.dataset}`,
    raw: `Mission started on ${e.dataset} (${e.max_runs} runs x ${e.run_minutes}min, ${e.max_tool_calls} tool calls)`,
    tone: 'sky',
  }),
  geoguard_started: (e) => ({
    text: 'Checking the dataset before training...',
    raw: `GeoGuard scanning ${e.dataset}...`,
    tone: 'slate',
  }),
  geoguard_finished: (e) => ({
    text: e.n_findings
      ? `Dataset check complete — ${e.n_findings} potential issue${e.n_findings === 1 ? '' : 's'} found`
      : 'Dataset check complete — no issues found',
    raw: `GeoGuard scan complete${e.n_findings != null ? ` -- ${e.n_findings} finding(s)` : ''}`,
    tone: 'slate',
  }),
  dataiq_structural_scan_started: (e) => ({
    text: 'Checking data quality...',
    raw: 'DataIQ structural scan running...',
    tone: 'slate',
  }),
  dataiq_structural_scan_finished: (e) => ({
    text: e.n_invalid
      ? `Data quality check complete — ${e.n_invalid} problem sample${e.n_invalid === 1 ? '' : 's'} flagged`
      : 'Data quality check complete — no structural issues found',
    raw: `DataIQ structural scan complete${e.n_invalid ? ` -- ${e.n_invalid} invalid sample(s) flagged` : ' -- no issues found'}`,
    tone: 'slate',
  }),
  dataiq_loss_scan_started: (e) => ({
    text: 'Double-checking training data for mislabeled examples...',
    raw: 'DataIQ loss scan running...',
    tone: 'slate',
  }),
  dataiq_loss_scan_finished: (e) => ({
    text: e.n_flagged
      ? `Data review complete — ${e.n_flagged} sample${e.n_flagged === 1 ? '' : 's'} worth a second look`
      : 'Data review complete — nothing suspicious found',
    raw: `DataIQ loss scan complete${e.n_flagged ? ` -- ${e.n_flagged} suspicious sample(s)` : ''}`,
    tone: 'slate',
  }),
  stresslab_started: (e) => ({
    text: 'Testing the model under tougher conditions...',
    raw: 'StressLab robustness check running...',
    tone: 'slate',
  }),
  stresslab_finished: (e) => ({
    text: e.verdict && e.verdict.includes('fragile')
      ? 'Robustness check complete — model is sensitive to some conditions'
      : 'Robustness check complete — model held up well',
    raw: `StressLab complete${e.verdict ? ` -- ${e.verdict}` : ''}`,
    tone: e.verdict && e.verdict.includes('fragile') ? 'amber' : 'slate',
  }),
  model_compatibility_check_started: (e) => ({
    text: 'Checking the model fits this dataset...',
    raw: 'Checking pretrained-model compatibility...',
    tone: 'slate',
  }),
  model_compatibility_check_finished: (e) => ({
    text: e.compatible ? 'Model is a good fit for this dataset' : 'Model is NOT a good fit for this dataset',
    raw: `Compatibility check: ${e.compatible ? 'compatible' : 'not compatible'}`,
    tone: e.compatible ? 'slate' : 'amber',
  }),
  training_started: (e) => {
    const model = modelLabel(e.config) || e.task_type || 'the model'
    return {
      text: `Training started with ${friendlyModelName(model)}`,
      raw: `Training started -- ${model}`,
      tone: 'sky',
    }
  },
  training_finished: (e) => ({
    text: 'Training complete — checking model performance',
    raw: 'Training finished, evaluating on validation...',
    tone: 'sky',
  }),
  training_failed: (e) => ({
    text: `Training did not complete successfully${e.dataset ? ` for ${e.dataset}` : ''}`,
    raw: `Training FAILED${e.dataset ? ` (${e.dataset})` : ''}`,
    tone: 'red',
  }),
  training_timed_out: (e) => ({
    text: 'Training ran out of time and was stopped',
    raw: `Training timed out after ${e.timeout_seconds ? Math.round(e.timeout_seconds / 60) + 'min' : 'its budget'}`,
    tone: 'red',
  }),
  evaluation: (e) => ({
    text: e.split === 'test'
      ? 'Final one-time accuracy check performed'
      : 'Checked model performance on held-out data',
    raw: `Evaluated on ${e.split || 'validation'} split`,
    tone: e.split === 'test' ? 'red' : 'slate',
  }),
  decision_recorded: (e) => ({
    text: e.decision === 'kept'
      ? `Experiment kept${e.rationale ? ` — ${friendlyRationale(e.rationale)}` : ''}`
      : `Experiment discarded${e.rationale ? ` — ${friendlyRationale(e.rationale)}` : ''}`,
    raw: `Run ${shortId(e.run_id)} marked ${e.decision || '?'}${e.rationale ? ` -- ${e.rationale}` : ''}`,
    tone: e.decision === 'kept' ? 'green' : 'slate',
  }),
  adaptation_recorded: (e) => ({
    text: `Agent adjusted its plan — ${truncate(e.new_plan, 80) || 'see details'}`,
    raw: `Adapted after rejection: ${e.new_plan || 'see detail'}`,
    tone: 'amber',
  }),
  model_passport_generated: (e) => ({
    text: 'Model summary report created',
    raw: 'Model passport generated',
    tone: 'slate',
  }),
  model_passport_generation_failed: (e) => ({
    text: 'Model summary report could not be created (did not affect training)',
    raw: 'Model passport generation failed (non-fatal)',
    tone: 'amber',
  }),
  final_model_submitted: (e) => ({
    text: `Final model selected — run ${shortId(e.run_id)}`,
    raw: `Final model submitted -- run ${shortId(e.run_id)}`,
    tone: 'green',
  }),
  rollback_succeeded: (e) => ({
    text: `Switched back to an earlier model version (${e.version})`,
    raw: `Rolled back ${e.dataset} to ${e.version}`,
    tone: 'sky',
  }),
  rollback_failed: (e) => ({
    text: 'Could not switch to the earlier model version',
    raw: `Rollback FAILED (${e.stage || 'verification'})`,
    tone: 'red',
  }),
  approval_requested: (e) => ({
    text: `Waiting on engineer approval — ${e.field}=${e.value}`,
    raw: `Approval requested: ${e.field}=${e.value} -- ${e.reason || ''}`,
    tone: 'amber',
  }),
  approval_decided: (e) => ({
    text: `Engineer ${e.status === 'approved' ? 'approved' : 'rejected'} a proposed change (${e.field}=${e.value})`,
    raw: `${e.status === 'approved' ? 'Approved' : 'Rejected'}: ${e.field}=${e.value}`,
    tone: e.status === 'approved' ? 'green' : 'red',
  }),
  control_transition: (e) => {
    if (e.requested_by !== 'engineer') return null // agent's own ack (e.g. pause acknowledged) is noise here
    const labels = {
      PAUSE_REQUESTED: { text: 'Engineer paused the mission', raw: 'Engineer requested pause' },
      RUNNING_AUTONOMOUSLY: { text: 'Engineer returned control to the autonomous agent', raw: 'Engineer resumed the mission' },
      MANUAL_CONTROL: { text: 'Engineer took control of the agent', raw: 'Engineer took manual control' },
      RESUME_REQUESTED: { text: 'Engineer returned control to the autonomous agent', raw: 'Engineer requested resume' },
      ROLLBACK_REQUESTED: { text: 'Engineer requested switching to an earlier model', raw: 'Engineer requested a rollback' },
    }
    const picked = labels[e.to_state] || { text: `Engineer action: ${e.to_state}`, raw: `Engineer action: ${e.to_state}` }
    return { ...picked, tone: 'amber' }
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

/** resnet34 -> ResNet34, convnext_tiny -> ConvNeXt Tiny, unet -> UNet --
 *  a small, honest cosmetic cleanup (capitalization/spacing only, never
 *  inventing a different name) so a non-technical judge reads a model
 *  name instead of a raw config-key string. Falls back to the original
 *  string verbatim for anything not in this short, known list rather
 *  than guessing at a transform for a name it doesn't recognize. */
const KNOWN_MODEL_NAMES = {
  resnet18: 'ResNet18', resnet34: 'ResNet34', resnet50: 'ResNet50',
  convnext_tiny: 'ConvNeXt Tiny', convnext_small: 'ConvNeXt Small',
  unet: 'UNet', fpn: 'FPN',
}
function friendlyModelName(raw) {
  if (!raw) return raw
  return KNOWN_MODEL_NAMES[raw] || raw
}

/** Agent-written rationale is free text (see record_decision in the
 *  system prompt's own example: "switched to adamw, val accuracy +2.1
 *  vs parent, keeping") -- there is no fixed template to parse into a
 *  guaranteed-correct summary. This only extracts a signed delta number
 *  when one is clearly present (the one fact worth calling out in plain
 *  words: did it help or hurt) and otherwise truncates the rationale
 *  verbatim rather than reword text it can't reliably interpret -- never
 *  fabricate what a free-text rationale "means". The full text is always
 *  still available in the raw/expanded view regardless. */
function friendlyRationale(rationale) {
  if (!rationale) return ''
  const deltaMatch = rationale.match(/([+-][\d.]+)/)
  if (deltaMatch) {
    const delta = parseFloat(deltaMatch[1])
    if (Number.isFinite(delta) && delta !== 0) {
      const direction = delta > 0 ? 'improved' : 'hurt'
      return `change ${direction} validation performance (${truncate(rationale, 50)})`
    }
  }
  return truncate(rationale, 70)
}

/** A few tool_name -> plain-English activity labels, used only for the
 *  "paused while engineer control is active" line below -- not a full
 *  tool-name translation table (most tool calls are suppressed
 *  entirely, see formatToolCall). */
const TOOL_ACTIVITY_LABEL = {
  run_geoguard: 'Dataset scan',
  run_dataiq_structural_scan: 'Data quality check',
  run_dataiq_loss_scan: 'Data review',
  run_stresslab: 'Robustness check',
  launch_training: 'Training',
  submit_final_model: 'Model submission',
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
    const raw = e.result_truncated || e.exception || ''
    // The hook's own cancellation message for a paused/manually-controlled
    // agent always contains this exact marker (see gotrain/agent.py's
    // LimitsAndLoggingHooks._on_before) -- safe to pattern-match on,
    // not a guess at arbitrary error text.
    if (raw.includes('(engineer-controlled)')) {
      const label = TOOL_ACTIVITY_LABEL[tool] || 'Action'
      return {
        text: `${label} paused while engineer control is active`,
        raw: `${tool} refused: ${truncate(raw, 90)}`,
        tone: 'amber',
      }
    }
    return {
      text: `An automated check (${tool.replaceAll('_', ' ')}) was refused`,
      raw: `${tool} refused: ${truncate(raw, 90)}`,
      tone: 'amber',
    }
  }
  return null
}

/**
 * One raw event -> {key, text, raw, tone, timestamp, event} or null to
 * suppress it. null is a deliberate, common outcome -- most tool_call
 * rows and the agent's own pause-acknowledgement transitions are noise,
 * not signal.
 *
 * `text` is always the friendly line; `raw` is always present too (every
 * formatter above sets it, or it falls back to `text` itself when a
 * formatter doesn't distinguish the two) -- the Live Event Feed uses
 * `raw` for its expand/detail view so the original technical phrasing is
 * never lost, just not the first thing shown. `event` (the raw backend
 * event dict) is kept on the result for a genuinely unrecognized event
 * type, so its detail view can show its real fields instead of nothing.
 */
export function humanizeEvent(e) {
  if (!e || !e.event) return null
  let formatted
  let isUnknown = false
  if (e.event === 'tool_call') {
    formatted = formatToolCall(e)
  } else {
    const fn = EVENT_FORMATTERS[e.event]
    if (fn) {
      formatted = fn(e)
    } else {
      // A genuinely new/unrecognized event type -- shown plainly rather
      // than suppressed (an unrecognized event might still be
      // important), but never as raw JSON: a cleaned-up label plus a
      // flag so the UI can offer its real fields only on request.
      isUnknown = true
      formatted = { text: `Activity: ${e.event.replaceAll('_', ' ')}`, tone: 'slate' }
    }
  }
  if (!formatted) return null
  return {
    key: `${e.timestamp}-${e.event}-${e.run_id || e.key || ''}`,
    timestamp: e.timestamp,
    event: e,
    isUnknown,
    ...formatted,
    raw: formatted.raw ?? formatted.text,
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
