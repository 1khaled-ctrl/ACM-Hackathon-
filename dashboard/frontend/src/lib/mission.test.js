// Targeted tests for the Live Event Feed humanization in mission.js.
// Run with: node --test src/lib/mission.test.js
// (from dashboard/frontend/) -- uses Node's built-in test runner
// (node:test / node:assert), no new npm dependency. mission.js is pure
// JS with no React/network dependency, so it's directly importable here.
//
// Scope: humanizeEvent/buildEventFeed must (1) produce a friendly `text`
// for every real backend event type, (2) always preserve the original
// technical phrasing in `raw` so nothing is lost, (3) suppress noise
// (agent's own pause-ack, high-volume tool calls) exactly as before,
// (4) give unknown event types a clean fallback instead of raw JSON,
// and (5) dedupe. Backend event semantics are not touched by this file
// at all -- these tests only exercise the display layer.

import assert from 'node:assert/strict'
import { test } from 'node:test'

import { buildEventFeed, humanizeEvent } from './mission.js'

test('geoguard_finished reads as a dataset check, not a tool name, with findings', () => {
  const h = humanizeEvent({ timestamp: 't1', event: 'geoguard_finished', n_findings: 1 })
  assert.equal(h.text, 'Dataset check complete — 1 potential issue found')
  assert.ok(h.raw.includes('GeoGuard'))
})

test('geoguard_finished with zero findings says "no issues found"', () => {
  const h = humanizeEvent({ timestamp: 't1', event: 'geoguard_finished', n_findings: 0 })
  assert.equal(h.text, 'Dataset check complete — no issues found')
})

test('dataiq_structural_scan_finished with no issues matches the brief\'s example', () => {
  const h = humanizeEvent({ timestamp: 't1', event: 'dataiq_structural_scan_finished', n_invalid: 0 })
  assert.equal(h.text, 'Data quality check complete — no structural issues found')
})

test('training_started names the model in plain form', () => {
  const h = humanizeEvent({
    timestamp: 't1', event: 'training_started', config: { backbone: 'resnet34' }, task_type: 'classification',
  })
  assert.equal(h.text, 'Training started with ResNet34')
  assert.ok(h.raw.includes('resnet34'), 'raw must keep the literal config value')
})

test('training_started falls back to the raw string for an unrecognized model name', () => {
  const h = humanizeEvent({
    timestamp: 't1', event: 'training_started', config: { backbone: 'some_future_model' }, task_type: 'classification',
  })
  assert.equal(h.text, 'Training started with some_future_model')
})

test('training_finished matches the brief\'s example exactly', () => {
  const h = humanizeEvent({ timestamp: 't1', event: 'training_finished' })
  assert.equal(h.text, 'Training complete — checking model performance')
})

test('decision_recorded kept with a numeric-delta rationale names direction without inventing a field claim', () => {
  const h = humanizeEvent({
    timestamp: 't1', event: 'decision_recorded', run_id: 'abcdef12-3456',
    decision: 'kept', rationale: 'switched to adamw, val accuracy +2.1 vs parent, keeping',
  })
  assert.ok(h.text.startsWith('Experiment kept'))
  assert.ok(h.text.includes('improved'), `expected "improved" in: ${h.text}`)
  // The real rationale text must still be reachable, not discarded.
  assert.ok(h.raw.includes('+2.1'))
})

test('decision_recorded discarded with a negative delta says "hurt", not "improved"', () => {
  const h = humanizeEvent({
    timestamp: 't1', event: 'decision_recorded', run_id: 'abcdef12',
    decision: 'discarded', rationale: 'tried larger batch size, val accuracy -3.4 vs parent, discarding',
  })
  assert.ok(h.text.startsWith('Experiment discarded'))
  assert.ok(h.text.includes('hurt'), `expected "hurt" in: ${h.text}`)
})

test('decision_recorded with a rationale that has no parseable delta truncates rather than fabricates', () => {
  const h = humanizeEvent({
    timestamp: 't1', event: 'decision_recorded', run_id: 'abcdef12',
    decision: 'kept', rationale: 'this is the first baseline run for this dataset',
  })
  assert.ok(h.text.includes('this is the first baseline run'))
  assert.ok(!h.text.includes('improved') && !h.text.includes('hurt'))
})

test('control_transition: engineer manual control matches the brief\'s example', () => {
  const h = humanizeEvent({
    timestamp: 't1', event: 'control_transition', to_state: 'MANUAL_CONTROL', requested_by: 'engineer',
  })
  assert.equal(h.text, 'Engineer took control of the agent')
})

test('control_transition: engineer resume matches the brief\'s example', () => {
  const h = humanizeEvent({
    timestamp: 't1', event: 'control_transition', to_state: 'RESUME_REQUESTED', requested_by: 'engineer',
  })
  assert.equal(h.text, 'Engineer returned control to the autonomous agent')
})

test('control_transition from the agent itself (pause acknowledgement) is suppressed as noise', () => {
  const h = humanizeEvent({
    timestamp: 't1', event: 'control_transition', to_state: 'PAUSED_FOR_ENGINEER', requested_by: 'agent',
  })
  assert.equal(h, null)
})

test('a tool_call refused while engineer-controlled reads as a paused activity, matching the brief', () => {
  const h = humanizeEvent({
    timestamp: 't1', event: 'tool_call', tool: 'run_geoguard', status: 'error',
    result_truncated: "GoTrain is currently MANUAL_CONTROL (engineer-controlled) — no tool calls are permitted right now.",
  })
  assert.equal(h.text, 'Dataset scan paused while engineer control is active')
  assert.ok(h.raw.includes('run_geoguard refused'))
})

test('a high-volume tool_call (query_history) is suppressed entirely', () => {
  const h = humanizeEvent({ timestamp: 't1', event: 'tool_call', tool: 'query_history', status: 'success' })
  assert.equal(h, null)
})

test('launch_training tool_call is suppressed (training_started/finished already cover it)', () => {
  const h = humanizeEvent({ timestamp: 't1', event: 'tool_call', tool: 'launch_training', status: 'success' })
  assert.equal(h, null)
})

test('an unrecognized event type gets a clean fallback label, never raw JSON as the headline', () => {
  const h = humanizeEvent({ timestamp: 't1', event: 'some_future_event_type', weird_field: { nested: true } })
  assert.ok(h !== null)
  assert.ok(!h.text.includes('{'), 'the friendly text must never contain raw JSON braces')
  assert.equal(h.isUnknown, true)
  // The original event is still reachable for a detail view, just not
  // dumped directly into the headline.
  assert.deepEqual(h.event.weird_field, { nested: true })
})

test('an event with no "event" field at all is dropped, not crashed on', () => {
  assert.equal(humanizeEvent(null), null)
  assert.equal(humanizeEvent({}), null)
  assert.equal(humanizeEvent({ timestamp: 't1' }), null)
})

test('every formatted entry carries a raw/technical message distinct from or equal to text, never undefined', () => {
  const events = [
    { timestamp: 't1', event: 'agent_initialized', dataset: 'm-eurosat', max_runs: 3, run_minutes: 5, max_tool_calls: 40 },
    { timestamp: 't2', event: 'geoguard_started', dataset: 'm-eurosat' },
    { timestamp: 't3', event: 'final_model_submitted', run_id: 'abcdef12' },
    { timestamp: 't4', event: 'rollback_succeeded', dataset: 'm-eurosat', version: 'v2' },
  ]
  for (const e of events) {
    const h = humanizeEvent(e)
    assert.ok(h, `expected a formatted entry for ${e.event}`)
    assert.ok(typeof h.raw === 'string' && h.raw.length > 0, `missing raw text for ${e.event}`)
  }
})

test('buildEventFeed dedupes identical events and returns newest first', () => {
  const events = [
    { timestamp: 't1', event: 'geoguard_started', dataset: 'm-eurosat' },
    { timestamp: 't2', event: 'geoguard_finished', dataset: 'm-eurosat', n_findings: 0 },
    { timestamp: 't2', event: 'geoguard_finished', dataset: 'm-eurosat', n_findings: 0 }, // exact duplicate
  ]
  const feed = buildEventFeed(events)
  assert.equal(feed.length, 2, 'the exact-duplicate event must be deduped')
  assert.equal(feed[0].timestamp, 't2', 'newest event must come first')
  assert.equal(feed[1].timestamp, 't1')
})

test('buildEventFeed on an empty/undefined event list returns an empty array, not an error', () => {
  assert.deepEqual(buildEventFeed([]), [])
  assert.deepEqual(buildEventFeed(undefined), [])
})
