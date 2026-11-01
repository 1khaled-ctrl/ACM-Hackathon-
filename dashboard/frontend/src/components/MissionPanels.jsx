import { useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity, Award, ChevronDown, ClipboardList, GitFork, HeartPulse, Radio, ShieldAlert, XCircle,
} from 'lucide-react'
import { Badge, Panel, PanelTitle, cn } from './ui'
import { buildEventFeed, buildTimeline, finalEpochStatsByRun, isLikelyDiminishingReturns, normalizeLiveRun } from '../lib/mission'

function useJson(url, deps) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    if (!url) { setData(null); setLoading(false); return }
    let cancelled = false
    setLoading(true)
    fetch(new URL(url, window.location.href))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (!cancelled) { setData(d); setLoading(false) } })
      .catch(() => { if (!cancelled) { setData(null); setLoading(false) } })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return { data, loading }
}

/**
 * Three panels built on one `mission` object -- the exact shape
 * dashboard/api/server.py's build_current_mission() and
 * gotrain/history.py's load_session() both return (same function,
 * called from two places), so a live mission and its later History
 * Details page can never show different runs for the same mission.
 *
 * mission = {
 *   session_id, dataset, status, started_at, finished_at,
 *   runs_requested, minutes_per_run, max_tool_calls,
 *   runs: [...ledger records, already scoped to this session...],
 *   events: [...events.jsonl + approvals.jsonl, scoped + deduped...],
 *   best_run_id, final_model_run_id, timed_out_runs, interventions, ...
 * }
 */

const fmtPct = (v) => (typeof v === 'number' ? `${(v * 100).toFixed(2)}%` : '--')
const fmtNum4 = (v) => (typeof v === 'number' ? v.toFixed(4) : '--')
const fmtTime = (iso) => (iso ? new Date(iso).toLocaleTimeString() : '--')
const modelLabel = (config) => (config && (config.backbone || config.encoder || config.architecture)) || '--'

function runIndexLookup(runs) {
  const map = new Map()
  ;(runs || []).forEach((r, i) => map.set(r.run_id, i))
  return map
}

// ---------------------------------------------------------------------
// Training Health -- compact card over the SAME diagnoses
// gotrain/experiment_planner.py's get_training_diagnostics already
// computes (no new heuristic here). Computed server-side, once, in
// gotrain/history.py's _derive() (mission.training_health), so this
// component only renders -- it never re-derives the state itself, and
// is identical whether `mission` came from the live snapshot or from
// History Details.
// ---------------------------------------------------------------------

const HEALTH_TONE = {
  HEALTHY: 'green',
  OVERFITTING: 'amber',
  UNDERFITTING: 'amber',
  UNSTABLE: 'red',
  UNKNOWN: 'slate',
}

export function TrainingHealthCard({ mission }) {
  const health = mission?.training_health
  const state = health?.state || 'UNKNOWN'
  const evidence = health?.evidence || []

  return (
    <Panel>
      <PanelTitle right={<Badge tone={HEALTH_TONE[state] || 'slate'}>{state}</Badge>}>
        <span className="inline-flex items-center gap-1.5">
          <HeartPulse className="h-3 w-3" /> Training Health
        </span>
      </PanelTitle>
      <div className="px-4 py-2.5">
        {evidence.length > 0 ? (
          <ul className="space-y-0.5">
            {evidence.map((line, i) => (
              <li key={i} className="flex items-start gap-1.5 font-mono text-[0.64rem] text-slate-400">
                <Activity className="mt-0.5 h-2.5 w-2.5 shrink-0 text-slate-600" />
                {line}
              </li>
            ))}
          </ul>
        ) : (
          <div className="font-mono text-[0.64rem] text-slate-600">
            {state === 'UNKNOWN'
              ? 'no completed run to diagnose yet'
              : 'no concerning signal in the latest completed run'}
          </div>
        )}
      </div>
    </Panel>
  )
}

// ---------------------------------------------------------------------
// 1. Agent Decision Timeline
// ---------------------------------------------------------------------

export function DecisionTimeline({ mission, currentRunId }) {
  const items = useMemo(() => buildTimeline(mission), [mission])
  const runsById = useMemo(() => {
    const m = new Map()
    ;(mission?.runs || []).forEach((r) => m.set(r.run_id, r))
    return m
  }, [mission])
  const indexById = useMemo(() => runIndexLookup(mission?.runs), [mission])

  return (
    <Panel>
      <PanelTitle right={<Badge>{items.length} {items.length === 1 ? 'entry' : 'entries'}</Badge>}>
        Agent Decision Timeline
      </PanelTitle>
      <div className="max-h-[28rem] overflow-y-auto px-4 py-3">
        {!mission ? (
          <Empty>no mission launched yet</Empty>
        ) : items.length === 0 ? (
          <Empty>
            {mission.status === 'RUNNING' ? 'no experiments yet -- the agent is still inspecting the dataset' : 'no experiments were recorded for this mission'}
          </Empty>
        ) : (
          <ol className="relative space-y-4 border-l border-edge pl-5">
            {items.map((item, i) =>
              item.kind === 'run' ? (
                <TimelineRunEntry
                  key={item.data.run_id || i}
                  run={item.data}
                  index={item.index}
                  parent={item.data.parent_id ? runsById.get(item.data.parent_id) : null}
                  isLive={item.data.run_id === currentRunId}
                  isBest={item.data.run_id === mission.best_run_id}
                  isSubmitted={item.data.run_id === mission.final_model_run_id}
                />
              ) : (
                <TimelineFailureEntry key={`${item.kind}-${item.timestamp}-${i}`} entry={item.data} kind={item.kind} />
              ),
            )}
            {mission.status === 'RUNNING' && currentRunId && !runsById.has(currentRunId) && (
              <LiveRunEntry />
            )}
          </ol>
        )}
      </div>
    </Panel>
  )
}

function TimelineRunEntry({ run, index, parent, isLive, isBest, isSubmitted }) {
  const h = run.hypothesis
  const outcomeTone = run.hypothesis_outcome === 'supported' ? 'green'
    : run.hypothesis_outcome === 'not_supported' ? 'red' : 'slate'
  const diminishing = isLikelyDiminishingReturns(run, parent)

  return (
    <li className="relative">
      <span
        className={cn(
          'absolute -left-[1.56rem] top-1.5 h-2.5 w-2.5 rounded-full',
          isSubmitted ? 'bg-emerald-400 shadow-glow' : isBest ? 'bg-sky-400' : 'bg-slate-600',
        )}
      />
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-[0.68rem] font-semibold text-slate-200">
          Run #{index + 1} -- {(run.run_id || '').slice(0, 8)}
        </span>
        {isSubmitted && <Badge tone="green">submitted</Badge>}
        {isBest && !isSubmitted && <Badge tone="sky">best so far</Badge>}
        {diminishing && <Badge tone="amber">diminishing returns</Badge>}
        <span className="ml-auto font-mono text-[0.58rem] text-slate-600">{fmtTime(run.timestamp)}</span>
      </div>

      {h ? (
        <div className="mt-1.5 rounded-lg border border-edge/60 bg-slate-900/40 px-3 py-2">
          <div className="font-mono text-[0.62rem] text-slate-400">
            <span className="text-slate-500">hypothesis:</span> {h.change || '--'}
          </div>
          <div className="font-mono text-[0.6rem] text-slate-500">{h.why}</div>
        </div>
      ) : (
        <div className="mt-1.5 font-mono text-[0.6rem] text-slate-500">baseline run -- no parent to hypothesize against</div>
      )}

      <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[0.62rem] text-slate-500">
        <span>model: <span className="text-sky-300">{modelLabel(run.config)}</span></span>
        {run.config_diff && Object.keys(run.config_diff).length > 0 && (
          <span>
            changed:{' '}
            {Object.entries(run.config_diff).slice(0, 3).map(([k, v]) => (
              <span key={k} className="text-slate-400">{k}={String(v?.to ?? v)} </span>
            ))}
          </span>
        )}
      </div>

      <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1">
        <span className="font-display text-sm font-bold tabular-nums text-emerald-300">
          {fmtNum4(run.val_metric)}
        </span>
        {typeof run.delta_vs_parent === 'number' && (
          <span className={cn('font-mono text-[0.62rem]', run.delta_vs_parent >= 0 ? 'text-emerald-400' : 'text-rose-400')}>
            {run.delta_vs_parent >= 0 ? '+' : ''}{(run.delta_vs_parent * 100).toFixed(2)}pp vs parent
          </span>
        )}
        {run.hypothesis_outcome && <Badge tone={outcomeTone}>{run.hypothesis_outcome.replaceAll('_', ' ')}</Badge>}
        {run.decision && (
          <Badge tone={run.decision === 'kept' ? 'green' : run.decision === 'discarded' ? 'slate' : 'amber'}>
            {run.decision}
          </Badge>
        )}
      </div>

      {run.rationale && (
        <div className="mt-1 font-mono text-[0.6rem] text-slate-500">"{run.rationale}"</div>
      )}
      {isLive && <div className="mt-1 font-mono text-[0.58rem] uppercase tracking-wider text-sky-400">training now...</div>}
    </li>
  )
}

function TimelineFailureEntry({ entry, kind }) {
  const isTimeout = kind === 'training_timed_out'
  return (
    <li className="relative">
      <span className="absolute -left-[1.56rem] top-1.5 h-2.5 w-2.5 rounded-full bg-rose-500" />
      <div className="flex flex-wrap items-center gap-2">
        <XCircle className="h-3.5 w-3.5 text-rose-400" />
        <span className="font-mono text-[0.68rem] font-semibold text-rose-300">
          {isTimeout ? 'Training timed out' : 'Training failed'}
        </span>
        <span className="ml-auto font-mono text-[0.58rem] text-slate-600">{fmtTime(entry.timestamp)}</span>
      </div>
      <div className="mt-1 font-mono text-[0.6rem] text-slate-500">
        {isTimeout
          ? `exceeded its ${entry.timeout_seconds ? Math.round(entry.timeout_seconds / 60) + 'min' : ''} time budget -- no checkpoint recorded`
          : 'subprocess exited with an error -- no checkpoint recorded'}
      </div>
    </li>
  )
}

function LiveRunEntry() {
  return (
    <li className="relative">
      <span className="absolute -left-[1.56rem] top-1.5 h-2.5 w-2.5 animate-pulse rounded-full bg-sky-400" />
      <div className="font-mono text-[0.68rem] text-sky-300">training in progress...</div>
    </li>
  )
}

// ---------------------------------------------------------------------
// 2. Run Comparison
// ---------------------------------------------------------------------

export function RunComparison({ mission, currentRun }) {
  const runs = mission?.runs || []
  const normalizedLive = normalizeLiveRun(currentRun, mission)
  const liveRow = normalizedLive && !runs.some((r) => r.run_id === normalizedLive.run_id) ? normalizedLive : null
  const rows = liveRow ? [...runs, liveRow] : runs
  const requested = mission?.runs_requested
  const epochStatsByRun = useMemo(() => finalEpochStatsByRun(mission?.live_metrics), [mission])

  return (
    <Panel>
      <PanelTitle
        right={
          requested != null && (
            <Badge tone={rows.length < requested ? 'amber' : 'slate'}>
              {rows.length} / {requested} runs
            </Badge>
          )
        }
      >
        Run Comparison
      </PanelTitle>
      {!mission ? (
        <Empty>no mission launched yet</Empty>
      ) : rows.length === 0 ? (
        <Empty>no experiments yet for this mission</Empty>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left">
            <thead>
              <tr className="border-b border-edge/60 font-mono text-[0.58rem] uppercase tracking-wider text-slate-500">
                <th className="px-3 py-2">#</th>
                <th className="px-3 py-2">Model / Change</th>
                <th className="px-3 py-2">Val Metric</th>
                <th className="px-3 py-2">Train Loss</th>
                <th className="px-3 py-2">Duration</th>
                <th className="px-3 py-2">vs Parent</th>
                <th className="px-3 py-2">Decision</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => {
                const isLiveRow = liveRow && r.run_id === liveRow.run_id
                // The live row's own train_loss/elapsed_seconds (fresh every
                // second via the WebSocket) are more current than
                // live_metrics.jsonl's last written epoch, which can lag by
                // up to one epoch behind.
                const epochStats = isLiveRow ? { train_loss: r.train_loss, elapsed_seconds: r.elapsed_seconds } : epochStatsByRun.get(r.run_id)
                const durationSeconds = epochStats?.elapsed_seconds ?? (typeof r.gpu_minutes === 'number' ? r.gpu_minutes * 60 : null)
                return (
                  <ComparisonRow
                    key={r.run_id || i}
                    run={r}
                    index={i}
                    trainLoss={epochStats?.train_loss}
                    durationSeconds={durationSeconds}
                    isBest={r.run_id === mission.best_run_id}
                    isSubmitted={r.run_id === mission.final_model_run_id}
                    isLive={isLiveRow}
                  />
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  )
}

// Statistical Improvement Verification (gotrain/statistics.py, folded
// into mission.runs[].statistical_verdict by
// gotrain/history.py::_attach_statistical_verdicts -- same tri-state
// is_improvement_significant() output, just named/toned for display).
const STAT_VERDICT_TONE = {
  VERIFIED: 'green',
  NOT_ABOVE_NOISE: 'amber',
  INSUFFICIENT_EVIDENCE: 'slate',
}
const STAT_VERDICT_LABEL = {
  VERIFIED: 'verified',
  NOT_ABOVE_NOISE: 'not above noise',
  INSUFFICIENT_EVIDENCE: 'insufficient evidence',
}

function StatVerdictBadge({ verdict }) {
  const state = verdict?.state
  if (!state) return null
  return (
    <div className="mt-0.5" title={verdict.reason || ''}>
      <Badge tone={STAT_VERDICT_TONE[state] || 'slate'}>{STAT_VERDICT_LABEL[state] || state}</Badge>
    </div>
  )
}

function ComparisonRow({ run, index, trainLoss, durationSeconds, isBest, isSubmitted, isLive }) {
  const changeKeys = run.config_diff ? Object.keys(run.config_diff) : []
  const firstChange = changeKeys.length ? run.config_diff[changeKeys[0]] : null
  const change = changeKeys.length
    ? `${changeKeys[0]}=${String(firstChange?.to ?? firstChange)}`
    : run.parent_id ? 'config changed' : 'baseline'

  return (
    <tr
      className={cn(
        'border-b border-edge/30 font-mono text-[0.68rem]',
        isSubmitted ? 'bg-emerald-500/[0.06]' : isBest ? 'bg-sky-500/[0.05]' : '',
      )}
    >
      <td className="px-3 py-2 text-slate-500">{index + 1}</td>
      <td className="px-3 py-2">
        <div className="text-slate-200">{modelLabel(run.config)}</div>
        <div className="text-[0.58rem] text-slate-500">{change}</div>
      </td>
      <td className="px-3 py-2">
        {isLive ? (
          <span className="text-sky-300">{typeof run.val_metric === 'number' ? fmtNum4(run.val_metric) : 'running...'}</span>
        ) : (
          <span className="font-bold text-emerald-300">{fmtNum4(run.val_metric)}</span>
        )}
      </td>
      <td className="px-3 py-2 text-amber-300">{fmtNum4(trainLoss)}</td>
      <td className="px-3 py-2 text-slate-400">
        {typeof durationSeconds === 'number' ? `${Math.round(durationSeconds)}s` : '--'}
      </td>
      <td className="px-3 py-2">
        {typeof run.delta_vs_parent === 'number' ? (
          <span className={run.delta_vs_parent >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
            {run.delta_vs_parent >= 0 ? '+' : ''}{(run.delta_vs_parent * 100).toFixed(2)}pp
          </span>
        ) : '--'}
        {run.statistical_verdict && <StatVerdictBadge verdict={run.statistical_verdict} />}
      </td>
      <td className="px-3 py-2">
        <div className="flex items-center gap-1.5">
          {isSubmitted && <Award className="h-3 w-3 text-emerald-400" />}
          <Badge tone={isLive ? 'sky' : run.decision === 'kept' ? 'green' : run.decision === 'discarded' ? 'slate' : 'amber'}>
            {isLive ? 'running' : run.decision || 'pending'}
          </Badge>
        </div>
      </td>
    </tr>
  )
}

// ---------------------------------------------------------------------
// 3. Live Event Feed
// ---------------------------------------------------------------------

const DOT_TONE = {
  sky: 'bg-sky-400', green: 'bg-emerald-400', amber: 'bg-amber-400', red: 'bg-rose-400', slate: 'bg-slate-500',
}

export function EventFeed({ mission, title = 'Live Event Feed', maxItems = 60 }) {
  const feed = useMemo(() => buildEventFeed(mission?.events).slice(0, maxItems), [mission, maxItems])
  const [expanded, setExpanded] = useState(() => new Set())

  const toggle = (key) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  return (
    <Panel>
      <PanelTitle right={<Radio className="h-3.5 w-3.5 text-emerald-400" />}>{title}</PanelTitle>
      <div className="max-h-[26rem] divide-y divide-edge/40 overflow-y-auto">
        <AnimatePresence initial={false}>
          {feed.map((item) => {
            // "Technical details" is only worth showing when it actually
            // differs from the friendly line -- several formatters set
            // text===raw (e.g. the "no events yet" style one-liners),
            // and an expand control with nothing new inside it is noise.
            const hasDetail = item.raw && item.raw !== item.text
            const isOpen = expanded.has(item.key)
            return (
              <motion.div
                key={item.key}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                className="px-4 py-2"
              >
                <div className="flex items-start gap-2.5">
                  <span className={cn('mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full', DOT_TONE[item.tone] || DOT_TONE.slate)} />
                  <button
                    type="button"
                    onClick={() => hasDetail && toggle(item.key)}
                    disabled={!hasDetail}
                    title={hasDetail ? item.raw : undefined}
                    className={cn(
                      'min-w-0 flex-1 text-left font-mono text-[0.68rem] text-slate-200',
                      hasDetail && 'cursor-pointer hover:text-white',
                    )}
                  >
                    {item.text}
                    {item.isUnknown && (
                      <span className="ml-1.5 font-mono text-[0.56rem] uppercase tracking-wider text-slate-600">
                        unrecognized event
                      </span>
                    )}
                  </button>
                  {hasDetail && (
                    <ChevronDown
                      className={cn('mt-0.5 h-3 w-3 shrink-0 text-slate-600 transition-transform', isOpen && 'rotate-180')}
                    />
                  )}
                  <span className="shrink-0 font-mono text-[0.58rem] text-slate-600">{fmtTime(item.timestamp)}</span>
                </div>
                {hasDetail && isOpen && (
                  <div className="ml-4 mt-1 rounded-md border border-edge/50 bg-slate-900/50 px-2.5 py-1.5 font-mono text-[0.6rem] text-slate-400">
                    {item.raw}
                  </div>
                )}
              </motion.div>
            )
          })}
        </AnimatePresence>
        {feed.length === 0 && <Empty>no events yet</Empty>}
      </div>
    </Panel>
  )
}

const Empty = ({ children }) => (
  <div className="grid place-items-center px-4 py-8 text-center font-mono text-[0.62rem] uppercase tracking-wider text-slate-600">
    {children}
  </div>
)

// Same small stat-tile as History.jsx's own Field (not imported from
// there -- that component isn't exported, and duplicating this tiny a
// presentational piece is simpler than exporting it across files).
function Field({ label, value, sub, tone = 'text-slate-100' }) {
  return (
    <div className="bg-panel px-4 py-2.5">
      <div className="font-mono text-[0.56rem] uppercase tracking-[0.18em] text-slate-500">{label}</div>
      <div className={cn('mt-0.5 font-display text-sm font-bold tabular-nums', tone)}>{value}</div>
      {sub && <div className="font-mono text-[0.56rem] text-slate-600">{sub}</div>}
    </div>
  )
}

// ---------------------------------------------------------------------
// 4. Recursive Self-Improvement -- generation/lineage view.
// GET /api/lineage/{dataset} (gotrain.ledger.get_lineage_tree, enriched
// server-side with each run's own metrics.json warm-start record).
// Spans ALL missions for the dataset on purpose: a later mission can
// warm-start from an earlier mission's checkpoint, so a single-mission
// view would silently hide real parent/child edges.
// ---------------------------------------------------------------------

export function LineageCard({ dataset }) {
  const { data, loading } = useJson(dataset ? `api/lineage/${encodeURIComponent(dataset)}` : null, [dataset])
  const runs = data?.runs || []

  return (
    <Panel>
      <PanelTitle right={<GitFork className="h-3.5 w-3.5 text-violet-400" />}>
        Recursive Self-Improvement
      </PanelTitle>
      {!dataset || loading ? (
        <Empty>{loading ? 'loading…' : 'no dataset selected'}</Empty>
      ) : runs.length === 0 ? (
        <Empty>no runs yet for this dataset</Empty>
      ) : (
        <div className="max-h-80 overflow-y-auto px-4 py-3">
          <ul className="space-y-2">
            {runs.map((r) => (
              <li
                key={r.run_id}
                className="rounded-lg border border-edge/60 bg-slate-900/40 px-3 py-2"
                style={{ marginLeft: `${Math.min(r.generation, 6) * 16}px` }}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-[0.62rem] text-slate-300">
                    gen {r.generation} · {r.run_id.slice(0, 8)}
                  </span>
                  <span className="font-mono text-[0.6rem] text-sky-300">{modelLabel(r.config)}</span>
                  {r.version_id && <Badge tone="slate">{r.version_id}</Badge>}
                  {r.warm_started_from ? (
                    <Badge tone="green">warm-started</Badge>
                  ) : r.warm_start_requested ? (
                    <Badge tone="amber">warm-start requested, unconfirmed</Badge>
                  ) : null}
                  <span className="ml-auto font-mono text-[0.68rem] font-bold text-emerald-300">
                    {fmtPct(r.val_metric)}
                  </span>
                  {typeof r.delta_vs_parent === 'number' && (
                    <span className={cn('font-mono text-[0.6rem]', r.delta_vs_parent >= 0 ? 'text-emerald-400' : 'text-rose-400')}>
                      {r.delta_vs_parent >= 0 ? '+' : ''}{(r.delta_vs_parent * 100).toFixed(2)}pp
                    </span>
                  )}
                </div>
                {r.warm_started_from?.parent_run_id && (
                  <div className="mt-1 font-mono text-[0.58rem] text-slate-500">
                    warm-started from {r.warm_started_from.parent_run_id.slice(0, 8)}
                    {r.warm_started_from.parent_version ? ` (${r.warm_started_from.parent_version})` : ''}
                  </div>
                )}
                {r.children.length > 0 && (
                  <div className="mt-1 font-mono text-[0.58rem] text-slate-600">
                    {r.children.length} child run{r.children.length > 1 ? 's' : ''}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  )
}

// ---------------------------------------------------------------------
// 5. StressLab Robustness -- per-run report.
// GET /api/stresslab/{dataset}/{run_id} (gotrain.stresslab.runner,
// logs/stresslab/<dataset>/<run_id>.json). StressLab now runs
// automatically once, on the run submitted via submit_final_model (see
// gotrain.agent._auto_run_stresslab) -- never on every experiment. Three
// distinct "nothing to show" states, never inferred from clean
// validation accuracy alone:
//   NOT RUN       -- this run was never submitted as final (StressLab
//                     only triggers at submission time).
//   UNAVAILABLE   -- it WAS triggered but skipped on purpose (no
//                     StressLab path for this task_type, or no
//                     checkpoint was ever registered for this run).
//   FAILED        -- it WAS triggered and raised; the submission itself
//                     still succeeded, only the robustness check didn't.
// ---------------------------------------------------------------------

const FRAGILITY_TONE = { robust: 'green', fragile: 'amber', severely_fragile: 'red', unknown: 'slate' }

export function StressLabCard({ dataset, runId }) {
  const { data, loading } = useJson(
    dataset && runId ? `api/stresslab/${encodeURIComponent(dataset)}/${encodeURIComponent(runId)}` : null,
    [dataset, runId],
  )

  return (
    <Panel>
      <PanelTitle right={<ShieldAlert className="h-3.5 w-3.5 text-amber-400" />}>
        StressLab Robustness
      </PanelTitle>
      {!dataset || !runId ? (
        <Empty>no run selected</Empty>
      ) : loading ? (
        <Empty>loading…</Empty>
      ) : data?.unavailable ? (
        <Empty>UNAVAILABLE — {data.reason}</Empty>
      ) : data?.failed ? (
        <Empty>FAILED — {data.reason}</Empty>
      ) : !data || data.not_run ? (
        <Empty>NOT RUN — run_stresslab has not been called for this run</Empty>
      ) : (
        <div className="px-4 py-3">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <span className="font-mono text-[0.64rem] text-slate-400">run {runId.slice(0, 8)}</span>
            <Badge tone={data.summary.n_fragile > 0 ? 'amber' : 'green'}>{data.summary.overall_verdict}</Badge>
            <span className="ml-auto font-mono text-[0.6rem] text-slate-500">
              clean metric {fmtPct(data.summary.clean_metric)}
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left">
              <thead>
                <tr className="border-b border-edge/60 font-mono text-[0.56rem] uppercase tracking-wider text-slate-500">
                  <th className="px-2 py-1.5">Perturbation</th>
                  <th className="px-2 py-1.5">Severity</th>
                  <th className="px-2 py-1.5">Clean</th>
                  <th className="px-2 py-1.5">Perturbed</th>
                  <th className="px-2 py-1.5">Drop</th>
                  <th className="px-2 py-1.5">Fragility</th>
                </tr>
              </thead>
              <tbody>
                {(data.results || []).map((r, i) => (
                  <tr key={i} className="border-b border-edge/30 font-mono text-[0.64rem]">
                    <td className="px-2 py-1.5 text-slate-200">{r.perturbation}</td>
                    <td className="px-2 py-1.5 text-slate-400">{r.severity}</td>
                    <td className="px-2 py-1.5 text-slate-400">{fmtPct(r.clean_metric)}</td>
                    <td className="px-2 py-1.5 text-slate-400">{fmtPct(r.perturbed_metric)}</td>
                    <td className="px-2 py-1.5 text-amber-300">
                      {typeof r.relative_drop === 'number' ? `${(r.relative_drop * 100).toFixed(1)}%` : '--'}
                    </td>
                    <td className="px-2 py-1.5">
                      <Badge tone={FRAGILITY_TONE[r.fragility] || 'slate'}>{r.fragility}</Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Panel>
  )
}

// ---------------------------------------------------------------------
// 6. GoAudit -- latest engineering report for the dataset.
// GET /api/goaudit/{dataset} (gotrain.goaudit, logs/goaudit/<dataset>/).
// Historical reports are listed and selectable via GET
// /api/goaudit/{dataset}/{timestamp} -- never generated from the
// frontend, only ever read back from what generate_goaudit_report (the
// agent tool) already wrote.
// ---------------------------------------------------------------------

export function GoAuditCard({ dataset }) {
  const [selected, setSelected] = useState(null)
  const { data: latestWrap, loading: loadingLatest } = useJson(
    dataset ? `api/goaudit/${encodeURIComponent(dataset)}` : null, [dataset],
  )
  const { data: historicalReport, loading: loadingHistorical } = useJson(
    dataset && selected ? `api/goaudit/${encodeURIComponent(dataset)}/${encodeURIComponent(selected)}` : null,
    [dataset, selected],
  )

  const report = selected ? historicalReport : latestWrap?.report
  const loading = selected ? loadingHistorical : loadingLatest
  const history = latestWrap?.history || []

  return (
    <Panel>
      <PanelTitle
        right={
          history.length > 1 ? (
            <select
              className="rounded-md border border-edge bg-slate-900/70 px-2 py-1 font-mono text-[0.6rem] text-slate-200"
              value={selected || ''}
              onChange={(e) => setSelected(e.target.value || null)}
            >
              <option value="">latest</option>
              {history.slice().reverse().map((ts) => (
                <option key={ts} value={ts}>{ts}</option>
              ))}
            </select>
          ) : (
            <ClipboardList className="h-3.5 w-3.5 text-sky-400" />
          )
        }
      >
        GoAudit Report
      </PanelTitle>
      {!dataset ? (
        <Empty>no dataset selected</Empty>
      ) : loading ? (
        <Empty>loading…</Empty>
      ) : !report ? (
        <Empty>no GoAudit report generated yet for this dataset</Empty>
      ) : (
        <div className="space-y-3 px-4 py-3">
          <div className="grid grid-cols-2 gap-px bg-edge/50 sm:grid-cols-4">
            <Field label="Runs" value={report.progress.n_runs} />
            <Field label="Trend" value={report.progress.trend.replaceAll('_', ' ')} tone="text-sky-300" />
            <Field label="Best val" value={fmtPct(report.progress.best_val_metric)} tone="text-emerald-300" />
            <Field
              label="Noise evidence"
              value={report.statistical_confidence.has_data ? 'available' : 'none yet'}
              tone={report.statistical_confidence.has_data ? 'text-emerald-300' : 'text-slate-500'}
            />
          </div>

          {report.failures.length > 0 && (
            <div>
              <div className="mb-1 font-mono text-[0.58rem] uppercase tracking-wider text-rose-400">
                Failures ({report.failures.length})
              </div>
              <ul className="space-y-1">
                {report.failures.map((f, i) => (
                  <li key={i} className="font-mono text-[0.62rem] text-rose-300">
                    <Badge tone={f.severity === 'critical' ? 'red' : 'amber'}>{f.severity}</Badge> {f.message}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div>
            <div className="mb-1 font-mono text-[0.58rem] uppercase tracking-wider text-slate-500">
              Recommendations
            </div>
            <ul className="space-y-1">
              {report.recommendations.map((r, i) => (
                <li key={i} className="font-mono text-[0.62rem] text-slate-300">• {r}</li>
              ))}
            </ul>
          </div>

          {report.decisions.length > 0 && (
            <div>
              <div className="mb-1 font-mono text-[0.58rem] uppercase tracking-wider text-slate-500">
                Decisions ({report.decisions.length})
              </div>
              <div className="max-h-40 space-y-1 overflow-y-auto">
                {report.decisions.map((d, i) => (
                  <div key={i} className="font-mono text-[0.6rem] text-slate-400">
                    {d.run_id?.slice(0, 8)} — <span className="text-slate-300">{d.decision}</span>
                    {typeof d.val_metric === 'number' && ` · ${fmtPct(d.val_metric)}`}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="font-mono text-[0.56rem] text-slate-600">
            generated {fmtTime(report.generated_at)} · {report.statistical_confidence.note}
          </div>
        </div>
      )}
    </Panel>
  )
}
