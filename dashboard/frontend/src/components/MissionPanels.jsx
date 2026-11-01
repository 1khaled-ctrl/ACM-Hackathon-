import { useMemo } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Award, Radio, XCircle } from 'lucide-react'
import { Badge, Panel, PanelTitle, cn } from './ui'
import { buildEventFeed, buildTimeline, finalEpochStatsByRun, isLikelyDiminishingReturns, normalizeLiveRun } from '../lib/mission'

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

  return (
    <Panel>
      <PanelTitle right={<Radio className="h-3.5 w-3.5 text-emerald-400" />}>{title}</PanelTitle>
      <div className="max-h-[26rem] divide-y divide-edge/40 overflow-y-auto">
        <AnimatePresence initial={false}>
          {feed.map((item) => (
            <motion.div
              key={item.key}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-start gap-2.5 px-4 py-2"
            >
              <span className={cn('mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full', DOT_TONE[item.tone] || DOT_TONE.slate)} />
              <div className="min-w-0 flex-1">
                <div className="font-mono text-[0.68rem] text-slate-200">{item.text}</div>
              </div>
              <span className="shrink-0 font-mono text-[0.58rem] text-slate-600">{fmtTime(item.timestamp)}</span>
            </motion.div>
          ))}
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
