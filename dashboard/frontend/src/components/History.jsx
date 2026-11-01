import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { ArrowLeft, CheckCircle2, Clock, History as HistoryIcon, OctagonX, TriangleAlert, X } from 'lucide-react'
import { Badge, Button, Panel, PanelTitle, cn } from './ui'
import {
  DecisionTimeline, EventFeed, GoAuditCard, LineageCard, RunComparison, StressLabCard, TrainingHealthCard,
} from './MissionPanels'

/**
 * Mission history: every session launched from this dashboard.
 *
 * Reads /api/history (list) and /api/history/<id> (detail), both backed
 * by gotrain/history.py. A "session" is one press of Start -- the
 * grouping layer that ledger.jsonl and events.jsonl don't have.
 *
 * Rendered as an overlay rather than another always-on panel: history is
 * something you go look at, not something you monitor, and the live
 * dashboard is already dense.
 */

const STATUS_TONE = {
  RUNNING: 'sky',
  COMPLETED: 'green',
  STOPPED: 'amber',
  FAILED: 'red',
  UNKNOWN: 'slate',
}

const STATUS_ICON = {
  RUNNING: Clock,
  COMPLETED: CheckCircle2,
  STOPPED: OctagonX,
  FAILED: TriangleAlert,
}

const fmtDur = (s) => {
  if (s == null) return '--'
  const t = Math.floor(s)
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const sec = t % 60
  if (h) return `${h}h ${m}m ${sec}s`
  if (m) return `${m}m ${String(sec).padStart(2, '0')}s`
  return `${sec}s`
}
const fmtPct = (v) => (typeof v === 'number' ? `${(v * 100).toFixed(2)}%` : '--')
const fmtClock = (iso) => (iso ? new Date(iso).toLocaleString() : '--')
const fmtTime = (iso) => (iso ? new Date(iso).toLocaleTimeString() : '--')

export function HistoryButton({ onClick }) {
  return (
    <Button variant="ghost" onClick={onClick} className="px-2.5 py-1 text-xs">
      <HistoryIcon className="h-3.5 w-3.5" /> History
    </Button>
  )
}

export function HistoryOverlay({ open, onClose }) {
  const [sessions, setSessions] = useState(null)
  const [detail, setDetail] = useState(null)
  const [loadingDetail, setLoadingDetail] = useState(false)

  useEffect(() => {
    if (!open) return
    setDetail(null)
    setSessions(null)
    fetch(new URL('api/history', window.location.href))
      .then((r) => r.json())
      .then((d) => setSessions(d.sessions || []))
      .catch(() => setSessions([]))
  }, [open])

  // Esc closes the detail view first, then the overlay -- standard
  // layered-dismiss behaviour.
  useEffect(() => {
    if (!open) return
    const onKey = (e) => {
      if (e.key !== 'Escape') return
      if (detail) setDetail(null)
      else onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, detail, onClose])

  const openDetail = async (sessionId) => {
    setLoadingDetail(true)
    try {
      const r = await fetch(new URL(`api/history/${sessionId}`, window.location.href))
      setDetail(r.ok ? await r.json() : null)
    } catch {
      setDetail(null)
    }
    setLoadingDetail(false)
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-40 overflow-y-auto bg-void/90 backdrop-blur-sm"
          onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
        >
          <div className="mx-auto max-w-[1100px] px-6 py-8">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                {detail && (
                  <Button variant="ghost" className="px-2 py-1 text-xs" onClick={() => setDetail(null)}>
                    <ArrowLeft className="h-3.5 w-3.5" /> All missions
                  </Button>
                )}
                <h2 className="font-display text-lg font-bold text-white">
                  {detail ? 'Mission Detail' : 'Mission History'}
                </h2>
              </div>
              <Button variant="ghost" className="px-2 py-1 text-xs" onClick={onClose}>
                <X className="h-3.5 w-3.5" /> Close
              </Button>
            </div>

            {detail ? (
              <SessionDetail session={detail} />
            ) : loadingDetail ? (
              <Panel><div className="px-4 py-8 text-center font-mono text-xs text-slate-500">loading…</div></Panel>
            ) : (
              <SessionList sessions={sessions} onOpen={openDetail} />
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

function SessionList({ sessions, onOpen }) {
  if (sessions === null) {
    return <Panel><div className="px-4 py-8 text-center font-mono text-xs text-slate-500">loading…</div></Panel>
  }
  if (!sessions.length) {
    return (
      <Panel>
        <div className="px-4 py-10 text-center">
          <HistoryIcon className="mx-auto mb-2 h-6 w-6 text-slate-600" />
          <div className="font-mono text-xs uppercase tracking-[0.15em] text-slate-500">No missions yet</div>
          <div className="mt-1 font-mono text-[0.62rem] text-slate-600">
            every mission launched from this dashboard is recorded here
          </div>
        </div>
      </Panel>
    )
  }

  return (
    <div className="space-y-2.5">
      {sessions.map((s) => {
        const Icon = STATUS_ICON[s.status] || Clock
        return (
          <motion.button
            key={s.session_id}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            whileHover={{ y: -1 }}
            onClick={() => onOpen(s.session_id)}
            className="block w-full rounded-xl border border-edge/80 bg-panel/70 px-4 py-3 text-left transition-colors hover:border-emerald-400/40"
          >
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2.5">
                <Icon className={cn(
                  'h-4 w-4',
                  s.status === 'COMPLETED' ? 'text-emerald-400'
                    : s.status === 'FAILED' ? 'text-rose-400'
                      : s.status === 'RUNNING' ? 'animate-pulse text-sky-400' : 'text-amber-400',
                )} />
                <span className="font-display text-sm font-bold text-white">{s.dataset}</span>
                <Badge tone={STATUS_TONE[s.status] || 'slate'}>{s.status}</Badge>
              </div>
              <span className="font-mono text-[0.6rem] text-slate-600">{fmtClock(s.started_at)}</span>
            </div>

            {s.status === 'STOPPED' && s.stop_reason && (
              <div className="mt-1.5 font-mono text-[0.64rem] text-amber-400/90">
                STOPPED — {s.stop_reason}
              </div>
            )}

            <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 font-mono text-[0.66rem] text-slate-400">
              <span>duration <span className="text-slate-200">{fmtDur(s.duration_seconds)}</span></span>
              <span>
                experiments{' '}
                <span className="text-slate-200">{s.runs_completed ?? 0} / {s.runs_requested ?? '?'}</span>
              </span>
              <span>
                best val{' '}
                <span className="text-emerald-300">{fmtPct(s.best_val_metric)}</span>
              </span>
              {s.best_model && <span>model <span className="text-sky-300">{s.best_model}</span></span>}
              {s.timed_out_runs > 0 && (
                <span className="text-amber-400">{s.timed_out_runs} timed out</span>
              )}
              {s.interventions?.length > 0 && (
                <span className="text-amber-400">{s.interventions.length} engineer action(s)</span>
              )}
            </div>
          </motion.button>
        )
      })}
    </div>
  )
}

function SessionDetail({ session: s }) {
  const curves = (s.live_metrics || []).map((m) => ({
    epoch: m.epoch,
    loss: m.train_loss,
    val: m.val_metric,
    run: (m.run_id || '').slice(0, 8),
  }))

  return (
    <div className="space-y-4">
      <Panel>
        <PanelTitle right={<Badge tone={STATUS_TONE[s.status] || 'slate'}>{s.status}</Badge>}>
          {s.dataset} · {s.session_id}
        </PanelTitle>
        {s.status === 'STOPPED' && s.stop_reason && (
          <div className="border-b border-edge/60 bg-amber-500/[0.06] px-4 py-2 font-mono text-[0.68rem] text-amber-300">
            STOPPED — {s.stop_reason}
          </div>
        )}
        <div className="grid grid-cols-2 gap-px bg-edge/50 sm:grid-cols-3 lg:grid-cols-4">
          <Field label="Dataset" value={s.dataset} />
          <Field label="Runs requested" value={s.runs_requested} />
          <Field label="Min / run" value={s.minutes_per_run} />
          <Field label="Max tool calls" value={s.max_tool_calls ?? '--'} />
          <Field label="Started" value={fmtTime(s.started_at)} sub={fmtClock(s.started_at).split(',')[0]} />
          <Field label="Finished" value={fmtTime(s.finished_at)} />
          <Field label="Duration" value={fmtDur(s.duration_seconds)} />
          <Field label="Experiments" value={`${s.runs_completed ?? 0} / ${s.runs_requested ?? '?'}`} />
          <Field label="Best validation" value={fmtPct(s.best_val_metric)} tone="text-emerald-300" />
          <Field label="Best model" value={s.best_model || '--'} tone="text-sky-300" />
          <Field
            label="Final submitted"
            value={s.final_model_run_id ? s.final_model_run_id.slice(0, 8) : 'none'}
            tone={s.final_model_run_id ? 'text-emerald-300' : 'text-slate-500'}
          />
          <Field
            label="Timed out"
            value={s.timed_out_runs ?? 0}
            tone={s.timed_out_runs ? 'text-amber-300' : 'text-slate-300'}
          />
        </div>
      </Panel>

      {/* Training Health, Decision Timeline, Run Comparison and Event Feed
          are the SAME components the live dashboard uses
          (dashboard/frontend/src/components/MissionPanels.jsx) over the
          SAME `mission` shape -- gotrain/history.py's load_session()
          output, where training_health was already computed and saved at
          _derive() time -- so a finished mission's History Details page
          can never show a different picture than what was visible while
          it was running. */}
      <TrainingHealthCard mission={s} />
      <DecisionTimeline mission={s} currentRunId={null} />
      <RunComparison mission={s} currentRun={null} />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <LineageCard dataset={s.dataset} />
        <GoAuditCard dataset={s.dataset} />
      </div>
      {/* StressLab now auto-runs on the SUBMITTED model (see
          gotrain.agent._auto_run_stresslab), not necessarily the highest
          val_metric run -- final_model_run_id is the correct run to show
          a report for. Falls back to best_run_id only for an older
          session that finished before this auto-run existed, so a report
          saved under the old manual-call flow still renders. */}
      <StressLabCard dataset={s.dataset} runId={s.final_model_run_id || s.best_run_id} />

      {curves.length > 0 && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <DetailChart title="Training loss / epoch" data={curves} dataKey="loss" color="#fbbf24" />
          <DetailChart title="Validation metric / epoch" data={curves} dataKey="val" color="#22c55e" domain={[0, 1]} />
        </div>
      )}

      <EventFeed mission={s} title={`Event Feed (${s.events?.length ?? 0} raw events)`} />
    </div>
  )
}

function Field({ label, value, sub, tone = 'text-slate-100' }) {
  return (
    <div className="bg-panel px-4 py-2.5">
      <div className="font-mono text-[0.56rem] uppercase tracking-[0.18em] text-slate-500">{label}</div>
      <div className={cn('mt-0.5 font-display text-sm font-bold tabular-nums', tone)}>{value}</div>
      {sub && <div className="font-mono text-[0.56rem] text-slate-600">{sub}</div>}
    </div>
  )
}

function DetailChart({ title, data, dataKey, color, domain }) {
  const has = data.some((d) => typeof d[dataKey] === 'number')
  return (
    <Panel>
      <PanelTitle>{title}</PanelTitle>
      <div className="h-48 px-2 py-3">
        {has ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 6, right: 12, bottom: 0, left: -20 }}>
              <CartesianGrid stroke="#1c2c44" strokeDasharray="3 3" />
              <XAxis dataKey="epoch" stroke="#475569" fontSize={9} tickLine={false} />
              <YAxis stroke="#475569" fontSize={9} tickLine={false} domain={domain || ['auto', 'auto']} />
              <Tooltip
                contentStyle={{
                  background: '#0d1829', border: '1px solid #1c2c44',
                  borderRadius: 8, fontSize: 11, fontFamily: 'monospace',
                }}
              />
              <Line
                type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2}
                dot={{ r: 2, fill: color }} isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="grid h-full place-items-center font-mono text-[0.62rem] uppercase tracking-wider text-slate-600">
            no epoch telemetry
          </div>
        )}
      </div>
    </Panel>
  )
}
