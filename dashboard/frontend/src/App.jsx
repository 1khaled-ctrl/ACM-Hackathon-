import { useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity, AlertTriangle, Cpu, Database, GitBranch, LayoutGrid,
  LineChart as LineChartIcon, Satellite, Zap,
} from 'lucide-react'
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

import { Wordmark } from './components/Logo'
import { HistoryButton, HistoryOverlay } from './components/History'
import { Launcher } from './components/Launcher'
import { MissionControl } from './components/MissionControl'
import {
  DecisionTimeline, EventFeed, GoAuditCard, LineageCard, RunComparison, StressLabCard, TrainingHealthCard,
} from './components/MissionPanels'
import { Pipeline } from './components/Pipeline'
import { Backdrop, Badge, Meter, Panel, PanelTitle, Ring, Stat, cn } from './components/ui'

// Relative base so the app works at "/" locally and under SageMaker's
// /jupyterlab/default/proxy/8501/ prefix without a rebuild.
const API = (p) => new URL(p.replace(/^\//, ''), window.location.href).toString()

const STATE_TONE = {
  RUNNING_AUTONOMOUSLY: 'green',
  PAUSE_REQUESTED: 'amber',
  PAUSED_FOR_ENGINEER: 'sky',
  MANUAL_CONTROL: 'amber',
  RESUME_REQUESTED: 'sky',
  ROLLBACK_REQUESTED: 'amber',
  ROLLBACK_VERIFYING: 'amber',
  FAILED_SAFE: 'red',
  COMPLETED: 'green',
}

const fmtDur = (s) => {
  if (s == null) return '--'
  const t = Math.floor(s)
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const sec = t % 60
  if (h) return `${h}h ${m}m`
  if (m) return `${m}m ${sec}s`
  return `${sec}s`
}
const fmtNum = (v, d = 4) => (typeof v === 'number' ? v.toFixed(d) : '--')

export default function App() {
  const [snap, setSnap] = useState(null)
  const [connected, setConnected] = useState(false)
  const [datasets, setDatasets] = useState([])
  const [toast, setToast] = useState(null)
  // 'stats' = the per-epoch numbers; 'curves' = both live charts in the
  // same panel, so they're reachable above the fold without scrolling.
  const [runView, setRunView] = useState('stats')
  const [historyOpen, setHistoryOpen] = useState(false)
  const wsRef = useRef(null)

  // Live socket, with auto-reconnect so a server restart heals itself.
  useEffect(() => {
    let closed = false
    let retry
    const connect = () => {
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const url = `${proto}://${window.location.host}${window.location.pathname.replace(/\/$/, '')}/ws`
      const ws = new WebSocket(url)
      wsRef.current = ws
      ws.onopen = () => setConnected(true)
      ws.onmessage = (e) => { try { setSnap(JSON.parse(e.data)) } catch {} }
      ws.onclose = () => {
        setConnected(false)
        if (!closed) retry = setTimeout(connect, 1500)
      }
      ws.onerror = () => ws.close()
    }
    connect()
    return () => { closed = true; clearTimeout(retry); wsRef.current?.close() }
  }, [])

  useEffect(() => {
    fetch(API('api/datasets'))
      .then((r) => r.json())
      .then((d) => setDatasets(d.datasets || []))
      .catch(() => {})
  }, [])

  const notify = (msg, bad = false) => {
    setToast({ msg, bad })
    setTimeout(() => setToast(null), 4200)
  }

  const post = async (path, body) => {
    try {
      const r = await fetch(API(path), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body ?? {}),
      })
      const data = await r.json().catch(() => ({}))
      if (!r.ok) { notify(data.detail || `${r.status} error`, true); return null }
      return data
    } catch (e) {
      notify(String(e), true)
      return null
    }
  }

  const state = snap?.control_state
  const run = snap?.current_run || {}
  const live = snap?.live_metrics || []

  const chart = useMemo(
    () => live.map((r) => ({
      epoch: r.epoch,
      loss: r.train_loss,
      val: r.val_metric,
    })),
    [live],
  )

  const epochPct = run.epochs ? (run.epoch || 0) / run.epochs : 0

  // logs/status.json's current_run can lag behind (or outlive) a mission
  // boundary -- never attribute it to the mission panels below unless
  // its dataset actually matches the mission currently being shown. Same
  // guard MissionPanels' normalizeLiveRun applies internally; checked
  // again here because currentRunId is passed as a bare id string and
  // can't carry its own dataset check through to DecisionTimeline.
  const liveRunBelongsToMission = Boolean(
    run.run_id && snap?.mission && run.dataset === snap.mission.dataset,
  )

  // Set by POST /api/agent/start the instant a launch is requested --
  // before the new subprocess has done anything. Used to tell "stale tail
  // of the previous session" apart from events the agent just produced,
  // and to show a clear "launching" state instead of looking frozen or
  // (worse) looking like it finished instantly on leftover data.
  const launch = snap?.status?.launch
  const newestEventTimestamp = snap?.events?.[0]?.timestamp
  const launchPendingFirstEvent =
    snap?.agent_running && launch?.launched_at &&
    (!newestEventTimestamp || newestEventTimestamp < launch.launched_at)

  return (
    <div className="min-h-screen font-sans text-slate-200">
      <Backdrop />

      {/* ---- header ---- */}
      <header className="sticky top-0 z-20 border-b border-edge/70 bg-void/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-4 px-6 py-3">
          <Wordmark height={58} />
          <div className="flex items-center gap-2.5">
            <Badge tone={connected ? 'green' : 'red'} pulse={connected}>
              {connected ? 'live' : 'reconnecting'}
            </Badge>
            {state && <Badge tone={STATE_TONE[state] || 'slate'}>{state.replaceAll('_', ' ')}</Badge>}
            <Badge tone={snap?.agent_running ? 'sky' : 'slate'}>
              agent {snap?.agent_running ? 'up' : 'down'}
            </Badge>
            <HistoryButton onClick={() => setHistoryOpen(true)} />
          </div>
        </div>
      </header>

      <HistoryOverlay open={historyOpen} onClose={() => setHistoryOpen(false)} />

      <main className="mx-auto max-w-[1600px] space-y-4 px-6 pb-3 pt-5">
        <Pipeline snapshot={snap} />

        {/* ---- control deck ---- */}
        <MissionControl snap={snap} post={post} notify={notify} datasets={datasets} />

        <Panel>
          <PanelTitle>Launch</PanelTitle>
          <div className="px-4 py-3">
            <Launcher datasets={datasets} snap={snap} post={post} notify={notify} />
          </div>
        </Panel>

        {/* ---- headline stats + rings ---- */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          <Panel className="lg:col-span-8">
            <PanelTitle
              right={
                <div className="flex items-center gap-2.5">
                  <ViewSwitch value={runView} onChange={setRunView} />
                  {launchPendingFirstEvent
                    ? <Badge tone="sky" pulse>launching</Badge>
                    : run.run_id ? <Badge tone="green" pulse>training</Badge> : <Badge>idle</Badge>}
                </div>
              }
            >
              Current Run
            </PanelTitle>
            {runView === 'curves' ? (
              <div className="grid grid-cols-1 gap-px bg-edge/50 sm:grid-cols-2">
                <MiniChart
                  title="Training loss"
                  data={chart}
                  dataKey="loss"
                  color="#fbbf24"
                />
                <MiniChart
                  title={run.metric_name || 'Validation metric'}
                  data={chart}
                  dataKey="val"
                  color="#22c55e"
                  domain={[0, 1]}
                />
              </div>
            ) : launchPendingFirstEvent ? (
              <LaunchingNotice dataset={launch?.dataset} />
            ) : run.run_id ? (
              <div className="grid grid-cols-2 divide-x divide-y divide-edge/50 sm:grid-cols-3 lg:grid-cols-4">
                <Stat label="Dataset" value={run.dataset || '--'} sub={run.task_type} />
                <Stat label="Model" value={run.model || '--'} sub={run.optimizer} />
                <Stat
                  label="Epoch"
                  value={run.epochs ? `${run.epoch ?? 0}/${run.epochs}` : '--'}
                  sub={run.scheduler ? `sched ${run.scheduler}` : null}
                />
                <Stat label="Train loss" value={fmtNum(run.train_loss)} tone="text-amber-300" />
                <Stat
                  label={run.metric_name || 'val metric'}
                  value={fmtNum(run.val_metric)}
                  tone="text-emerald-300"
                  sub="VALIDATION split"
                />
                <Stat label="Best so far" value={fmtNum(run.best_val_metric)} tone="text-emerald-300" />
                <Stat label="Elapsed" value={fmtDur(run.elapsed_seconds)} />
                <Stat label="ETA" value={fmtDur(snap?.eta_seconds)} tone="text-sky-300" />
              </div>
            ) : (
              <IdleNotice agentRunning={snap?.agent_running} />
            )}
          </Panel>

          <Panel className="lg:col-span-4">
            <PanelTitle>Telemetry</PanelTitle>
            <div className="flex items-center justify-around px-2 py-4">
              <Ring
                value={epochPct}
                label={run.run_id ? (run.epochs ? `${Math.round(epochPct * 100)}%` : '--') : '—'}
                sub="run"
              />
              <Ring
                value={typeof run.val_metric === 'number' ? run.val_metric : 0}
                label={run.run_id ? fmtNum(run.val_metric, 3) : '—'}
                sub="val"
                tone="#22c55e"
                size={96}
                stroke={7}
              />
            </div>
            <div className="border-t border-edge/50 pb-3">
              <Meter label="CPU" percent={snap?.telemetry?.cpu_percent} />
              <Meter label="RAM" percent={snap?.telemetry?.ram_percent} tone="from-sky-400 to-indigo-500" />
              <Meter
                label="GPU"
                percent={snap?.telemetry?.gpu?.util_percent}
                tone="from-violet-400 to-fuchsia-500"
              />
              {snap?.telemetry?.gpu ? (
                <>
                  <Meter
                    label="VRAM"
                    percent={
                      snap.telemetry.gpu.mem_total_gb
                        ? (snap.telemetry.gpu.mem_used_gb / snap.telemetry.gpu.mem_total_gb) * 100
                        : null
                    }
                    valueLabel={`${snap.telemetry.gpu.mem_used_gb} / ${snap.telemetry.gpu.mem_total_gb} GB`}
                    tone="from-amber-400 to-rose-400"
                  />
                  <div className="flex items-center gap-1.5 px-4 pt-0.5 font-mono text-[0.58rem] uppercase tracking-wider text-slate-500">
                    <Zap className="h-3 w-3" /> {snap.telemetry.gpu.name}
                  </div>
                </>
              ) : (
                <div className="flex items-center gap-1.5 px-4 pt-1 font-mono text-[0.6rem] uppercase tracking-wider text-slate-600">
                  <Cpu className="h-3 w-3" /> no cuda device
                </div>
              )}
            </div>
          </Panel>
        </div>

        {/* ---- charts ---- */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <ChartPanel
            title="Training loss / epoch"
            data={chart}
            dataKey="loss"
            color="#fbbf24"
            icon={<Activity className="h-3.5 w-3.5" />}
          />
          <ChartPanel
            title="Validation metric / epoch"
            data={chart}
            dataKey="val"
            color="#22c55e"
            icon={<GitBranch className="h-3.5 w-3.5" />}
            domain={[0, 1]}
          />
        </div>

        {/* ---- training health (compact) ---- */}
        <TrainingHealthCard mission={snap?.mission} />

        {/* ---- decision timeline (full width -- hypothesis -> config -> result -> outcome -> decision) ---- */}
        <DecisionTimeline mission={snap?.mission} currentRunId={liveRunBelongsToMission ? run.run_id : null} />

        {/* ---- run comparison (full width table) ---- */}
        <RunComparison mission={snap?.mission} currentRun={liveRunBelongsToMission ? run : null} />

        {/* ---- recursive self-improvement lineage + GoAudit (dataset-wide, span missions) ---- */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <LineageCard dataset={snap?.mission?.dataset} />
          <GoAuditCard dataset={snap?.mission?.dataset} />
        </div>

        {/* ---- StressLab robustness for the mission's best run so far ---- */}
        <StressLabCard dataset={snap?.mission?.dataset} runId={snap?.mission?.best_run_id} />

        {/* ---- live event feed + datasets ---- */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          <div className="lg:col-span-9">
            <EventFeed
              mission={snap?.mission}
              title={launchPendingFirstEvent ? `Live Event Feed — launching on ${launch?.dataset}…` : 'Live Event Feed'}
            />
          </div>

          <Panel className="lg:col-span-3">
            <PanelTitle right={<Database className="h-3.5 w-3.5 text-sky-400" />}>
              Datasets
            </PanelTitle>
            <div className="max-h-[22rem] divide-y divide-edge/40 overflow-y-auto">
              {datasets.length ? (
                datasets.map((d) => (
                  <div key={d.dataset} className="px-4 py-2.5">
                    <div className="font-mono text-[0.72rem] text-slate-200">{d.dataset}</div>
                    <div className="font-mono text-[0.58rem] text-slate-600">
                      {d.label_info?.n_classes ? `${d.label_info.n_classes} cls` : d.label_info?.kind || ''}
                      {d.n_bands ? ` - ${d.n_bands}b` : ''}
                      {d.split_sizes?.train ? ` - ${d.split_sizes.train} train` : ''}
                    </div>
                    <div className="font-mono text-[0.55rem] text-amber-500/70">
                      test: {String(d.split_sizes?.test ?? 'locked')}
                    </div>
                  </div>
                ))
              ) : (
                <Empty>no local datasets found</Empty>
              )}
            </div>
          </Panel>
        </div>

        <footer className="pb-3 pt-0 text-center font-mono text-[0.6rem] uppercase tracking-[0.25em] text-slate-700">
          GoTrain NEXUS - validation-only telemetry - test split locked
        </footer>
      </main>

      <AnimatePresence>
        {toast && (
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 24 }}
            className={cn(
              'fixed bottom-6 left-1/2 z-50 -translate-x-1/2 rounded-lg border px-4 py-2.5 text-sm backdrop-blur',
              toast.bad
                ? 'border-rose-400/40 bg-rose-950/80 text-rose-200'
                : 'border-emerald-400/40 bg-emerald-950/80 text-emerald-100',
            )}
          >
            <span className="flex items-center gap-2">
              {toast.bad && <AlertTriangle className="h-4 w-4" />}
              {toast.msg}
            </span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/** Shown in the gap between clicking Start and the new subprocess's first
 *  real action -- without this, this panel would otherwise show whichever
 *  comes first: leftover numbers from the previous session's last run, or
 *  a premature "no mission running" (IdleNotice) even though agent_running
 *  just flipped true. Several seconds genuinely pass here (import Strands,
 *  build the Bedrock model, first tool call) -- this is not an error state. */
function LaunchingNotice({ dataset }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-4 py-10 text-center">
      <Satellite className="h-6 w-6 animate-pulse text-sky-400" />
      <div className="font-mono text-xs uppercase tracking-[0.15em] text-sky-300">
        Launching on {dataset || 'dataset'}…
      </div>
      <div className="font-mono text-[0.62rem] text-slate-600">
        starting the agent process — this takes a few seconds, not an error
      </div>
    </div>
  )
}

/** Shown in place of a blank/dashed stats grid when no run is active --
 *  distinguishes "the agent is up and will start a run shortly" from
 *  "nothing is running at all", instead of looking like broken telemetry. */
function IdleNotice({ agentRunning }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-4 py-10 text-center">
      <Satellite className={cn('h-6 w-6 text-slate-600', agentRunning && 'animate-pulse text-sky-400')} />
      <div className="font-mono text-xs uppercase tracking-[0.15em] text-slate-500">
        {agentRunning ? 'Waiting for next experiment…' : 'No mission running'}
      </div>
      <div className="font-mono text-[0.62rem] text-slate-600">
        {agentRunning
          ? 'the agent is between runs -- planning, evaluating, or deciding next steps'
          : 'use Launch below to start a Quick Test or a full Mission'}
      </div>
    </div>
  )
}

/** Segmented toggle in the Current Run header: flip the panel body
 *  between the per-epoch stat grid and both live curves, so the graphs
 *  are one click away instead of requiring a scroll past the fold. */
function ViewSwitch({ value, onChange }) {
  const opts = [
    { key: 'stats', label: 'Stats', Icon: LayoutGrid },
    { key: 'curves', label: 'Curves', Icon: LineChartIcon },
  ]
  return (
    <div className="flex overflow-hidden rounded-md border border-edge">
      {opts.map(({ key, label, Icon }) => (
        <button
          key={key}
          onClick={() => onChange(key)}
          title={`Show ${label.toLowerCase()}`}
          className={cn(
            'flex items-center gap-1 px-2 py-1 font-mono text-[0.6rem] uppercase tracking-wider transition-colors',
            value === key
              ? 'bg-emerald-500/15 text-emerald-200'
              : 'text-slate-500 hover:bg-slate-800/60 hover:text-slate-300',
          )}
        >
          <Icon className="h-3 w-3" /> {label}
        </button>
      ))}
    </div>
  )
}

/** Compact chart for the Current Run panel's "Curves" view. Same data as
 *  the full-width ChartPanels lower down, sized to sit two-up inside one
 *  panel body. */
function MiniChart({ title, data, dataKey, color, domain }) {
  const has = data.some((d) => typeof d[dataKey] === 'number')
  return (
    <div className="bg-panel px-3 py-3">
      <div className="mb-1 font-mono text-[0.6rem] uppercase tracking-[0.18em]" style={{ color }}>
        {title}
      </div>
      <div className="h-40">
        {has ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 6, right: 10, bottom: 0, left: -22 }}>
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
                dot={{ r: 2, fill: color }} activeDot={{ r: 3.5 }} isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <Empty>awaiting per-epoch telemetry</Empty>
        )}
      </div>
    </div>
  )
}

function ChartPanel({ title, data, dataKey, color, icon, domain }) {
  const has = data.some((d) => typeof d[dataKey] === 'number')
  return (
    <Panel>
      <PanelTitle right={<span style={{ color }}>{icon}</span>}>{title}</PanelTitle>
      <div className="h-64 px-2 py-3">
        {has ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 6, right: 14, bottom: 4, left: -14 }}>
              <CartesianGrid stroke="#1c2c44" strokeDasharray="3 3" />
              <XAxis dataKey="epoch" stroke="#475569" fontSize={10} tickLine={false} />
              <YAxis stroke="#475569" fontSize={10} tickLine={false} domain={domain || ['auto', 'auto']} />
              <Tooltip
                contentStyle={{
                  background: '#0d1829', border: '1px solid #1c2c44',
                  borderRadius: 8, fontSize: 11, fontFamily: 'monospace',
                }}
              />
              <Line
                type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2}
                dot={{ r: 2.5, fill: color }} activeDot={{ r: 4 }} isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <Empty>awaiting per-epoch telemetry</Empty>
        )}
      </div>
    </Panel>
  )
}

const Empty = ({ children }) => (
  <div className="grid h-full place-items-center px-4 py-8 text-center font-mono text-[0.62rem] uppercase tracking-wider text-slate-600">
    {children}
  </div>
)
