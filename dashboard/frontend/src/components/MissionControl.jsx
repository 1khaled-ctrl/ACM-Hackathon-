import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  AlertTriangle, Check, Hand, Pause, Play, RotateCcw, Square, Undo2, X,
} from 'lucide-react'
import { Badge, Button, Panel, PanelTitle, cn } from './ui'

/**
 * Mission Control: the engineer-facing surface over gotrain.control.
 *
 * Design intent (per the UX fix requested):
 *  - Pause = temporary freeze. Resume continues the SAME mission/state.
 *  - Manual Control = engineer takes the wheel after the agent reaches its
 *    next safe boundary (between tool calls). While here, the engineer gets
 *    real actions (approve/reject pending proposals, roll back a model) --
 *    not just a lone "resume" button -- then returns to autopilot.
 *  - Stop = terminates the mission process entirely. Destructive, requires
 *    confirmation, visually distinct (danger-red, separate from the
 *    pause/manual pair) so it can never be mistaken for "pause".
 *  - PAUSE_REQUESTED / a transition in flight is shown as a pending state
 *    with elapsed time ("requested Ns ago"), not as a parallel steady
 *    state -- so it reads as "working on it", never "stuck".
 *
 * None of this invents new backend states: PAUSE_REQUESTED already *is*
 * the pending state in gotrain/control.py; this component just presents
 * it as one instead of rendering it like RUNNING_AUTONOMOUSLY's sibling.
 */

const PENDING_STATES = new Set([
  'PAUSE_REQUESTED', 'RESUME_REQUESTED', 'ROLLBACK_REQUESTED', 'ROLLBACK_VERIFYING',
])

const PENDING_COPY = {
  PAUSE_REQUESTED: 'Pause requested — finishing current safe step…',
  RESUME_REQUESTED: 'Resume requested — resynchronizing with the agent…',
  ROLLBACK_REQUESTED: 'Rollback requested — preparing verification…',
  ROLLBACK_VERIFYING: 'Verifying rollback target — checkpoint integrity + inference sanity check…',
}

function useElapsed(since) {
  const [, tick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => tick((n) => n + 1), 1000)
    return () => clearInterval(id)
  }, [])
  if (!since) return null
  const s = Math.max(0, Math.floor((Date.now() - new Date(since).getTime()) / 1000))
  return s
}

export function MissionControl({ snap, post, notify, datasets }) {
  const state = snap?.control_state
  const control = snap?.control || {}
  const agentRunning = snap?.agent_running
  const [busy, setBusy] = useState(false)
  const [confirmStop, setConfirmStop] = useState(false)
  const elapsed = useElapsed(control.updated_at)

  // Any click disables the whole deck until the snapshot reflects the new
  // state (the next WS frame), so a double-click can't race two transitions.
  const act = async (path, body, label) => {
    setBusy(true)
    const r = await post(path, body)
    if (r) notify(label)
    setBusy(false)
    return r
  }
  const transition = (action, label) => act('api/control', { action }, label)

  const isPending = PENDING_STATES.has(state)
  const disabled = busy || isPending

  return (
    <Panel glow>
      <PanelTitle right={<span className="font-mono text-[0.6rem] text-slate-600">{snap?.logs_dir}</span>}>
        Mission Control
      </PanelTitle>

      <AnimatePresence mode="wait">
        {isPending ? (
          <PendingBanner key="pending" state={state} elapsed={elapsed} />
        ) : (
          <motion.div key="idle" exit={{ opacity: 0 }} />
        )}
      </AnimatePresence>

      <div className="flex flex-wrap items-center gap-3 px-4 py-4">
        <StatusCore state={state} />

        <div className="mx-1 h-8 w-px bg-edge" />

        {state === 'RUNNING_AUTONOMOUSLY' && (
          <>
            <Button variant="warn" disabled={disabled} onClick={() => transition('pause', 'Pause requested')}>
              <Pause className="h-4 w-4" /> Pause
            </Button>
            <Button disabled={disabled} onClick={() => transition('manual', 'Taking manual control…')}>
              <Hand className="h-4 w-4" /> Manual Control
            </Button>
          </>
        )}

        {state === 'PAUSE_REQUESTED' && (
          <Button variant="ghost" disabled={busy} onClick={() => transition('cancel_pause', 'Pause cancelled')}>
            <X className="h-4 w-4" /> Cancel pause request
          </Button>
        )}

        {state === 'PAUSED_FOR_ENGINEER' && (
          <>
            <Button variant="primary" disabled={disabled} onClick={() => transition('resume', 'Resuming…')}>
              <Play className="h-4 w-4" /> Resume (continue mission)
            </Button>
            <Button disabled={disabled} onClick={() => transition('manual', 'Taking manual control…')}>
              <Hand className="h-4 w-4" /> Manual Control
            </Button>
          </>
        )}

        {state === 'MANUAL_CONTROL' && (
          <Button variant="primary" disabled={disabled} onClick={() => transition('resume', 'Returning to autopilot…')}>
            <RotateCcw className="h-4 w-4" /> Return to Autopilot
          </Button>
        )}

        {state === 'FAILED_SAFE' && (
          <Button variant="danger" disabled={disabled} onClick={() => transition('acknowledge_failure', 'Failure acknowledged')}>
            <AlertTriangle className="h-4 w-4" /> Acknowledge failure → Pause
          </Button>
        )}

        {state === 'COMPLETED' && (
          <Badge tone="green">mission complete — see "Start New Mission" below</Badge>
        )}

        {/* Stop is visually and functionally separate: a hard subprocess
            kill, not a control-plane transition, so it is never grouped
            with Pause/Manual and always asks for confirmation. */}
        {agentRunning && state !== 'COMPLETED' && (
          <div className="ml-auto">
            {confirmStop ? (
              <ConfirmStop
                busy={busy}
                onCancel={() => setConfirmStop(false)}
                onConfirm={async () => {
                  setConfirmStop(false)
                  setBusy(true)
                  const r = await post('api/agent/stop')
                  if (r) notify(`Mission terminated (pid ${r.stopped_pid})`)
                  setBusy(false)
                }}
              />
            ) : (
              <Button variant="danger" disabled={busy} onClick={() => setConfirmStop(true)}>
                <Square className="h-4 w-4" /> Stop Mission
              </Button>
            )}
          </div>
        )}
      </div>

      {state === 'MANUAL_CONTROL' && <ManualControlPanel post={post} notify={notify} datasets={datasets} />}
    </Panel>
  )
}

function StatusCore({ state }) {
  const label = (state || 'unknown').replaceAll('_', ' ')
  const dot = {
    RUNNING_AUTONOMOUSLY: 'bg-emerald-400',
    PAUSED_FOR_ENGINEER: 'bg-sky-400',
    MANUAL_CONTROL: 'bg-amber-400',
    FAILED_SAFE: 'bg-rose-400',
    COMPLETED: 'bg-emerald-400',
  }[state] || 'bg-slate-500'
  return (
    <div className="flex items-center gap-2">
      <span className={cn('h-2 w-2 rounded-full', dot, PENDING_STATES.has(state) && 'animate-pulse')} />
      <span className="font-mono text-xs uppercase tracking-wider text-slate-300">{label}</span>
    </div>
  )
}

function PendingBanner({ state, elapsed }) {
  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      className="overflow-hidden border-b border-amber-400/20 bg-amber-500/[0.06] px-4 py-2.5"
    >
      <div className="flex items-center gap-2.5">
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400 opacity-60" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-amber-400" />
        </span>
        <span className="font-mono text-[0.72rem] text-amber-200">
          {PENDING_COPY[state] || `${state} …`}
        </span>
        {elapsed != null && (
          <span className="font-mono text-[0.62rem] text-amber-400/70">requested {elapsed}s ago</span>
        )}
      </div>
    </motion.div>
  )
}

function ConfirmStop({ busy, onCancel, onConfirm }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex items-center gap-2 rounded-lg border border-rose-400/40 bg-rose-950/40 px-3 py-1.5"
    >
      <span className="font-mono text-[0.68rem] text-rose-200">Terminate the mission?</span>
      <Button variant="danger" className="px-2.5 py-1 text-xs" disabled={busy} onClick={onConfirm}>
        <Check className="h-3.5 w-3.5" /> Confirm
      </Button>
      <Button variant="ghost" className="px-2.5 py-1 text-xs" disabled={busy} onClick={onCancel}>
        Cancel
      </Button>
    </motion.div>
  )
}

/** Real engineer actions while in MANUAL_CONTROL: review pending approval
 *  proposals and roll back to a prior model version. Both call straight
 *  into gotrain.approval / gotrain.rollback (same functions the legacy
 *  Streamlit dashboard used) via thin FastAPI passthroughs. */
function ManualControlPanel({ post, notify, datasets }) {
  const [proposals, setProposals] = useState(null)
  const [rollbackDataset, setRollbackDataset] = useState('')
  const [rollbackInfo, setRollbackInfo] = useState(null)
  const [busyKey, setBusyKey] = useState(null)

  const refreshProposals = () => {
    fetch(new URL('api/approvals?status=pending', window.location.href))
      .then((r) => r.json())
      .then((d) => setProposals(d.proposals || []))
      .catch(() => setProposals([]))
  }

  useEffect(() => { refreshProposals() }, [])

  const decide = async (key, approve) => {
    setBusyKey(key)
    const r = await post('api/approvals/decide', { key, approve })
    if (r) {
      notify(approve ? 'Proposal approved' : 'Proposal rejected')
      refreshProposals()
    }
    setBusyKey(null)
  }

  const loadRollbackVersions = async (dataset) => {
    setRollbackDataset(dataset)
    setRollbackInfo(null)
    if (!dataset) return
    try {
      const r = await fetch(new URL(`api/rollback/versions?dataset=${encodeURIComponent(dataset)}`, window.location.href))
      setRollbackInfo(await r.json())
    } catch {
      setRollbackInfo({ versions: [] })
    }
  }

  const doRollback = async (version) => {
    // /api/datasets exposes "benchmark" (e.g. "classification_v1.0"), not a
    // bare task_type -- derive it rather than guessing a default, so a
    // segmentation rollback can never be silently sent as "classification".
    const benchmark = datasets?.find((d) => d.dataset === rollbackDataset)?.benchmark || ''
    const taskType = benchmark.startsWith('classification') ? 'classification'
      : benchmark.startsWith('segmentation') ? 'segmentation' : null
    if (!taskType) {
      notify(`Unknown task type for ${rollbackDataset} -- pick a dataset from the known list`, true)
      return
    }
    setBusyKey(`rb-${version}`)
    const r = await post('api/rollback/execute', { dataset: rollbackDataset, version, task_type: taskType })
    if (r) {
      notify(`Rolled back ${rollbackDataset} to ${version}`)
      loadRollbackVersions(rollbackDataset)
    }
    setBusyKey(null)
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="grid grid-cols-1 gap-4 border-t border-edge/60 px-4 py-4 lg:grid-cols-2"
    >
      <div>
        <div className="mb-2 font-mono text-[0.62rem] uppercase tracking-[0.18em] text-amber-300/80">
          Pending Approvals {proposals ? `(${proposals.length})` : ''}
        </div>
        {proposals === null ? (
          <div className="font-mono text-xs text-slate-600">loading…</div>
        ) : proposals.length === 0 ? (
          <div className="font-mono text-xs text-slate-600">nothing pending review</div>
        ) : (
          <div className="space-y-2">
            {proposals.map((p) => (
              <div key={p.key} className="rounded-lg border border-edge bg-slate-900/50 px-3 py-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-xs text-slate-200">
                    {p.dataset} · {p.field}=<span className="text-sky-300">{String(p.value)}</span>
                  </span>
                  <div className="flex gap-1.5">
                    <Button
                      className="px-2 py-1 text-xs" variant="primary"
                      disabled={busyKey === p.key}
                      onClick={() => decide(p.key, true)}
                    >
                      <Check className="h-3 w-3" /> Approve
                    </Button>
                    <Button
                      className="px-2 py-1 text-xs" variant="danger"
                      disabled={busyKey === p.key}
                      onClick={() => decide(p.key, false)}
                    >
                      <X className="h-3 w-3" /> Reject
                    </Button>
                  </div>
                </div>
                <div className="mt-1 font-mono text-[0.6rem] text-slate-500">{p.reason}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div>
        <div className="mb-2 font-mono text-[0.62rem] uppercase tracking-[0.18em] text-amber-300/80">
          Rollback Model
        </div>
        <input
          placeholder="dataset name, e.g. m-eurosat"
          value={rollbackDataset}
          onChange={(e) => loadRollbackVersions(e.target.value)}
          className="w-full rounded-md border border-edge bg-slate-900/70 px-2.5 py-1.5 font-mono text-xs text-slate-200 placeholder:text-slate-600"
        />
        {rollbackInfo && (
          <div className="mt-2 max-h-32 space-y-1 overflow-y-auto">
            {rollbackInfo.versions?.length ? (
              rollbackInfo.versions.slice().reverse().map((v) => (
                <div key={v} className="flex items-center justify-between rounded-md border border-edge/60 bg-slate-900/40 px-2.5 py-1.5">
                  <span className="font-mono text-xs text-slate-300">
                    {v} {v === rollbackInfo.active_version && <Badge tone="green">active</Badge>}
                  </span>
                  <Button
                    className="px-2 py-1 text-xs"
                    disabled={v === rollbackInfo.active_version || busyKey === `rb-${v}`}
                    onClick={() => doRollback(v)}
                  >
                    <Undo2 className="h-3 w-3" /> Roll back
                  </Button>
                </div>
              ))
            ) : (
              <div className="font-mono text-xs text-slate-600">no registered versions for this dataset</div>
            )}
          </div>
        )}
      </div>
    </motion.div>
  )
}
