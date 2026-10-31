import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { FlaskConical, Play, Rocket, RotateCcw } from 'lucide-react'
import { Button, Panel, PanelTitle, cn } from './ui'

/**
 * Mission launcher. Two presets instead of one bare numeric form, so
 * "1 run / 1 minute" never looks like a real mission by default:
 *   - Quick Test: 1 run, 1 minute -- explicitly labelled as a smoke test.
 *   - Mission (default): 3 runs, 5 minutes/run -- a real attempt.
 * Both are still just the same /api/agent/start call with different
 * numbers; nothing about run_gotrain.py or the control plane changes.
 */
// run_minutes becomes a HARD subprocess kill in
// gotrain/tools/training.py (timeout_seconds) -- training that hasn't
// finished by then is killed and logs training_timed_out, producing no
// ledger entry at all. 1 min is below what a real GEO-Bench run needs,
// so Quick Test is labelled as a wiring check, not a cheap experiment.
const PRESETS = {
  quick: {
    label: 'Quick Test', icon: FlaskConical, maxRuns: 1, runMinutes: 1,
    hint: '1 run · 1 min — wiring check; training will likely time out',
  },
  mission: {
    label: 'Mission', icon: Rocket, maxRuns: 3, runMinutes: 5,
    hint: '3 runs · 5 min/run — a real attempt',
  },
}

export function Launcher({ datasets, snap, post, notify }) {
  const [preset, setPreset] = useState('mission')
  const [dataset, setDataset] = useState('')
  const [custom, setCustom] = useState(false)
  const [maxRuns, setMaxRuns] = useState(PRESETS.mission.maxRuns)
  const [runMinutes, setRunMinutes] = useState(PRESETS.mission.runMinutes)
  const [starting, setStarting] = useState(false)

  useEffect(() => {
    if (!dataset && datasets.length) setDataset(datasets[0].dataset)
  }, [datasets, dataset])

  useEffect(() => {
    if (!custom) {
      setMaxRuns(PRESETS[preset].maxRuns)
      setRunMinutes(PRESETS[preset].runMinutes)
    }
  }, [preset, custom])

  const agentRunning = snap?.agent_running
  const state = snap?.control_state
  const justCompleted = state === 'COMPLETED'

  const start = async () => {
    setStarting(true)
    const r = await post('api/agent/start', {
      dataset,
      max_runs: Number(maxRuns),
      run_minutes: Number(runMinutes),
    })
    if (r) notify(`Mission started on ${dataset} (${maxRuns} runs × ${runMinutes}min, pid ${r.pid})`)
    setStarting(false)
  }

  // COMPLETED wins over agent_running. The agent's OS process routinely
  // lingers for a few seconds after it submits a final model (boto3 /
  // Strands teardown), so agent_running is still true while the control
  // state is already COMPLETED. Checking agentRunning first meant that
  // window rendered "mission in progress" with NO start button at all --
  // the mission was over, but there was no way to begin a new one.
  // COMPLETED is terminal in gotrain/control.py, so it is the truth here.
  if (agentRunning && !justCompleted) {
    return (
      <div className="flex items-center gap-2 font-mono text-xs text-slate-500">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
        mission in progress — use Mission Control to pause, take manual control, or stop
      </div>
    )
  }

  const field = 'rounded-md border border-edge bg-slate-900/70 px-2 py-1.5 font-mono text-xs text-slate-200'

  return (
    <div className="flex flex-wrap items-center gap-2.5">
      {justCompleted && (
        <span className="mr-1 font-mono text-[0.62rem] uppercase tracking-wider text-emerald-400">
          ready to go again:
        </span>
      )}

      <div className="flex overflow-hidden rounded-lg border border-edge">
        {Object.entries(PRESETS).map(([key, p]) => {
          const Icon = p.icon
          const active = preset === key && !custom
          return (
            <button
              key={key}
              onClick={() => { setPreset(key); setCustom(false) }}
              title={p.hint}
              className={cn(
                'flex items-center gap-1.5 px-3 py-2 font-mono text-xs transition-colors',
                active ? 'bg-sky-500/15 text-sky-200' : 'text-slate-500 hover:bg-slate-800/60',
              )}
            >
              <Icon className="h-3.5 w-3.5" /> {p.label}
            </button>
          )
        })}
      </div>

      <select className={field} value={dataset} onChange={(e) => setDataset(e.target.value)}>
        {datasets.length === 0 && <option value="">no datasets</option>}
        {datasets.map((d) => (
          <option key={d.dataset} value={d.dataset}>{d.dataset}</option>
        ))}
      </select>

      {custom ? (
        <>
          <label className="flex items-center gap-1 font-mono text-[0.6rem] uppercase text-slate-500">
            runs
            <input type="number" min={1} max={50} value={maxRuns}
              onChange={(e) => setMaxRuns(e.target.value)} className={cn(field, 'w-14')} />
          </label>
          <label className="flex items-center gap-1 font-mono text-[0.6rem] uppercase text-slate-500">
            min/run
            <input type="number" min={1} max={600} value={runMinutes}
              onChange={(e) => setRunMinutes(e.target.value)} className={cn(field, 'w-16')} />
          </label>
        </>
      ) : (
        <button
          onClick={() => setCustom(true)}
          className="font-mono text-[0.6rem] uppercase tracking-wider text-slate-600 underline-offset-2 hover:text-slate-400 hover:underline"
        >
          {PRESETS[preset].hint} · customize
        </button>
      )}

      <Button variant="primary" onClick={start} disabled={!dataset || starting}>
        {justCompleted ? <RotateCcw className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        {justCompleted ? 'Start New Mission' : starting ? 'Starting…' : `Start ${custom ? '' : PRESETS[preset].label}`}
      </Button>
    </div>
  )
}
