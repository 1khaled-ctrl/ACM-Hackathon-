import { motion } from 'framer-motion'
import { Panel, PanelTitle, cn } from './ui'

/**
 * Animated agent pipeline. Which stage is "active" is derived from real
 * state (control state + whether a run is live + what the ledger holds),
 * never from a timer -- so the glow genuinely tracks the agent.
 */
const STAGES = [
  { key: 'inspect', label: 'Inspect', hint: 'GeoGuard / dataset scan' },
  { key: 'plan', label: 'Plan', hint: 'hypothesis + config' },
  { key: 'train', label: 'Train', hint: 'TRAIN split' },
  { key: 'validate', label: 'Validate', hint: 'VALIDATION split' },
  { key: 'decide', label: 'Decide', hint: 'ledger + compare' },
  { key: 'submit', label: 'Submit', hint: 'final model' },
]

export function Pipeline({ snapshot }) {
  const active = deriveStage(snapshot)
  const activeIdx = STAGES.findIndex((s) => s.key === active)

  return (
    <Panel>
      <PanelTitle>Agent Pipeline</PanelTitle>
      <div className="flex items-stretch gap-1 overflow-x-auto px-4 py-5">
        {STAGES.map((stage, i) => {
          const done = activeIdx > i
          const isActive = activeIdx === i
          return (
            <div key={stage.key} className="flex min-w-0 flex-1 items-center gap-1">
              <div className="min-w-0 flex-1">
                <motion.div
                  className={cn(
                    'relative overflow-hidden rounded-lg border px-3 py-2.5 text-center',
                    isActive
                      ? 'border-emerald-400/60 bg-emerald-500/10'
                      : done
                        ? 'border-sky-400/30 bg-sky-500/[0.07]'
                        : 'border-edge/70 bg-slate-900/40',
                  )}
                  animate={isActive ? { boxShadow: ['0 0 0px rgba(34,197,94,0)', '0 0 22px rgba(34,197,94,0.45)', '0 0 0px rgba(34,197,94,0)'] } : {}}
                  transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
                >
                  {isActive && (
                    <motion.div
                      className="absolute inset-0 -z-10"
                      style={{
                        background:
                          'linear-gradient(100deg, transparent 20%, rgba(34,197,94,0.18) 50%, transparent 80%)',
                        backgroundSize: '200% 100%',
                      }}
                      animate={{ backgroundPosition: ['-200% 0', '200% 0'] }}
                      transition={{ duration: 2.4, repeat: Infinity, ease: 'linear' }}
                    />
                  )}
                  <div
                    className={cn(
                      'font-display text-[0.8rem] font-semibold',
                      isActive ? 'text-emerald-200' : done ? 'text-sky-300' : 'text-slate-500',
                    )}
                  >
                    {stage.label}
                  </div>
                  <div className="mt-0.5 truncate font-mono text-[0.55rem] uppercase tracking-wider text-slate-600">
                    {stage.hint}
                  </div>
                </motion.div>
              </div>
              {i < STAGES.length - 1 && (
                <div className="relative h-px w-5 shrink-0 bg-edge">
                  {isActive && (
                    <motion.div
                      className="absolute inset-y-0 left-0 w-2 bg-emerald-300"
                      animate={{ x: [0, 12, 0], opacity: [0, 1, 0] }}
                      transition={{ duration: 1.4, repeat: Infinity, ease: 'easeInOut' }}
                    />
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </Panel>
  )
}

function deriveStage(snapshot) {
  if (!snapshot) return null
  const { control_state, current_run, status, history } = snapshot
  if (control_state === 'COMPLETED') return 'submit'
  if (current_run && current_run.run_id) {
    // Mid-run: epoch 0 hasn't validated yet, otherwise it validates every epoch.
    return current_run.epoch ? 'validate' : 'train'
  }
  const stage = (status?.stage || '').toLowerCase()
  if (stage.includes('explor') || stage.includes('inspect')) return 'inspect'
  if (stage.includes('eval')) return 'validate'
  if (stage.includes('train')) return 'train'
  if (history && history.length) return 'decide'
  return 'inspect'
}
