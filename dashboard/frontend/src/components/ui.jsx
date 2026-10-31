import { motion } from 'framer-motion'
import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export const cn = (...a) => twMerge(clsx(a))

/** Panel with a subtle gradient edge -- the base surface for every card. */
export function Panel({ className, children, glow = false, ...rest }) {
  return (
    <div
      className={cn(
        'relative rounded-xl border border-edge/80 bg-panel/70 backdrop-blur-sm',
        'before:pointer-events-none before:absolute before:inset-0 before:rounded-xl',
        'before:bg-gradient-to-b before:from-white/[0.04] before:to-transparent',
        glow && 'shadow-glow',
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  )
}

export function PanelTitle({ children, right }) {
  return (
    <div className="flex items-center justify-between border-b border-edge/60 px-4 py-2.5">
      <h2 className="font-mono text-[0.68rem] uppercase tracking-[0.2em] text-slate-400">
        {children}
      </h2>
      {right}
    </div>
  )
}

const TONES = {
  sky: 'bg-sky-500/10 text-sky-300 border-sky-400/30',
  green: 'bg-emerald-500/10 text-emerald-300 border-emerald-400/30',
  amber: 'bg-amber-500/10 text-amber-300 border-amber-400/30',
  red: 'bg-rose-500/10 text-rose-300 border-rose-400/30',
  slate: 'bg-slate-500/10 text-slate-300 border-slate-400/25',
}

export function Badge({ tone = 'slate', children, pulse = false }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5',
        'font-mono text-[0.64rem] uppercase tracking-wider',
        TONES[tone] ?? TONES.slate,
      )}
    >
      {pulse && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />}
      {children}
    </span>
  )
}

export function Button({ variant = 'default', className, children, ...rest }) {
  const variants = {
    default: 'bg-slate-800/80 hover:bg-slate-700 text-slate-100 border-edge',
    primary:
      'bg-gradient-to-r from-emerald-500 to-sky-500 hover:from-emerald-400 hover:to-sky-400 text-slate-950 border-transparent font-semibold shadow-glow',
    danger: 'bg-rose-600/90 hover:bg-rose-500 text-white border-transparent',
    warn: 'bg-amber-500/90 hover:bg-amber-400 text-slate-950 border-transparent font-semibold',
    ghost: 'bg-transparent hover:bg-slate-800/60 text-slate-300 border-edge/60',
  }
  return (
    <motion.button
      whileHover={{ y: -1 }}
      whileTap={{ scale: 0.97 }}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-lg border px-3.5 py-2',
        'text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-40',
        variants[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </motion.button>
  )
}

/** Big headline number. */
export function Stat({ label, value, sub, tone = 'text-white' }) {
  return (
    <div className="px-4 py-3">
      <div className="font-mono text-[0.6rem] uppercase tracking-[0.2em] text-slate-500">
        {label}
      </div>
      <div className={cn('mt-1 font-display text-2xl font-bold tabular-nums', tone)}>{value}</div>
      {sub && <div className="mt-0.5 font-mono text-[0.65rem] text-slate-500">{sub}</div>}
    </div>
  )
}

/** Animated SVG progress ring. */
export function Ring({ value = 0, size = 112, stroke = 8, label, sub, tone = '#22c55e' }) {
  const pct = Math.max(0, Math.min(1, value || 0))
  const r = (size - stroke) / 2
  const circ = 2 * Math.PI * r
  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} stroke="#1c2c44" strokeWidth={stroke} fill="none" />
        <motion.circle
          cx={size / 2} cy={size / 2} r={r}
          stroke={tone} strokeWidth={stroke} fill="none" strokeLinecap="round"
          strokeDasharray={circ}
          initial={false}
          animate={{ strokeDashoffset: circ * (1 - pct) }}
          transition={{ type: 'spring', stiffness: 60, damping: 18 }}
          style={{ filter: `drop-shadow(0 0 6px ${tone}88)` }}
        />
      </svg>
      <div className="absolute grid place-items-center text-center">
        <div className="font-display text-xl font-bold tabular-nums text-white">{label}</div>
        {sub && <div className="font-mono text-[0.58rem] uppercase tracking-wider text-slate-500">{sub}</div>}
      </div>
    </div>
  )
}

/** Thin labelled meter, used for CPU/RAM telemetry. */
export function Meter({ label, percent, tone = 'from-emerald-400 to-sky-500' }) {
  const p = percent == null ? null : Math.max(0, Math.min(100, percent))
  return (
    <div className="px-4 py-2">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="font-mono text-[0.62rem] uppercase tracking-[0.18em] text-slate-500">{label}</span>
        <span className="font-mono text-xs tabular-nums text-slate-300">
          {p == null ? '--' : `${p.toFixed(0)}%`}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
        <motion.div
          className={cn('h-full rounded-full bg-gradient-to-r', tone)}
          initial={false}
          animate={{ width: `${p ?? 0}%` }}
          transition={{ type: 'spring', stiffness: 80, damping: 20 }}
        />
      </div>
    </div>
  )
}

/** Ambient background: drifting nebula blobs + a faint grid. */
export function Backdrop() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-void">
      <div
        className="absolute inset-0 opacity-[0.35]"
        style={{
          backgroundImage:
            'linear-gradient(rgba(34,197,94,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(34,197,94,0.05) 1px, transparent 1px)',
          backgroundSize: '46px 46px',
        }}
      />
      <motion.div
        className="absolute -top-40 -left-32 h-[34rem] w-[34rem] rounded-full blur-[120px]"
        style={{ background: 'radial-gradient(circle, rgba(34,197,94,0.15), transparent 70%)' }}
        animate={{ x: [0, 70, 0], y: [0, 40, 0] }}
        transition={{ duration: 26, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute -bottom-48 right-[-10rem] h-[38rem] w-[38rem] rounded-full blur-[130px]"
        style={{ background: 'radial-gradient(circle, rgba(56,189,248,0.15), transparent 70%)' }}
        animate={{ x: [0, -60, 0], y: [0, -30, 0] }}
        transition={{ duration: 32, repeat: Infinity, ease: 'easeInOut' }}
      />
    </div>
  )
}
