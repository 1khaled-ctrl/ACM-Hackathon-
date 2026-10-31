import { motion } from 'framer-motion'
import logoUrl from '../assets/gotrain-logo.png'

/**
 * Brand lockup. Uses the real GoTrain logo asset (globe-G + satellite,
 * "Autonomous Geospatial Model Training").
 *
 * The PNG already has a dark navy background that matches the app's
 * --void, so it sits flush in the header with no cutout needed.
 */
export function Wordmark({ height = 58 }) {
  return (
    <div className="flex items-center gap-3">
      <motion.img
        src={logoUrl}
        alt="GoTrain - Autonomous Geospatial Model Training"
        style={{ height }}
        className="w-auto select-none rounded-md"
        initial={{ opacity: 0, x: -8 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        draggable={false}
      />
      <div className="hidden border-l border-edge/80 pl-3 leading-none sm:block">
        <div className="font-display text-[0.95rem] font-bold tracking-[0.3em] text-emerald-300">
          NEXUS
        </div>
        <div className="mt-1 font-mono text-[0.55rem] uppercase tracking-[0.25em] text-slate-500">
          mission control
        </div>
      </div>
    </div>
  )
}

/**
 * Compact animated mark for favicon-ish / small spots: an orbiting
 * satellite ring, echoing the logo's orbit without shrinking the full
 * lockup into illegibility.
 */
export function OrbitMark({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none" aria-hidden="true">
      <defs>
        <linearGradient id="gt-g" x1="0" y1="0" x2="64" y2="64">
          <stop offset="0%" stopColor="#22d3ee" />
          <stop offset="100%" stopColor="#22c55e" />
        </linearGradient>
      </defs>
      <circle cx="32" cy="32" r="15" stroke="url(#gt-g)" strokeWidth="2" opacity="0.9" />
      <motion.g
        style={{ originX: '32px', originY: '32px' }}
        animate={{ rotate: 360 }}
        transition={{ duration: 12, repeat: Infinity, ease: 'linear' }}
      >
        <ellipse
          cx="32" cy="32" rx="27" ry="10.5"
          stroke="url(#gt-g)" strokeWidth="1.6" opacity="0.8"
          transform="rotate(-22 32 32)"
        />
        <circle cx="59" cy="26" r="2.4" fill="#4ade80" />
      </motion.g>
    </svg>
  )
}
