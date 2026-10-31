export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['Space Grotesk', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      colors: {
        // Sampled from the GoTrain logo: deep navy ground, emerald + sky accents.
        void: '#060c18',
        deep: '#0a1324',
        panel: '#0d1829',
        edge: '#1c2c44',
      },
      boxShadow: {
        glow: '0 0 24px -4px rgba(34,197,94,0.42)',
        'glow-sky': '0 0 28px -4px rgba(56,189,248,0.45)',
      },
      animation: {
        'spin-slow': 'spin 14s linear infinite',
        'pulse-slow': 'pulse 3.5s cubic-bezier(0.4,0,0.6,1) infinite',
        shimmer: 'shimmer 2.4s linear infinite',
      },
      keyframes: {
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
    },
  },
  plugins: [],
}
