/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontSize: {
        'sm': '0.875rem',
        'md': '1rem',
        'lg': '1.125rem',
      },
      textColor: {
        skin: {
          base: 'var(--color-text-base)',
          muted: 'var(--color-text-muted)',
          inverted: 'var(--color-text-inverted)',
          accent: 'var(--color-accent)',
          error: 'var(--color-error)',
        },
      },
      backgroundColor: {
        skin: {
          fill: 'var(--color-fill)',
          'fill-hover': 'var(--color-fill-hover)',
          'button-accent': 'var(--color-button-accent)',
          'button-accent-hover': 'var(--color-button-accent-hover)',
          'button-muted': 'var(--color-button-muted)',
        },
      },
      borderColor: {
        skin: {
          border: 'var(--color-border)',
          accent: 'var(--color-accent)',
        },
      },
      ringColor: {
        skin: {
          accent: 'var(--color-accent)',
        },
      },
    },
  },
  plugins: [],
}