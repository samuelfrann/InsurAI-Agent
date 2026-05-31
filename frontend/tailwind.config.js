/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        navy: {
          DEFAULT: '#0a1628',
          mid: '#0f2040',
          light: '#1a3560',
        },
        gold: {
          DEFAULT: '#c9a84c',
          light: '#e8c870',
          soft: 'rgba(201,168,76,0.12)',
          border: 'rgba(201,168,76,0.3)',
        },
        cream: {
          DEFAULT: '#f8f4ed',
          dark: '#ede8df',
        },
      },
      fontFamily: {
        sans: ['DM Sans', 'sans-serif'],
        serif: ['DM Serif Display', 'serif'],
      },
    },
  },
  plugins: [],
}
