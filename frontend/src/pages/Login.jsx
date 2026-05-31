import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getToken } from '../lib/api'

const BASE = import.meta.env.VITE_API_URL || ''

export default function Login() {
  const navigate = useNavigate()

  // Already logged in
  if (getToken()) {
    navigate('/chat', { replace: true })
    return null
  }

  const [username, setUsername]     = useState('')
  const [password, setPassword]     = useState('')
  const [showPass, setShowPass]     = useState(false)
  const [error, setError]           = useState('')
  const [loading, setLoading]       = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const res = await fetch(BASE + '/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim(), password }),
      })
      const data = await res.json()
      if (!res.ok) {
        setError(data.detail || 'Incorrect username or password.')
        return
      }
      localStorage.setItem('insurai_token', data.access_token)
      localStorage.setItem('insurai_user', data.username)
      navigate('/chat', { replace: true })
    } catch {
      setError('Server unreachable. Make sure the API is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      className="min-h-screen flex items-center justify-center relative overflow-hidden anim-fade-up"
      style={{ background: 'var(--cream)' }}
    >
      {/* decorative ring */}
      <div
        className="pointer-events-none fixed"
        style={{
          top: -100, right: -100,
          width: 400, height: 400,
          border: '1px solid rgba(201,168,76,0.12)',
          borderRadius: '50%',
        }}
      />

      <div className="relative z-10 w-full max-w-sm px-6">
        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <div
            className="w-12 h-12 rounded-xl flex items-center justify-center mb-3"
            style={{ background: 'var(--navy)', border: '0.5px solid var(--gold-border)' }}
          >
            <svg viewBox="0 0 24 24" width="26" height="26" fill="none">
              <path d="M12 2L4 5.5V11c0 5 3.5 9.5 8 11 4.5-1.5 8-6 8-11V5.5L12 2z" fill="#c9a84c"/>
              <text x="12" y="15.5" textAnchor="middle" fontSize="7" fontWeight="700" fontFamily="sans-serif" fill="#0a1628">AI</text>
            </svg>
          </div>
          <h1 className="font-serif text-2xl" style={{ color: 'var(--text-dark)' }}>InsurAI</h1>
          <p className="text-sm mt-1" style={{ color: 'var(--text-light)' }}>Staff Copilot · Sign in to continue</p>
        </div>

        {/* Card */}
        <div
          className="rounded-2xl p-8"
          style={{
            background: 'var(--white)',
            border: '0.5px solid var(--cream-dark)',
            boxShadow: '0 20px 60px rgba(10,22,40,0.08)',
          }}
        >
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            {/* Username */}
            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium" style={{ color: 'var(--text-mid)' }}>
                Username
              </label>
              <input
                type="text"
                value={username}
                onChange={e => setUsername(e.target.value)}
                required
                autoFocus
                className="w-full rounded-lg px-3 py-2.5 text-sm outline-none transition-all"
                style={{
                  background: 'var(--cream)',
                  border: '1.5px solid var(--cream-dark)',
                  color: 'var(--text-dark)',
                }}
                onFocus={e => { e.target.style.borderColor = 'var(--gold)'; e.target.style.boxShadow = '0 0 0 3px rgba(201,168,76,0.1)' }}
                onBlur={e => { e.target.style.borderColor = 'var(--cream-dark)'; e.target.style.boxShadow = 'none' }}
                placeholder="admin"
              />
            </div>

            {/* Password */}
            <div className="flex flex-col gap-1">
              <label className="text-xs font-medium" style={{ color: 'var(--text-mid)' }}>
                Password
              </label>
              <div className="relative">
                <input
                  type={showPass ? 'text' : 'password'}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                  className="w-full rounded-lg px-3 py-2.5 text-sm outline-none transition-all pr-10"
                  style={{
                    background: 'var(--cream)',
                    border: '1.5px solid var(--cream-dark)',
                    color: 'var(--text-dark)',
                  }}
                  onFocus={e => { e.target.style.borderColor = 'var(--gold)'; e.target.style.boxShadow = '0 0 0 3px rgba(201,168,76,0.1)' }}
                  onBlur={e => { e.target.style.borderColor = 'var(--cream-dark)'; e.target.style.boxShadow = 'none' }}
                  placeholder="••••••••"
                />
                <button
                  type="button"
                  onClick={() => setShowPass(v => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-xs"
                  style={{ color: 'var(--text-light)' }}
                >
                  {showPass ? 'Hide' : 'Show'}
                </button>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div
                className="rounded-lg px-3 py-2 text-xs"
                style={{ background: 'var(--danger-bg)', color: 'var(--danger)' }}
              >
                {error}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-xl py-3 text-sm font-semibold transition-all mt-1"
              style={{
                background: loading ? 'rgba(10,22,40,0.5)' : 'var(--navy)',
                color: '#fff',
                cursor: loading ? 'not-allowed' : 'pointer',
              }}
            >
              {loading ? 'Signing in…' : 'Sign in'}
            </button>
          </form>
        </div>

        <p className="text-center mt-6 text-xs" style={{ color: 'var(--text-light)' }}>
          Staff access only · Contact your admin to create an account
        </p>
      </div>
    </div>
  )
}
