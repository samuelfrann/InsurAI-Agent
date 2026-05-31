import { useState, useEffect } from 'react'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import Sidebar from './Sidebar'
import { authFetch } from '../lib/api'

export default function Layout() {
  const [sessions, setSessions] = useState([])
  const [currentThreadId, setCurrentThreadId] = useState(null)
  const navigate = useNavigate()
  const location = useLocation()

  // Global theme state (used across all pages under Layout)
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem('theme') === 'dark')

  // 1. THE SLIDING STATE
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode)
    localStorage.setItem('theme', darkMode ? 'dark' : 'light')
  }, [darkMode])


  const username = localStorage.getItem('insurai_user') || 'Admin'

  const fetchSessions = () => {
    authFetch('/sessions')
      .then(r => {
        if (!r.ok) {
          if (r.status === 401) navigate('/login')
          return []
        }
        return r.json()
      })
      .then(data => setSessions(Array.isArray(data) ? data : []))
      .catch(() => setSessions([]))
  }

  useEffect(() => {
    fetchSessions()
  }, [])

  return (
    <div className="flex h-screen w-full bg-[#0a1628] overflow-hidden relative">


      {/* SLIDING SIDEBAR CONTAINER */}
      <div
        style={{
          width: isSidebarOpen ? 230 : 0,
          transition: 'width 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          overflow: 'hidden',
          flexShrink: 0,
          height: '100%',
          background: 'var(--navy)'
        }}
      >
        <div style={{ width: 230, height: '100%', position: 'relative' }}>
          {/* Collapse button — inside sidebar, always accessible */}
          <button
            onClick={() => setIsSidebarOpen(false)}
            title="Collapse sidebar"
            style={{
              position: 'absolute', top: '1.1rem', right: '0.75rem', zIndex: 50,
              background: 'transparent', border: 'none',
              color: 'rgba(255,255,255,0.25)', cursor: 'pointer', padding: 0, lineHeight: 0,
            }}
            onMouseEnter={e => e.currentTarget.style.color = 'rgba(255,255,255,0.8)'}
            onMouseLeave={e => e.currentTarget.style.color = 'rgba(255,255,255,0.25)'}
          >
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <line x1="9" y1="3" x2="9" y2="21" />
            </svg>
          </button>
          <Sidebar
            sessions={sessions}
            currentThreadId={currentThreadId}
            username={username}
            onNewChat={() => {
              setCurrentThreadId(null)
              navigate('/chat')
            }}
            onLoadSession={(threadId, title, sessionType) => {
              setCurrentThreadId(threadId)
              if (sessionType === 'fraud') {
                navigate(`/fraud/${threadId}`)
              }
            }}
            onSessionsChange={fetchSessions}
          />
        </div>
      </div>
      
      <main className="flex-1 min-w-0 h-full overflow-y-auto" style={{ position: 'relative', background: 'var(--cream)' }}>
        {/* Open button — only shown when sidebar is fully collapsed */}
        {!isSidebarOpen && (
          <button
            onClick={() => setIsSidebarOpen(true)}
            title="Open sidebar"
            style={{
              position: 'absolute', top: '1rem', left: '1rem', zIndex: 50,
              background: 'transparent', border: 'none',
              color: 'var(--text-mid)', cursor: 'pointer', padding: 0, lineHeight: 0,
            }}
            onMouseEnter={e => e.currentTarget.style.color = 'var(--text-dark)'}
            onMouseLeave={e => e.currentTarget.style.color = 'var(--text-mid)'}
          >
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <line x1="9" y1="3" x2="9" y2="21" />
            </svg>
          </button>
        )}

        {/* Theme toggle — only on Fraud pages (Chat has its own; Dashboard/Analytics are dark-only) */}
        {location.pathname.startsWith('/fraud') && (
          <button
            onClick={() => setDarkMode(v => !v)}
            title="Toggle theme"
            className="w-8 h-8 flex items-center justify-center rounded-lg transition-colors"
            style={{
              position: 'absolute',
              top: '1rem',
              right: '1.5rem',
              zIndex: 40,
              border: '0.5px solid var(--cream-dark)',
              background: 'var(--cream)',
              color: 'var(--text-mid)',
            }}
          >
            {darkMode ? (
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                <circle cx="12" cy="12" r="5" />
                <line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" />
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                <line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" />
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
              </svg>
            )}
          </button>
        )}

        <Outlet context={{ currentThreadId, setCurrentThreadId, fetchSessions, darkMode, setDarkMode }} />
      </main>
    </div>
  )
}