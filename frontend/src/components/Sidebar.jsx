import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { authFetch, logout } from '../lib/api'

export default function Sidebar({
  sessions = [],
  currentThreadId = null,
  username = '',
  onNewChat = () => {},
  onLoadSession = () => {},
  onSessionsChange = () => {}
}) {
  const navigate = useNavigate()
  const location = useLocation()
  const [renamingId, setRenamingId]         = useState(null)
  const [renameValue, setRenameValue]       = useState('')
  const [confirmDeleteId, setConfirmDeleteId] = useState(null)

  // Split the unified sessions into their correct categories
  const safeSessions = Array.isArray(sessions) ? sessions : []
  const chatSessions = safeSessions.filter(s => s.session_type === 'chat' || !s.session_type)
  const assessmentSessions = safeSessions.filter(s => s.session_type === 'fraud')

  function startRename(e, session) {
    e.stopPropagation()
    setRenamingId(session.thread_id)
    setRenameValue(session.title || 'Untitled')
    setConfirmDeleteId(null)
  }

  async function commitRename(threadId) {
    const title = renameValue.trim() || 'Untitled'
    setRenamingId(null)
    try {
      await authFetch(`/sessions/${threadId}`, {
        method: 'PATCH',
        body: JSON.stringify({ title }),
      })
      onSessionsChange()
    } catch {}
  }

  function startDelete(e, threadId) {
    e.stopPropagation()
    setConfirmDeleteId(threadId)
    setRenamingId(null)
  }

  async function confirmDelete(threadId) {
    try {
      await authFetch(`/sessions/${threadId}`, { method: 'DELETE' })
      if (currentThreadId === threadId) onNewChat()
      else onSessionsChange()
    } catch {}
    setConfirmDeleteId(null)
  }

  const navItems = [
    {
      label: 'Chat',
      path: '/chat',
      icon: (
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
        </svg>
      ),
    },
    {
      label: 'Fraud Tool',
      path: '/fraud',
      icon: (
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          <line x1="12" y1="9" x2="12" y2="13" />
          <line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
      ),
    },
    {
      label: 'Dashboard',
      path: '/dashboard',
      icon: (
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
          <rect x="14" y="14" width="7" height="7" /><rect x="3" y="14" width="7" height="7" />
        </svg>
      ),
    },
    {
      label: 'Analytics',
      path: '/analytics',
      icon: (
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" /><line x1="6" y1="20" x2="6" y2="14" />
        </svg>
      ),
    },
  ]

  const initials = (username || 'U').slice(0, 2).toUpperCase()

  // Reusable component to render a session row with rename/delete
  const SessionRow = ({ session, isAssessment }) => (
    <div
      key={session.thread_id}
      onClick={() => {
        onLoadSession(session.thread_id, session.title, session.session_type)
        if (!isAssessment) navigate('/chat')
      }}
      className="group flex items-center gap-2 px-2.5 py-1.5 rounded-lg cursor-pointer transition-colors"
      style={session.thread_id === currentThreadId ? { background: 'rgba(255,255,255,0.1)' } : {}}
    >
      <div
        className="w-1.5 h-1.5 rounded-full flex-shrink-0"
        style={{ background: session.thread_id === currentThreadId ? 'var(--gold)' : 'rgba(255,255,255,0.15)' }}
      />
      {renamingId === session.thread_id ? (
        <input
          autoFocus
          value={renameValue}
          onChange={e => setRenameValue(e.target.value)}
          onBlur={() => commitRename(session.thread_id)}
          onKeyDown={e => {
            if (e.key === 'Enter') commitRename(session.thread_id)
            if (e.key === 'Escape') setRenamingId(null)
            e.stopPropagation()
          }}
          onClick={e => e.stopPropagation()}
          className="flex-1 min-w-0 text-xs rounded px-1 outline-none"
          style={{ background: 'rgba(255,255,255,0.08)', border: '0.5px solid var(--gold-border)', color: '#fff', fontFamily: 'DM Sans, sans-serif' }}
        />
      ) : confirmDeleteId === session.thread_id ? (
        <>
          <span className="flex-1 text-xs" style={{ color: 'var(--danger)', whiteSpace: 'nowrap' }}>Delete?</span>
          <button onClick={e => { e.stopPropagation(); confirmDelete(session.thread_id) }} className="w-4 h-4 flex items-center justify-center rounded text-xs" style={{ background: 'rgba(229,62,62,0.2)', color: 'var(--danger)' }}>✓</button>
          <button onClick={e => { e.stopPropagation(); setConfirmDeleteId(null) }} className="w-4 h-4 flex items-center justify-center rounded text-xs" style={{ background: 'rgba(255,255,255,0.06)', color: 'rgba(255,255,255,0.4)' }}>✕</button>
        </>
      ) : (
        <>
          <span className="flex-1 text-xs truncate" style={{ color: session.thread_id === currentThreadId ? 'rgba(255,255,255,0.85)' : 'rgba(255,255,255,0.45)' }}>
            {session.title || 'Untitled'}
          </span>
          <button onClick={e => startRename(e, session)} className="hidden group-hover:flex w-4 h-4 items-center justify-center rounded" style={{ color: 'rgba(255,255,255,0.15)', background: 'transparent', border: 'none' }} title="Rename">
            <svg viewBox="0 0 24 24" width="10" height="10" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" />
              <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" />
            </svg>
          </button>
          <button onClick={e => startDelete(e, session.thread_id)} className="hidden group-hover:flex w-4 h-4 items-center justify-center rounded text-xs" style={{ color: 'rgba(255,255,255,0.15)', background: 'transparent', border: 'none' }} title="Delete">🗑️</button>
        </>
      )}
    </div>
  )

  return (
    <div className="flex flex-col h-full overflow-hidden" style={{ width: 230, background: 'var(--navy)' }}>
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-3.5 py-4 flex-shrink-0" style={{ borderBottom: '0.5px solid rgba(255,255,255,0.06)' }}>
        <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0" style={{ background: 'var(--navy-mid)', border: '0.5px solid var(--gold-border)' }}>
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none">
            <path d="M12 2L4 5.5V11c0 5 3.5 9.5 8 11 4.5-1.5 8-6 8-11V5.5L12 2z" fill="#c9a84c"/>
            <text x="12" y="15.5" textAnchor="middle" fontSize="7" fontWeight="700" fontFamily="sans-serif" fill="#0a1628">AI</text>
          </svg>
        </div>
        <div>
          <p className="text-sm font-serif text-white">InsurAI</p>
          <p className="text-xs" style={{ color: 'rgba(255,255,255,0.3)' }}>Copilot</p>
        </div>
      </div>

      {/* New chat */}
      <button onClick={onNewChat} className="flex items-center gap-2 mx-2.5 mt-2.5 mb-1 px-3 py-2 rounded-lg text-xs font-medium transition-colors" style={{ background: 'rgba(255,255,255,0.06)', border: '0.5px solid rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.7)' }}>
        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="var(--gold)" strokeWidth="2.2" strokeLinecap="round">
          <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
        </svg>
        New chat
      </button>

      {/* Nav */}
      <nav className="flex flex-col gap-px px-2 py-1 flex-shrink-0">
        <p className="text-xs font-medium uppercase tracking-widest px-2 pt-2 pb-1" style={{ color: 'rgba(255,255,255,0.2)' }}>Main</p>
        {navItems.map(item => (
          <button key={item.path} onClick={() => navigate(item.path)} className="flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-xs transition-colors text-left" style={location.pathname === item.path ? { background: 'rgba(255,255,255,0.1)', color: '#fff' } : { color: 'rgba(255,255,255,0.45)' }}>
            <span style={location.pathname === item.path ? { color: 'var(--gold)' } : {}}>{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>

      {/* Divider */}
      <div className="mx-2.5 my-1.5" style={{ height: '0.5px', background: 'rgba(255,255,255,0.06)' }} />

      {/* Recent chats */}
      <div className="flex flex-col flex-1 min-h-0 px-2">
        <p className="text-xs font-medium uppercase tracking-widest px-2 pt-1 pb-1 flex-shrink-0" style={{ color: 'rgba(255,255,255,0.2)' }}>Recent chats</p>
        <div className="flex-1 overflow-y-auto sessions-scroll flex flex-col gap-px pb-1 min-h-0">
          {chatSessions.length === 0 ? (
            <p className="text-xs px-2 py-1 italic" style={{ color: 'rgba(255,255,255,0.18)' }}>No chats yet</p>
          ) : chatSessions.map(session => <SessionRow key={session.thread_id} session={session} isAssessment={false} />)}
        </div>
      </div>

      {/* User footer */}
      <div className="flex items-center gap-2.5 px-3.5 py-3 flex-shrink-0" style={{ borderTop: '0.5px solid rgba(255,255,255,0.06)' }}>
        <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold flex-shrink-0" style={{ background: 'var(--gold-soft)', border: '0.5px solid var(--gold-border)', color: 'var(--gold)' }}>
          {initials}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-white truncate">{username || 'Staff'}</p>
          <p className="text-xs" style={{ color: 'rgba(255,255,255,0.3)' }}>Staff</p>
        </div>
        <button onClick={logout} className="w-7 h-7 flex items-center justify-center rounded-lg transition-colors flex-shrink-0" style={{ color: 'rgba(255,255,255,0.25)', background: 'transparent', border: 'none' }} title="Sign out">
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" />
            <polyline points="16 17 21 12 16 7" />
            <line x1="21" y1="12" x2="9" y2="12" />
          </svg>
        </button>
      </div>
    </div>
  )
}