import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { authFetch } from '../lib/api'

export default function AssessmentsPanel({ activeThreadId }) {
  const navigate = useNavigate()
  const [sessions, setSessions] = useState([])

  useEffect(() => {
    authFetch('/sessions')
      .then(r => r.json())
      .then(data => setSessions(data.filter(s => s.session_type === 'fraud')))
      .catch(() => {})
  }, [])

  return (
    <div style={{
      flex: '0 0 220px', background: '#0f2040',
      borderRight: '1px solid rgba(255,255,255,0.06)',
      display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden',
    }}>
      <div style={{ padding: '1.25rem 1rem 0.75rem', borderBottom: '1px solid rgba(255,255,255,0.06)', flexShrink: 0 }}>
        <h2 style={{ color: '#fff', fontSize: '0.85rem', fontWeight: 600, margin: '0 0 0.75rem', letterSpacing: '0.05em' }}>
          Assessments
        </h2>
        <button
          onClick={() => navigate('/fraud')}
          style={{ width: '100%', padding: '0.5rem', background: 'var(--gold-soft)', color: 'var(--gold)', border: '1px solid var(--gold-border)', borderRadius: 6, fontSize: '0.78rem', fontWeight: 600, cursor: 'pointer' }}
        >
          + New Assessment
        </button>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
        {sessions.length === 0 ? (
          <p style={{ color: 'rgba(255,255,255,0.3)', fontSize: '0.78rem', textAlign: 'center', marginTop: '1rem' }}>
            No assessments yet.
          </p>
        ) : sessions.map(session => {
          const isActive = activeThreadId === session.thread_id
          return (
            <div
              key={session.thread_id}
              onClick={() => navigate(`/fraud/${session.thread_id}`)}
              style={{
                padding: '0.7rem 0.8rem', borderRadius: 8, cursor: 'pointer',
                background: isActive ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.03)',
                border: `1px solid ${isActive ? 'rgba(201,168,76,0.3)' : 'rgba(255,255,255,0.05)'}`,
                transition: 'background 0.15s',
              }}
            >
              <p style={{ color: '#fff', fontSize: '0.82rem', fontWeight: 500, margin: '0 0 0.2rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {session.title}
              </p>
              <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.68rem', margin: 0 }}>
                {new Date(session.updated_at).toLocaleDateString()}
              </p>
            </div>
          )
        })}
      </div>
    </div>
  )
}
