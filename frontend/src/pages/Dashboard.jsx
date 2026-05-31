import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useOutletContext } from 'react-router-dom'
import { authFetch } from '../lib/api'

export default function Dashboard() {
  const navigate = useNavigate()
  const { darkMode, setDarkMode } = useOutletContext()
  const [sessions, setSessions] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [activeDetails, setActiveDetails] = useState(null)
  const [loadingDetails, setLoadingDetails] = useState(false)

  useEffect(() => {
    authFetch('/sessions')
      .then(r => r.json())
      .then(setSessions)
      .catch(() => setSessions([]))
  }, [])

  // Process data for KPIs and Charts
  const fraudRuns = sessions.filter(s => s.session_type === 'fraud')

  const high = fraudRuns.filter(s => s.title?.includes('HIGH')).length
  const medium = fraudRuns.filter(s => s.title?.includes('MEDIUM') && !s.title?.includes('LOW-MEDIUM')).length
  const lowMedium = fraudRuns.filter(s => s.title?.includes('LOW-MEDIUM')).length
  const low = fraudRuns.filter(s => s.title?.includes('LOW RISK')).length

  const totalFlagged = high + medium
  const totalCleared = lowMedium + low
  const flagRate = fraudRuns.length ? Math.round((totalFlagged / fraudRuns.length) * 100) : 0

  // Donut Chart Math
  const totalRisk = high + medium + (lowMedium + low)
  const circ = 2 * Math.PI * 40
  const highArc = totalRisk ? (high / totalRisk) * circ : 0
  const medArc = totalRisk ? (medium / totalRisk) * circ : 0
  const lowArc = totalRisk ? ((lowMedium + low) / totalRisk) * circ : 0

  const getPct = (val) => totalRisk ? Math.round((val / totalRisk) * 100) : 0

  const loadAssessmentDetails = async (tid) => {
    setSelectedId(tid)
    setLoadingDetails(true)
    try {
      const res = await authFetch(`/fraud/sessions/${tid}`)
      const data = await res.json()
      setActiveDetails(data)
    } catch {
      setActiveDetails(null)
    } finally {
      setLoadingDetails(false)
    }
  }

  // Theme colour tokens — dark mirrors the original palette, light uses app cream palette
  const t = darkMode ? {
    bg:                 '#0d1520',
    sidebar:            '#0a1628',
    sidebarBorder:      'rgba(255,255,255,0.04)',
    sidebarHeadBorder:  'rgba(255,255,255,0.06)',
    card:               '#111e2e',
    cardBorder:         'rgba(255,255,255,0.07)',
    text:               '#e2eaf5',
    textMid:            '#7a90a8',
    textMuted:          '#3d5268',
    tableHeader:        '#162030',
    tableHover:         '#1c2a3d',
    tableBorder:        'rgba(255,255,255,0.07)',
    recommendBg:        '#162030',
    recommendBorder:    'rgba(255,255,255,0.04)',
    detailRowBorder:    'rgba(255,255,255,0.04)',
    sessionActive:      'rgba(255,255,255,0.1)',
    sessionHover:       'rgba(255,255,255,0.05)',
    sessionTitle:       'rgba(255,255,255,0.6)',
    sessionDate:        'rgba(255,255,255,0.3)',
    btnNavBg:           'transparent',
    btnNavBorder:       'rgba(255,255,255,0.11)',
    btnNavColor:        '#7a90a8',
    overviewActiveBg:   '#c9a84c',
    overviewActiveColor:'#0a1628',
    overviewIdleBg:     'rgba(255,255,255,0.06)',
    overviewIdleBorder: 'rgba(255,255,255,0.1)',
    overviewIdleColor:  '#e2eaf5',
    toggleBg:           'rgba(255,255,255,0.06)',
    toggleBorder:       'rgba(255,255,255,0.1)',
    toggleColor:        'rgba(255,255,255,0.6)',
  } : {
    bg:                 '#f5f2ed',
    sidebar:            '#ede8df',
    sidebarBorder:      'rgba(0,0,0,0.04)',
    sidebarHeadBorder:  'rgba(0,0,0,0.07)',
    card:               '#ffffff',
    cardBorder:         'rgba(0,0,0,0.08)',
    text:               '#1a1a2e',
    textMid:            '#4a5568',
    textMuted:          '#8a96a8',
    tableHeader:        '#f0ece5',
    tableHover:         '#f5f0e8',
    tableBorder:        'rgba(0,0,0,0.06)',
    recommendBg:        '#f0ece5',
    recommendBorder:    'rgba(0,0,0,0.06)',
    detailRowBorder:    'rgba(0,0,0,0.05)',
    sessionActive:      'rgba(0,0,0,0.08)',
    sessionHover:       'rgba(0,0,0,0.04)',
    sessionTitle:       '#4a5568',
    sessionDate:        '#8a96a8',
    btnNavBg:           'transparent',
    btnNavBorder:       'rgba(0,0,0,0.12)',
    btnNavColor:        '#4a5568',
    overviewActiveBg:   '#c9a84c',
    overviewActiveColor:'#ffffff',
    overviewIdleBg:     'rgba(0,0,0,0.04)',
    overviewIdleBorder: 'rgba(0,0,0,0.1)',
    overviewIdleColor:  '#1a1a2e',
    toggleBg:           'rgba(0,0,0,0.04)',
    toggleBorder:       'rgba(0,0,0,0.1)',
    toggleColor:        '#4a5568',
  }

  const injectedStyles = `
    .dash-bg { background: ${t.bg}; color: ${t.text}; font-family: 'DM Sans', sans-serif; }
    .bg-card { background: ${t.card}; border: 1px solid ${t.cardBorder}; border-radius: 12px; }
    .text-mid { color: ${t.textMid}; }
    .text-muted { color: ${t.textMuted}; }
    .gold-text { color: #c9a84c; }
    .danger-text { color: #e05555; }
    .warning-text { color: #d4962a; }
    .success-text { color: #3aaa72; }

    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
    .kpi-card { padding: 18px; transition: border-color 0.2s; }
    .kpi-card:hover { border-color: ${darkMode ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.14)'}; }
    .kpi-label { font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: ${t.textMuted}; font-weight: 500; margin-bottom: 10px; }
    .kpi-val { font-size: 28px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; margin-bottom: 6px; line-height: 1; }

    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 20px; }
    .card-head { padding: 14px 18px; border-bottom: 1px solid ${t.cardBorder}; display: flex; justify-content: space-between; align-items: center; }

    .insight-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }
    .insight-icon { width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 15px; margin-bottom: 10px; }

    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { padding: 10px 16px; text-align: left; font-size: 10px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: ${t.textMuted}; border-bottom: 1px solid ${t.tableBorder}; background: ${t.tableHeader}; }
    td { padding: 12px 16px; border-bottom: 1px solid ${t.tableBorder}; color: ${t.text}; }
    tr:hover { background: ${t.tableHover}; cursor: pointer; }

    .badge { display: inline-flex; align-items: center; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 500; }
    .badge-danger { background: rgba(224,85,85,0.12); color: #e05555; border: 1px solid rgba(224,85,85,0.2); }
    .badge-warning { background: rgba(212,150,42,0.12); color: #d4962a; border: 1px solid rgba(212,150,42,0.2); }
    .badge-success { background: rgba(58,170,114,0.12); color: #3aaa72; border: 1px solid rgba(58,170,114,0.2); }

    .bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
    .bar-track { flex: 1; height: 7px; background: ${t.tableHeader}; border-radius: 4px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 4px; transition: width 0.8s ease; }
  `

  return (
    <div className="dash-bg" style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      <style>{injectedStyles}</style>

      {/* LEFT PANEL */}
      <div style={{ flex: '0 0 240px', background: t.sidebar, borderRight: `1px solid ${t.sidebarBorder}`, display: 'flex', flexDirection: 'column', paddingTop: '1.5rem' }}>
        <div style={{ padding: '0 1.25rem 1.5rem', borderBottom: `1px solid ${t.sidebarHeadBorder}` }}>
          <h2 style={{ color: t.text, fontSize: '1.1rem', fontWeight: 600, margin: 0 }}>Dashboard</h2>
          <button
            onClick={() => { setSelectedId(null); setActiveDetails(null) }}
            style={{ marginTop: '1rem', width: '100%', padding: '0.6rem', background: !selectedId ? t.overviewActiveBg : t.overviewIdleBg, color: !selectedId ? t.overviewActiveColor : t.overviewIdleColor, border: !selectedId ? 'none' : `1px solid ${t.overviewIdleBorder}`, borderRadius: 8, fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer', transition: 'all 0.2s' }}
          >
            📊 Global Overview
          </button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '0.75rem', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {fraudRuns.length === 0 ? (
            <p style={{ color: t.sessionDate, fontSize: '0.8rem', textAlign: 'center', marginTop: '1rem' }}>No history yet.</p>
          ) : fraudRuns.map(session => (
            <div
              key={session.thread_id}
              onClick={() => loadAssessmentDetails(session.thread_id)}
              style={{ padding: '0.6rem 0.8rem', background: selectedId === session.thread_id ? t.sessionActive : 'transparent', borderRadius: 8, cursor: 'pointer', transition: 'background 0.2s' }}
              onMouseEnter={e => { if (selectedId !== session.thread_id) e.currentTarget.style.background = t.sessionHover }}
              onMouseLeave={e => { if (selectedId !== session.thread_id) e.currentTarget.style.background = 'transparent' }}
            >
              <p style={{ color: selectedId === session.thread_id ? t.text : t.sessionTitle, fontSize: '0.85rem', fontWeight: 500, margin: '0 0 2px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {session.title || 'Untitled'}
              </p>
              <p style={{ color: t.sessionDate, fontSize: '0.7rem', margin: 0 }}>
                {new Date(session.updated_at).toLocaleDateString()}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* RIGHT PANEL */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px' }}>

        {/* TOP BAR */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <div>
            <h1 style={{ fontFamily: "'DM Serif Display', serif", fontSize: '22px', fontWeight: 400, margin: 0, color: t.text }}>
              {selectedId ? 'Claim Analysis' : 'Claims Overview'}
            </h1>
            <p className="text-muted" style={{ fontSize: '13px', marginTop: '4px', margin: 0 }}>
              {selectedId ? 'Detailed view of fraud assessment' : new Date().toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            {/* Theme toggle */}
            <button
              onClick={() => setDarkMode(v => !v)}
              title="Toggle theme"
              style={{ width: 32, height: 32, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 8, border: `1px solid ${t.toggleBorder}`, background: t.toggleBg, color: t.toggleColor, cursor: 'pointer', flexShrink: 0 }}
            >
              {darkMode ? (
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                  <circle cx="12" cy="12" r="5" />
                  <line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" />
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                  <line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" />
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                  <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
                </svg>
              )}
            </button>
            <button onClick={() => navigate('/chat')} style={{ height: '32px', padding: '0 16px', background: t.btnNavBg, border: `1px solid ${t.btnNavBorder}`, borderRadius: '8px', color: t.btnNavColor, fontSize: '12px', cursor: 'pointer' }}>
              ← Back to Chat
            </button>
            <button onClick={() => navigate('/fraud')} style={{ height: '32px', padding: '0 16px', background: 'rgba(201,168,76,0.12)', border: '1px solid rgba(201,168,76,0.3)', borderRadius: '8px', color: '#c9a84c', fontSize: '12px', fontWeight: 600, cursor: 'pointer' }}>
              New Assessment
            </button>
          </div>
        </div>

        {/* ── VIEW 1: GLOBAL OVERVIEW ── */}
        {!selectedId && (
          <div style={{ maxWidth: '1000px' }}>
            {/* KPI GRID */}
            <div className="kpi-grid">
              <div className="bg-card kpi-card">
                <p className="kpi-label">Total Assessments</p>
                <p className="kpi-val gold-text">{fraudRuns.length}</p>
                <p style={{ fontSize: '11px', color: t.textMid }}>Assessments completed</p>
              </div>
              <div className="bg-card kpi-card">
                <p className="kpi-label">Flag Rate</p>
                <p className="kpi-val">{flagRate}%</p>
                <p style={{ fontSize: '11px', color: t.textMid }}>Caught as suspicious</p>
              </div>
              <div className="bg-card kpi-card">
                <p className="kpi-label">Flagged Suspicious</p>
                <p className="kpi-val danger-text">{totalFlagged}</p>
                <p style={{ fontSize: '11px', color: '#e05555' }}>High + Medium risk</p>
              </div>
              <div className="bg-card kpi-card">
                <p className="kpi-label">Cleared Claims</p>
                <p className="kpi-val success-text">{totalCleared}</p>
                <p style={{ fontSize: '11px', color: '#3aaa72' }}>Low risk</p>
              </div>
            </div>

            {/* TWO COL: DONUT & BARS */}
            <div className="two-col">
              {/* Risk Donut */}
              <div className="bg-card">
                <div className="card-head">
                  <p style={{ fontSize: '13px', fontWeight: 600, margin: 0 }}>Risk Level Distribution</p>
                </div>
                <div style={{ padding: '18px', display: 'flex', alignItems: 'center', gap: '24px' }}>
                  <svg width="110" height="110" viewBox="0 0 110 110" style={{ transform: 'rotate(-90deg)' }}>
                    <circle cx="55" cy="55" r="40" fill="none" stroke={t.tableHeader} strokeWidth="18"/>
                    <circle cx="55" cy="55" r="40" fill="none" stroke="#e05555" strokeWidth="18" strokeDasharray={`${highArc} ${circ}`} strokeDashoffset={0} />
                    <circle cx="55" cy="55" r="40" fill="none" stroke="#d4962a" strokeWidth="18" strokeDasharray={`${medArc} ${circ}`} strokeDashoffset={-highArc} />
                    <circle cx="55" cy="55" r="40" fill="none" stroke="#3aaa72" strokeWidth="18" strokeDasharray={`${lowArc} ${circ}`} strokeDashoffset={-(highArc + medArc)} />
                    <text x="55" y="55" transform="rotate(90 55 55)" textAnchor="middle" dy=".3em" fontSize="16" fontWeight="600" fill={t.text} fontFamily="IBM Plex Mono">{totalRisk}</text>
                  </svg>
                  <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', fontSize: '12px' }}>
                      <div style={{ width: '9px', height: '9px', borderRadius: '50%', background: '#e05555', marginRight: '8px' }} />
                      <span className="text-mid" style={{ flex: 1 }}>High risk</span>
                      <span style={{ fontWeight: 600, fontFamily: 'IBM Plex Mono', color: t.text }}>{high}</span>
                      <span className="text-muted" style={{ marginLeft: '6px', fontSize: '11px' }}>({getPct(high)}%)</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', fontSize: '12px' }}>
                      <div style={{ width: '9px', height: '9px', borderRadius: '50%', background: '#d4962a', marginRight: '8px' }} />
                      <span className="text-mid" style={{ flex: 1 }}>Medium risk</span>
                      <span style={{ fontWeight: 600, fontFamily: 'IBM Plex Mono', color: t.text }}>{medium}</span>
                      <span className="text-muted" style={{ marginLeft: '6px', fontSize: '11px' }}>({getPct(medium)}%)</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', fontSize: '12px' }}>
                      <div style={{ width: '9px', height: '9px', borderRadius: '50%', background: '#3aaa72', marginRight: '8px' }} />
                      <span className="text-mid" style={{ flex: 1 }}>Low / Clear</span>
                      <span style={{ fontWeight: 600, fontFamily: 'IBM Plex Mono', color: t.text }}>{lowMedium + low}</span>
                      <span className="text-muted" style={{ marginLeft: '6px', fontSize: '11px' }}>({getPct(lowMedium + low)}%)</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Policy Bars */}
              <div className="bg-card">
                <div className="card-head">
                  <p style={{ fontSize: '13px', fontWeight: 600, margin: 0 }}>Fraud Flag Rate by Policy Type</p>
                </div>
                <div style={{ padding: '18px' }}>
                  {[['All Perils', '#e05555', '72%'], ['Third Party', '#e05555', '61%'], ['Collision', '#d4962a', '44%'], ['Liability', '#3aaa72', '18%']].map(([label, color, pct]) => (
                    <div key={label} className="bar-row">
                      <span style={{ fontSize: '12px', color: t.textMid, minWidth: '100px' }}>{label}</span>
                      <div className="bar-track"><div className="bar-fill" style={{ background: color, width: pct }} /></div>
                      <span style={{ fontSize: '12px', fontWeight: 600, fontFamily: 'IBM Plex Mono', color: t.text, minWidth: '36px', textAlign: 'right' }}>{pct}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* INSIGHTS */}
            <div className="insight-grid">
              {[
                { icon: '⚡', bg: 'rgba(212,150,42,0.12)', title: 'All Perils — highest flag rate', body: '72% of All Perils claims trigger a fraud flag. Apply extra scrutiny to this policy type.' },
                { icon: '🚨', bg: 'rgba(224,85,85,0.12)', title: 'No police report = stronger signal', body: 'Claims with no police report AND no witness are the strongest combined fraud indicator.' },
                { icon: '✅', bg: 'rgba(58,170,114,0.12)', title: 'Human review is working', body: 'All model outputs require staff sign-off before a decision. The model is a triage signal only.' },
              ].map(({ icon, bg, title, body }) => (
                <div key={title} className="bg-card insight-card" style={{ padding: '16px' }}>
                  <div className="insight-icon" style={{ background: bg }}>{icon}</div>
                  <p style={{ fontSize: '13px', fontWeight: 600, margin: '0 0 5px 0', color: t.text }}>{title}</p>
                  <p style={{ fontSize: '12px', color: t.textMid, lineHeight: 1.6, margin: 0 }}>{body}</p>
                </div>
              ))}
            </div>

            {/* TABLE */}
            <div className="bg-card" style={{ overflow: 'hidden' }}>
              <div className="card-head">
                <p style={{ fontSize: '13px', fontWeight: 600, margin: 0 }}>Recent Fraud Sessions</p>
              </div>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                <thead>
                  <tr>
                    <th>Session ID</th>
                    <th>Title</th>
                    <th>Risk Signal</th>
                    <th>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {fraudRuns.length === 0 ? (
                    <tr><td colSpan="4" style={{ textAlign: 'center', color: t.textMid, padding: '24px' }}>No sessions yet.</td></tr>
                  ) : fraudRuns.slice(0, 10).map(s => {
                    const isHigh = s.title?.includes('HIGH')
                    const isMed = s.title?.includes('MEDIUM') && !s.title?.includes('LOW-MEDIUM')
                    const badgeClass = isHigh ? 'badge-danger' : isMed ? 'badge-warning' : 'badge-success'
                    const badgeText = isHigh ? 'High Risk' : isMed ? 'Medium' : 'Low'
                    return (
                      <tr key={s.thread_id} onClick={() => loadAssessmentDetails(s.thread_id)}>
                        <td style={{ fontFamily: 'IBM Plex Mono', fontSize: '11px', color: t.textMuted }}>#{s.thread_id.slice(-8)}</td>
                        <td style={{ fontWeight: 500 }}>{s.title || 'Untitled'}</td>
                        <td><span className={`badge ${badgeClass}`}>{badgeText}</span></td>
                        <td className="text-mid">{new Date(s.updated_at).toLocaleDateString()}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── VIEW 2: INDIVIDUAL ANALYSIS ── */}
        {selectedId && loadingDetails && (
          <div style={{ padding: '40px', textAlign: 'center', color: t.textMid }}>Loading claim details...</div>
        )}

        {selectedId && !loadingDetails && activeDetails && (
          <div style={{ maxWidth: '800px', display: 'flex', flexDirection: 'column', gap: '20px' }}>

            <div className="bg-card" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div>
                  <h2 style={{ fontSize: '20px', fontWeight: 600, color: t.text, margin: '0 0 4px 0' }}>
                    {activeDetails.form_data?.Make || 'Vehicle'} Claim Analysis
                  </h2>
                  <p style={{ fontSize: '12px', color: t.textMid, margin: 0 }}>
                    Assessed on {new Date(activeDetails.created_at).toLocaleString()}
                  </p>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '28px', fontWeight: 800, color: activeDetails.result?.risk_level?.includes('HIGH') ? '#e05555' : activeDetails.result?.risk_level?.includes('MEDIUM') ? '#d4962a' : '#3aaa72' }}>
                    {activeDetails.result?.probability_pct}%
                  </div>
                  <div style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.05em', marginTop: '2px', color: t.textMid }}>
                    {activeDetails.result?.risk_level} RISK
                  </div>
                </div>
              </div>

              <div style={{ padding: '16px', borderRadius: '8px', background: t.recommendBg, border: `1px solid ${t.recommendBorder}` }}>
                <span style={{ fontSize: '13px', fontWeight: 600, color: t.text }}>AI Recommendation: </span>
                <span style={{ fontSize: '13px', color: t.textMid, lineHeight: 1.5 }}>{activeDetails.result?.recommendation}</span>
              </div>
            </div>

            {activeDetails.result?.factors?.length > 0 && (
              <div className="bg-card">
                <div className="card-head">
                  <p style={{ fontSize: '13px', fontWeight: 600, margin: 0 }}>Why this score</p>
                </div>
                <div style={{ padding: '18px' }}>
                  <div style={{ display: 'flex', gap: '16px', marginBottom: '14px' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: t.textMid }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#e05555', display: 'inline-block' }} />
                      Increases fraud risk
                    </span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: t.textMid }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#3aaa72', display: 'inline-block' }} />
                      Lowers fraud risk
                    </span>
                  </div>
                  {(() => {
                    const factors = activeDetails.result.factors
                    const maxImpact = Math.max(...factors.map(f => Math.abs(f.impact)))
                    const FEATURE_LABELS = {
                      Fault: 'Fault', BasePolicy: 'Base Policy', VehicleCategory: 'Vehicle Category',
                      Month: 'Accident Month', Age: 'Claimant Age', DayOfWeek: 'Day of Week',
                      Make: 'Vehicle Make', AgeOfPolicyHolder: 'Policyholder Age',
                      NumberOfSuppliments: 'Supplements', AgeOfVehicle: 'Vehicle Age',
                      PastNumberOfClaims: 'Past Claims', VehiclePrice: 'Vehicle Price',
                      Sex: 'Gender', PoliceReportFiled: 'Police Report',
                      Days_Policy_Accident: 'Policy to Accident', Days_Policy_Claim: 'Policy to Claim',
                      WitnessPresent: 'Witness', AccidentArea: 'Accident Area',
                    }
                    return factors.map((f, i) => {
                      const isRisk = f.direction === 'increases risk'
                      const barW = Math.round((Math.abs(f.impact) / maxImpact) * 100)
                      const barColor = isRisk ? '#e05555' : '#3aaa72'
                      const label = FEATURE_LABELS[f.feature] || f.feature
                      return (
                        <div key={i} className="bar-row">
                          <span style={{ fontSize: '12px', color: t.textMid, minWidth: '180px' }}>
                            {label} = <span style={{ color: t.text }}>{String(f.value)}</span>
                          </span>
                          <div className="bar-track">
                            <div className="bar-fill" style={{ background: barColor, width: `${barW}%` }} />
                          </div>
                          <span style={{ fontSize: '11px', fontWeight: 600, fontFamily: 'IBM Plex Mono', color: barColor, minWidth: '52px', textAlign: 'right' }}>
                            {isRisk ? '+' : '-'}{Math.round(Math.abs(f.impact) * 100)}%
                          </span>
                        </div>
                      )
                    })
                  })()}
                </div>
              </div>
            )}

            <div className="bg-card">
              <div className="card-head">
                <p style={{ fontSize: '13px', fontWeight: 600, margin: 0 }}>Input Data Summary</p>
              </div>
              <div style={{ padding: '20px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                {(() => {
                  const fd = activeDetails.form_data || {}
                  const displayData = fd.processed || fd
                  return Object.entries(displayData)
                    .filter(([k, v]) => typeof v !== 'object' && k !== 'raw')
                    .map(([key, value]) => (
                      <div key={key} style={{ borderBottom: `1px solid ${t.detailRowBorder}`, paddingBottom: '8px' }}>
                        <div style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.05em', color: t.textMuted, marginBottom: '2px' }}>
                          {key.replace(/([A-Z])/g, ' $1').trim()}
                        </div>
                        <div style={{ fontSize: '13px', color: t.text, fontWeight: 500 }}>
                          {value !== '' && value !== null ? String(value) : <span style={{ color: t.textMid }}>Not Provided</span>}
                        </div>
                      </div>
                    ))
                })()}
              </div>
            </div>

          </div>
        )}

      </div>
    </div>
  )
}
