import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useOutletContext } from 'react-router-dom'
import { authFetch } from '../lib/api'
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend
} from 'recharts'

export default function Analytics() {
  const navigate = useNavigate()
  const { darkMode, setDarkMode } = useOutletContext()
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    authFetch('/sessions')
      .then(r => r.json())
      .then(data => {
        setSessions(data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  // --- DATA PROCESSING ---
  const fraudRuns = sessions.filter(s => s.session_type === 'fraud')

  // 1. Time-Series: Last 14 Days
  const last14Days = [...Array(14)].map((_, i) => {
    const d = new Date()
    d.setDate(d.getDate() - (13 - i))
    return d.toISOString().split('T')[0]
  })

  const dailyData = last14Days.map(date => ({
    date: new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    fullDate: date,
    count: 0
  }))

  fraudRuns.forEach(s => {
    if (!s.updated_at) return
    const dateStr = new Date(s.updated_at).toISOString().split('T')[0]
    const dayMatch = dailyData.find(d => d.fullDate === dateStr)
    if (dayMatch) dayMatch.count++
  })

  // 2. Cross-Variable: Risk by Vehicle Make
  const makeStats = {}
  let highCount = 0, medCount = 0, lowCount = 0

  fraudRuns.forEach(s => {
    const parts = (s.title || '').split(' — ')
    const make = parts[0] || 'Unknown'
    const riskRaw = parts[1] || ''

    if (riskRaw.includes('HIGH')) highCount++
    else if (riskRaw.includes('MEDIUM') && !riskRaw.includes('LOW-MEDIUM')) medCount++
    else lowCount++

    if (!makeStats[make]) makeStats[make] = { name: make, total: 0, flagged: 0 }
    makeStats[make].total++
    if (riskRaw.includes('HIGH') || (riskRaw.includes('MEDIUM') && !riskRaw.includes('LOW-MEDIUM'))) {
      makeStats[make].flagged++
    }
  })

  const makeChartData = Object.values(makeStats)
    .sort((a, b) => b.total - a.total)
    .slice(0, 7)

  // 3. Risk Distribution Pie Data
  const pieData = [
    { name: 'High Risk', value: highCount, color: '#e05555' },
    { name: 'Medium Risk', value: medCount, color: '#d4962a' },
    { name: 'Low / Clear', value: lowCount, color: '#3aaa72' }
  ].filter(d => d.value > 0)

  // Theme colour tokens
  const t = darkMode ? {
    bg:           '#0d1520',
    card:         '#111e2e',
    cardBorder:   'rgba(255,255,255,0.07)',
    text:         '#e2eaf5',
    textMid:      '#7a90a8',
    headerBorder: 'rgba(255,255,255,0.04)',
    btnBorder:    'rgba(255,255,255,0.11)',
    btnColor:     '#7a90a8',
    tooltipBg:    '#0a1628',
    tooltipBorder:'rgba(255,255,255,0.1)',
    gridStroke:   'rgba(255,255,255,0.04)',
    axisColor:    '#7a90a8',
    barTotal:     '#1a3560',
    toggleBg:     'rgba(255,255,255,0.06)',
    toggleBorder: 'rgba(255,255,255,0.1)',
    toggleColor:  'rgba(255,255,255,0.6)',
  } : {
    bg:           '#f5f2ed',
    card:         '#ffffff',
    cardBorder:   'rgba(0,0,0,0.08)',
    text:         '#1a1a2e',
    textMid:      '#4a5568',
    headerBorder: 'rgba(0,0,0,0.06)',
    btnBorder:    'rgba(0,0,0,0.12)',
    btnColor:     '#4a5568',
    tooltipBg:    '#ffffff',
    tooltipBorder:'rgba(0,0,0,0.12)',
    gridStroke:   'rgba(0,0,0,0.06)',
    axisColor:    '#4a5568',
    barTotal:     '#c9a84c',
    toggleBg:     'rgba(0,0,0,0.04)',
    toggleBorder: 'rgba(0,0,0,0.1)',
    toggleColor:  '#4a5568',
  }

  const injectedStyles = `
    .analytics-bg { background: ${t.bg}; color: ${t.text}; font-family: 'DM Sans', sans-serif; min-height: 100%; }
    .a-card { background: ${t.card}; border: 1px solid ${t.cardBorder}; border-radius: 12px; }
    .a-card-head { padding: 16px 20px; border-bottom: 1px solid ${t.cardBorder}; }
    .recharts-tooltip-wrapper { outline: none; }
  `

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{ background: t.tooltipBg, border: `1px solid ${t.tooltipBorder}`, padding: '10px 14px', borderRadius: '8px' }}>
          <p style={{ margin: '0 0 6px 0', fontSize: '12px', color: t.textMid }}>{label}</p>
          {payload.map((p, i) => (
            <p key={i} style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: p.color || '#c9a84c' }}>
              {p.name}: {p.value}
            </p>
          ))}
        </div>
      )
    }
    return null
  }

  if (loading) return (
    <div style={{ background: t.bg, color: t.textMid, padding: '40px', textAlign: 'center', minHeight: '100%' }}>
      Loading analytics...
    </div>
  )

  return (
    <div className="analytics-bg">
      <style>{injectedStyles}</style>

      {/* HEADER */}
      <div style={{ padding: '30px 40px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: `1px solid ${t.headerBorder}` }}>
        <div>
          <h1 style={{ fontFamily: "'DM Serif Display', serif", fontSize: '26px', fontWeight: 400, margin: 0, color: t.text }}>Strategic Analytics</h1>
          <p style={{ fontSize: '13px', color: t.textMid, margin: '4px 0 0 0' }}>Macro trends and fraud model performance</p>
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
          <button onClick={() => navigate('/chat')} style={{ height: '36px', padding: '0 16px', background: 'transparent', border: `1px solid ${t.btnBorder}`, borderRadius: '8px', color: t.btnColor, fontSize: '12px', cursor: 'pointer' }}>
            ← Back to Workspace
          </button>
        </div>
      </div>

      <div style={{ padding: '30px 40px', maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>

        {/* ROW 1: Trend Line & Pie */}
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>

          {/* Trend Line */}
          <div className="a-card">
            <div className="a-card-head">
              <h2 style={{ fontSize: '14px', fontWeight: 600, margin: 0, color: t.text }}>Assessment Volume (Last 14 Days)</h2>
            </div>
            <div style={{ padding: '20px', height: '300px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={dailyData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#c9a84c" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#c9a84c" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={t.gridStroke} vertical={false} />
                  <XAxis dataKey="date" tick={{ fill: t.axisColor, fontSize: 11 }} axisLine={false} tickLine={false} dy={10} />
                  <YAxis allowDecimals={false} tick={{ fill: t.axisColor, fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="count" name="Assessments" stroke="#c9a84c" strokeWidth={3} fillOpacity={1} fill="url(#colorCount)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Risk Pie */}
          <div className="a-card">
            <div className="a-card-head">
              <h2 style={{ fontSize: '14px', fontWeight: 600, margin: 0, color: t.text }}>Overall Risk Distribution</h2>
            </div>
            <div style={{ padding: '20px', height: '300px', display: 'flex', flexDirection: 'column' }}>
              {pieData.length === 0 ? (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: t.textMid, fontSize: '13px' }}>No data yet</div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={pieData} cx="50%" cy="45%" innerRadius={60} outerRadius={85} paddingAngle={2} dataKey="value" stroke="none">
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                    <Legend verticalAlign="bottom" height={36} iconType="circle" wrapperStyle={{ fontSize: '12px', color: t.textMid }} />
                  </PieChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>

        {/* ROW 2: Bar Chart (Vehicle Makes) */}
        <div className="a-card">
          <div className="a-card-head" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2 style={{ fontSize: '14px', fontWeight: 600, margin: 0, color: t.text }}>Fraud Flags by Vehicle Make</h2>
            <span style={{ fontSize: '11px', color: t.textMid, padding: '4px 10px', background: darkMode ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)', borderRadius: '12px' }}>Top 7 volume</span>
          </div>
          <div style={{ padding: '20px 20px 10px 0', height: '320px' }}>
            {makeChartData.length === 0 ? (
              <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: t.textMid, fontSize: '13px' }}>No vehicle data available</div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={makeChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={t.gridStroke} vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: t.axisColor, fontSize: 11 }} axisLine={false} tickLine={false} dy={10} />
                  <YAxis allowDecimals={false} tick={{ fill: t.axisColor, fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill: darkMode ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)' }} />
                  <Bar dataKey="total" name="Total Claims" fill={t.barTotal} radius={[4, 4, 0, 0]} />
                  <Bar dataKey="flagged" name="Flagged (High/Med)" fill="#e05555" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

      </div>
    </div>
  )
}
