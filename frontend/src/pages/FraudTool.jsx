import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getFxRate, assessClaim, createFraudSession, saveFraudResult, authFetch } from '../lib/api'
import { nairaToUsd, usdToVehiclePriceBin } from '../lib/fraud'
import FraudChatPanel from '../components/FraudChatPanel'

const labelStyle = {
  display: 'block', fontSize: '0.78rem', color: 'var(--text-secondary)',
  marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.05em',
}

const inputStyle = {
  width: '100%', padding: '0.45rem 0.6rem', borderRadius: 6, fontSize: '0.85rem',
  background: 'var(--input-bg)', border: '1px solid var(--input-border)',
  color: 'var(--text-dark)', outline: 'none', boxSizing: 'border-box',
}

const toggleStyle = {
  background: 'none', border: '1px solid var(--input-border)',
  color: 'var(--text-secondary)', padding: '0.4rem 0.8rem', borderRadius: 6,
  fontSize: '0.8rem', cursor: 'pointer', marginBottom: '0.25rem',
}

const submitStyle = {
  marginTop: '1rem', padding: '0.6rem 1.4rem', borderRadius: 8,
  background: 'var(--gold)', color: '#0a1628', fontWeight: 700, fontSize: '0.9rem',
  border: 'none', cursor: 'pointer',
}

const optStyle = { background: 'var(--input-bg)', color: 'var(--text-dark)' }

const FIELDS = {
  Fault:               { type: 'select',      label: 'Fault',                       opts: ['Policy Holder', 'Third Party'] },
  BasePolicy:          { type: 'select',      label: 'Cover Type',                  opts: ['Liability', 'Collision', 'All Perils'] },
  VehicleCategory:     { type: 'select',      label: 'Vehicle Category',            opts: ['Saloon', 'Sports Utility', 'Jeep', 'Truck', 'Pickup', 'Sport', 'Utility', 'Sedan'] },
  Month:               { type: 'select',      label: 'Accident Month',              opts: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] },
  DayOfWeek:           { type: 'select',      label: 'Accident Day',                opts: ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'] },
  DayOfWeekClaimed:    { type: 'select',      label: 'Claim Day',                   opts: ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'] },
  MonthClaimed:        { type: 'select',      label: 'Claim Month',                 opts: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] },
  Make:                { type: 'select+text', label: 'Vehicle Make',                opts: ['Honda','Toyota','Ford','Mazda','Chevrolet','Pontiac','Dodge','BMW','Mercedes','VW','Hyundai','Kia','Nissan','Peugeot','Lexus'] },
  Sex:                 { type: 'select',      label: 'Gender',                      opts: ['Male','Female'] },
  PoliceReportFiled:   { type: 'select',      label: 'Police Report Filed',         opts: ['Yes','No'] },
  AgeOfVehicle:        { type: 'text',        label: 'Age of Vehicle (years)' },
  AgeOfPolicyHolder:   { type: 'text',        label: 'Age of Policyholder (years)' },
  Age:                 { type: 'text',        label: 'Claimant Age' },
  Year:                { type: 'text',        label: 'Claim Year' },
  PastNumberOfClaims:  { type: 'text',        label: 'Past Number of Claims' },
  NumberOfSuppliments: { type: 'text',        label: 'No. of Repair Supplements' },
}

const OPTIONAL_FIELDS = {
  MaritalStatus:        { type: 'select', label: 'Marital Status',            opts: ['Single','Married','Widow','Divorced'] },
  AccidentArea:         { type: 'select', label: 'Accident Area',             opts: ['Urban','Rural'] },
  WitnessPresent:       { type: 'select', label: 'Witness Present',           opts: ['Yes','No'] },
  AgentType:            { type: 'select', label: 'Agent Type',                opts: ['Internal','External'] },
  Days_Policy_Accident: { type: 'select', label: 'Days: Policy → Accident',  opts: ['none','1 to 7','8 to 15','15 to 30','more than 30'] },
  Days_Policy_Claim:    { type: 'select', label: 'Days: Policy → Claim',     opts: ['none','1 to 7','8 to 15','15 to 30','more than 30'] },
  AddressChange_Claim:  { type: 'select', label: 'Address Change (Claim)',   opts: ['no change','under 6 months','1 year','2 to 3 years','4 to 8 years'] },
  NumberOfCars:         { type: 'select', label: 'Number of Cars Insured',   opts: ['1 vehicle','2 vehicles','3 to 4','5 to 8','more than 8'] },
  WeekOfMonth:          { type: 'text',   label: 'Week of Month (Accident)' },
  WeekOfMonthClaimed:   { type: 'text',   label: 'Week of Month (Claim)' },
  DriverRating:         { type: 'text',   label: 'Driver Rating (1–4)' },
}

const RISK_COLORS = {
  HIGH:        { bg: 'rgba(239,68,68,0.15)',  border: '#ef4444', text: '#fca5a5' },
  MEDIUM:      { bg: 'rgba(251,191,36,0.15)', border: '#fbbf24', text: '#fcd34d' },
  'LOW-MEDIUM':{ bg: 'rgba(251,191,36,0.1)',  border: '#f59e0b', text: '#fde68a' },
  LOW:         { bg: 'rgba(34,197,94,0.15)',  border: '#22c55e', text: '#86efac' },
}

function mapAgeOfVehicle(n) {
  if (n <= 0) return 'new'
  if (n <= 2) return '2 years'
  if (n <= 3) return '3 years'
  if (n <= 4) return '4 years'
  if (n <= 5) return '5 years'
  if (n <= 6) return '6 years'
  if (n <= 7) return '7 years'
  return 'more than 7'
}

function mapAgeOfHolder(n) {
  if (n <= 17) return '16 to 17'
  if (n <= 20) return '18 to 20'
  if (n <= 25) return '21 to 25'
  if (n <= 30) return '26 to 30'
  if (n <= 35) return '31 to 35'
  if (n <= 40) return '36 to 40'
  if (n <= 50) return '41 to 50'
  if (n <= 65) return '51 to 65'
  return 'over 65'
}

function mapPastClaims(n) {
  if (!n || n === 0) return 'none'
  if (n === 1) return '1'
  if (n <= 3) return '2 to 3'
  return 'more than 3'
}

function mapSupplements(n) {
  if (!n || n === 0) return 'none'
  if (n <= 2) return '1 to 2'
  if (n <= 5) return '3 to 5'
  return 'more than 5'
}

function renderField(key, def, value, onChange) {
  if (def.type === 'text') {
    return (
      <div key={key}>
        <label style={labelStyle}>{def.label}</label>
        <input type="number" placeholder="Enter value…" value={value || ''} onChange={e => onChange(e.target.value)} style={inputStyle} />
      </div>
    )
  }

  if (def.type === 'select+text') {
    const isOther = value === '__other__' || (value && !def.opts.includes(value))
    const textValue = isOther && value !== '__other__' ? value : ''
    return (
      <div key={key}>
        <label style={labelStyle}>{def.label}</label>
        <select value={isOther ? '__other__' : (value || '')} onChange={e => onChange(e.target.value)} style={inputStyle}>
          <option value="">Select…</option>
          {def.opts.map(o => <option key={o} value={o} style={optStyle}>{o}</option>)}
          <option value="__other__" style={optStyle}>Other (type below)…</option>
        </select>
        {isOther && (
          <input type="text" placeholder="Type vehicle make…" value={textValue} onChange={e => onChange(e.target.value)} style={{ ...inputStyle, marginTop: 4 }} />
        )}
      </div>
    )
  }

  return (
    <div key={key}>
      <label style={labelStyle}>{def.label}</label>
      <select value={value || ''} onChange={e => onChange(e.target.value)} style={inputStyle}>
        <option value="">Select…</option>
        {def.opts.map(o => <option key={o} value={o} style={optStyle}>{o}</option>)}
      </select>
    </div>
  )
}

export default function FraudTool() {
  const { threadId: urlThreadId } = useParams()
  const navigate = useNavigate()
  const [fxRate, setFxRate]             = useState(1600)
  const [vehicleNgn, setVehicleNgn]     = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [loading, setLoading]           = useState(false)
  const [result, setResult]             = useState(null)
  const [error, setError]               = useState('')
  const [threadId, setThreadId]         = useState(urlThreadId || null)
  const [renamingId, setRenamingId]     = useState(null)
  const [renameValue, setRenameValue]   = useState('')
  const [confirmDeleteId, setConfirmDeleteId] = useState(null)
  
  // NEW: State to hold the dedicated Assessment History
  const [history, setHistory]           = useState([])
  const PENDING_ID = '__pending__'

  const [exifResult, setExifResult]   = useState(null)

  const handleExifResult = (result) => {
    setExifResult(result)
    const tid = threadId || urlThreadId
    if (tid && tid !== PENDING_ID) {
      localStorage.setItem(`exif_${tid}`, JSON.stringify(result))
    }
  }

  const [form, setForm] = useState(Object.fromEntries(Object.keys(FIELDS).map(k => [k, ''])))
  const [advanced, setAdvanced] = useState(Object.fromEntries(Object.keys(OPTIONAL_FIELDS).map(k => [k, ''])))

  const fetchHistory = () => {
    authFetch('/sessions')
      .then(r => r.json())
      .then(data => setHistory(data.filter(s => s.session_type === 'fraud')))
      .catch(() => setHistory([]))
  }

  useEffect(() => {
    getFxRate().then(setFxRate).catch(() => setFxRate(1600))
    fetchHistory()
  }, [])

  useEffect(() => {
    // Clear form whenever URL session changes (including going to /fraud with no ID)
    setResult(null)
    setError('')
    setVehicleNgn('')
    setForm(Object.fromEntries(Object.keys(FIELDS).map(k => [k, ''])))
    setAdvanced(Object.fromEntries(Object.keys(OPTIONAL_FIELDS).map(k => [k, ''])))
    setThreadId(urlThreadId || null)

    if (!urlThreadId) { setExifResult(null); return }

    try {
      const saved = localStorage.getItem(`exif_${urlThreadId}`)
      setExifResult(saved ? JSON.parse(saved) : null)
    } catch { setExifResult(null) }
    authFetch(`/fraud/sessions/${urlThreadId}`)
      .then(r => r.json())
      .then(data => {
        if (!data.form_data) return
        // New structure: form_data.raw — written by saveFraudResult
        const raw = data.form_data.raw
        if (raw) {
          if (raw.vehicleNgn) setVehicleNgn(String(raw.vehicleNgn))
          if (raw.form)       setForm(f => ({ ...f, ...raw.form }))
          if (raw.advanced)   setAdvanced(a => ({ ...a, ...raw.advanced }))
        } else {
          // Fallback: old flat structure written directly by /fraud/assess
          const fd = data.form_data
          if (data.vehicle_ngn) setVehicleNgn(String(data.vehicle_ngn))
          const nf = {}; Object.keys(FIELDS).forEach(k => nf[k] = fd[k] !== undefined ? String(fd[k]) : '')
          setForm(nf)
          const na = {}; Object.keys(OPTIONAL_FIELDS).forEach(k => na[k] = fd[k] !== undefined ? String(fd[k]) : '')
          setAdvanced(na)
        }
        if (data.result) setResult(data.result)
      })
      .catch(err => console.error('Load assessment failed:', err))
  }, [urlThreadId])

  const vehicleUsd = vehicleNgn ? nairaToUsd(parseFloat(vehicleNgn), fxRate) : null
  const vehicleBin = vehicleUsd ? usdToVehiclePriceBin(vehicleUsd) : null

  const set    = (key, val) => setForm(f => ({ ...f, [key]: val }))
  const setAdv = (key, val) => setAdvanced(a => ({ ...a, [key]: val }))

  async function ensureThreadId() {
    if (threadId && threadId !== PENDING_ID) return threadId
    const data = await createFraudSession()
    const tid = data.thread_id
    setThreadId(tid)
    fetchHistory()
    return tid
  }

  const loadAssessment = (tid) => {
    if (tid === PENDING_ID) return
    setHistory(h => h.filter(s => s.thread_id !== PENDING_ID))
    navigate(`/fraud/${tid}`)
  }

  const handleNewAssessment = () => {
    setHistory(h => [
      { thread_id: PENDING_ID, title: 'New Assessment', session_type: 'fraud', updated_at: new Date().toISOString(), _pending: true },
      ...h.filter(s => s.thread_id !== PENDING_ID),
    ])
    navigate('/fraud')
  }

  function startRename(e, session) {
    e.stopPropagation()
    setRenamingId(session.thread_id)
    setRenameValue(session.title || 'Untitled')
    setConfirmDeleteId(null)
  }

  async function commitRename(threadIdToRename) {
    const title = renameValue.trim() || 'Untitled'
    setRenamingId(null)
    try {
      await authFetch(`/sessions/${threadIdToRename}`, {
        method: 'PATCH',
        body: JSON.stringify({ title }),
      })
      fetchHistory()
    } catch {}
  }

  function startDelete(e, threadIdToDelete) {
    e.stopPropagation()
    setConfirmDeleteId(threadIdToDelete)
    setRenamingId(null)
  }

  async function confirmDelete(threadIdToDelete) {
    try {
      await authFetch(`/sessions/${threadIdToDelete}`, { method: 'DELETE' })
      localStorage.removeItem(`exif_${threadIdToDelete}`)
      if (threadId === threadIdToDelete) handleNewAssessment()
      fetchHistory()
    } catch {}
    setConfirmDeleteId(null)
  }

  const handleFieldsExtracted = (fields) => {
    Object.entries(fields).forEach(([key, value]) => {
      if (key === 'VehicleValueNGN') {
        setVehicleNgn(String(value))
      } else if (FIELDS[key]) {
        set(key, String(value))
      } else if (OPTIONAL_FIELDS[key]) {
        setAdv(key, String(value))
      }
    })
  }

  const handleSubmit = async () => {
    if (!vehicleBin) { setError('Enter vehicle value'); return }
    const missing = Object.keys(FIELDS).filter(k => !form[k] || form[k] === '__other__')
    if (missing.length) {
      setError(`Fill in: ${missing.map(k => FIELDS[k].label).join(', ')}`)
      return
    }
    setError(''); setLoading(true); setResult(null)
    try {
      const payload = {
        ...form,
        Age:                parseInt(form.Age),
        Year:               parseInt(form.Year),
        VehiclePrice:       vehicleBin,
        AgeOfVehicle:       mapAgeOfVehicle(parseInt(form.AgeOfVehicle)),
        AgeOfPolicyHolder:  mapAgeOfHolder(parseInt(form.AgeOfPolicyHolder)),
        PastNumberOfClaims: mapPastClaims(parseInt(form.PastNumberOfClaims)),
        NumberOfSuppliments:mapSupplements(parseInt(form.NumberOfSuppliments)),
        ...Object.fromEntries(Object.entries(advanced).filter(([, v]) => v !== '')),
        vehicle_ngn: parseFloat(vehicleNgn) || 0,
        ...(threadId && threadId !== PENDING_ID ? { thread_id: threadId } : {}),
      }
      const data = await assessClaim(payload)
      setResult(data)

      if (data?.thread_id) {
        setThreadId(data.thread_id)
        setHistory(h => h.filter(s => s.thread_id !== PENDING_ID))
        saveFraudResult(data.thread_id, {
          form_data: {
            processed: payload,
            raw: { vehicleNgn, form, advanced },
          },
          result:       data,
          vehicle_ngn:  parseFloat(vehicleNgn),
          vehicle_make: form.Make,
          base_policy:  form.BasePolicy,
        }).catch(err => console.error('Save failed:', err))
      }
      fetchHistory()
    } catch {
      setError('Assessment failed. Check backend.')
    } finally {
      setLoading(false)
    }
  }

  const colors = result ? RISK_COLORS[result.risk_level] || RISK_COLORS.LOW : null

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: '#0a1628' }}>
      
      {/* NEW: Dedicated Fraud History Panel (Left side) */}
      <div style={{ flex: '0 0 240px', background: '#0f2040', borderRight: '1px solid rgba(255,255,255,0.06)', display: 'flex', flexDirection: 'column', paddingTop: '3rem' }}>
        <div style={{ padding: '1.5rem 1.25rem', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          <h2 style={{ color: '#fff', fontSize: '0.95rem', fontWeight: 600, margin: 0, letterSpacing: '0.05em' }}>Assessments</h2>
          <button onClick={handleNewAssessment} style={{ marginTop: '1rem', width: '100%', padding: '0.6rem', background: 'var(--gold-soft)', color: 'var(--gold)', border: '1px solid var(--gold-border)', borderRadius: 6, fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer' }}>
            + New Assessment
          </button>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
           {history.length === 0 ? (
             <p style={{ color: 'rgba(255,255,255,0.3)', fontSize: '0.8rem', textAlign: 'center', marginTop: '1rem' }}>No history yet.</p>
           ) : history.map(session => {
             const isPending = session._pending === true
             const isActive = isPending ? threadId === null : threadId === session.thread_id
             return (
               <div
                 key={session.thread_id}
                 onClick={() => !isPending && loadAssessment(session.thread_id)}
                 className={isPending ? '' : 'group'}
                 style={{ padding: '0.8rem', background: isActive ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.03)', borderRadius: 8, cursor: isPending ? 'default' : 'pointer', border: `1px solid ${isPending ? 'rgba(201,168,76,0.25)' : 'rgba(255,255,255,0.05)'}`, transition: 'background 0.2s', opacity: isPending ? 0.75 : 1 }}
               >
                 {isPending ? (
                   <p style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.85rem', fontWeight: 500, margin: '0 0 0.3rem', fontStyle: 'italic' }}>
                     New Assessment
                   </p>
                 ) : renamingId === session.thread_id ? (
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
                     style={{ width: '100%', marginBottom: '0.3rem', fontSize: '0.8rem', borderRadius: 6, padding: '0.35rem 0.45rem', outline: 'none', border: '0.5px solid var(--gold-border)', background: 'rgba(255,255,255,0.08)', color: '#fff' }}
                   />
                 ) : confirmDeleteId === session.thread_id ? (
                   <div onClick={e => e.stopPropagation()} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: '0.3rem' }}>
                     <span style={{ flex: 1, color: '#f87171', fontSize: '0.78rem' }}>Delete?</span>
                     <button onClick={() => confirmDelete(session.thread_id)} style={{ width: 18, height: 18, borderRadius: 4, border: 'none', background: 'rgba(229,62,62,0.2)', color: '#f87171', cursor: 'pointer' }}>✓</button>
                     <button onClick={() => setConfirmDeleteId(null)} style={{ width: 18, height: 18, borderRadius: 4, border: 'none', background: 'rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.45)', cursor: 'pointer' }}>✕</button>
                   </div>
                 ) : (
                   <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: '0.3rem' }}>
                     <p style={{ flex: 1, color: '#fff', fontSize: '0.85rem', fontWeight: 500, margin: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                       {session.title}
                     </p>
                     <button onClick={e => startRename(e, session)} className="hidden group-hover:flex" style={{ width: 16, height: 16, alignItems: 'center', justifyContent: 'center', borderRadius: 4, border: 'none', background: 'transparent', color: 'rgba(255,255,255,0.35)', cursor: 'pointer' }} title="Rename">
                       <svg viewBox="0 0 24 24" width="10" height="10" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                         <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" />
                         <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" />
                       </svg>
                     </button>
                     <button onClick={e => startDelete(e, session.thread_id)} className="hidden group-hover:flex" style={{ width: 16, height: 16, alignItems: 'center', justifyContent: 'center', borderRadius: 4, border: 'none', background: 'transparent', color: 'rgba(255,255,255,0.35)', cursor: 'pointer' }} title="Delete">🗑️</button>
                   </div>
                 )}
                 <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.7rem', margin: 0 }}>
                   {isPending ? 'Not yet saved' : new Date(session.updated_at).toLocaleDateString()}
                 </p>
               </div>
             )
           })}
        </div>
      </div>

      {/* Main content: form (middle) + chat panel (right) */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* Form column */}
        <div style={{ flex: '0 0 55%', overflowY: 'auto', padding: '2rem', color: '#e2e8f0', background: 'var(--cream)' }}>
          <h1 style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--gold)', marginBottom: '1.25rem' }}>
            Fraud Assessment Tool
          </h1>

          {/* Vehicle Value */}
          <div style={{ marginBottom: '1rem' }}>
            <label style={labelStyle}>Vehicle Value (₦)</label>
            <input
              type="text"
              inputMode="numeric"
              placeholder="e.g. 15,000,000"
              value={vehicleNgn ? Number(vehicleNgn).toLocaleString('en-US') : ''}
              onChange={e => {
                const raw = e.target.value.replace(/,/g, '')
                if (raw === '' || /^\d+$/.test(raw)) setVehicleNgn(raw)
              }}
              style={inputStyle}
            />
          </div>

          {/* Required fields */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1rem' }}>
            {Object.entries(FIELDS).map(([key, def]) =>
              renderField(key, def, form[key], val => set(key, val))
            )}
          </div>

          {/* Advanced toggle */}
          <button onClick={() => setShowAdvanced(s => !s)} style={toggleStyle}>
            {showAdvanced ? '▾ Hide' : '▸ Show'} advanced fields (optional — auto-filled if skipped)
          </button>

          {showAdvanced && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', margin: '0.75rem 0' }}>
              {Object.entries(OPTIONAL_FIELDS).map(([key, def]) =>
                renderField(key, def, advanced[key], val => setAdv(key, val))
              )}
            </div>
          )}

          {error && <p style={{ color: '#f87171', fontSize: '0.83rem', margin: '0.5rem 0' }}>{error}</p>}

          <button onClick={handleSubmit} disabled={loading} style={submitStyle}>
            {loading ? 'Assessing…' : '🔍 Run Fraud Assessment'}
          </button>

          {/* Result card */}
          {result && (() => {
            const maxImpact = result.factors?.length
              ? Math.max(...result.factors.map(f => Math.abs(f.impact)))
              : 1

            const FEATURE_LABELS = {
              Fault: 'Fault', BasePolicy: 'Base Policy', VehicleCategory: 'Vehicle Category',
              Month: 'Accident Month', Age: 'Claimant Age', DayOfWeek: 'Day of Week',
              Make: 'Vehicle Make', AgeOfPolicyHolder: 'Policyholder Age',
              NumberOfSuppliments: 'Supplements', AgeOfVehicle: 'Vehicle Age',
              PastNumberOfClaims: 'Past Claims', VehiclePrice: 'Vehicle Price',
              Sex: 'Gender', PoliceReportFiled: 'Police Report',
              Days_Policy_Accident: 'Policy > Accident', Days_Policy_Claim: 'Policy > Claim',
              WitnessPresent: 'Witness', AccidentArea: 'Accident Area',
            }

            return (
              <div style={{ marginTop: '1.5rem', padding: '1.25rem', borderRadius: 10,
                background: colors.bg, border: `1px solid ${colors.border}` }}>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '1.1rem', fontWeight: 700, color: colors.text }}>
                    {result.risk_level} RISK
                  </span>
                  <span style={{ fontSize: '1.4rem', fontWeight: 800, color: colors.text }}>
                    {result.probability_pct}%
                  </span>
                </div>

                <p style={{ margin: '0.5rem 0 0', fontSize: '0.9rem', color: '#e2e8f0' }}>
                  {result.recommendation}
                </p>

                {result.factors?.length > 0 && (
                  <div style={{ marginTop: '1rem', paddingTop: '1rem',
                    borderTop: '1px solid rgba(255,255,255,0.08)' }}>
                    <p style={{ fontSize: '0.72rem', color: 'rgba(255,255,255,0.4)',
                      margin: '0 0 0.75rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                      Why this score
                    </p>
                    {result.factors.map((f, i) => {
                      const isRisk   = f.direction === 'increases risk'
                      const barW     = Math.round((Math.abs(f.impact) / maxImpact) * 100)
                      const barColor = isRisk ? '#f87171' : '#4ade80'
                      const label    = FEATURE_LABELS[f.feature] || f.feature
                      return (
                        <div key={i} style={{ display: 'flex', alignItems: 'center',
                          gap: 8, marginBottom: 8 }}>
                          <span style={{ fontSize: '0.78rem', color: 'rgba(255,255,255,0.65)',
                            width: 170, flexShrink: 0, whiteSpace: 'nowrap',
                            overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {label} = <span style={{ color: '#e2e8f0' }}>{String(f.value)}</span>
                          </span>
                          <div style={{ flex: 1, height: 6, background: 'rgba(255,255,255,0.08)',
                            borderRadius: 3, overflow: 'hidden' }}>
                            <div style={{ width: `${barW}%`, height: '100%',
                              background: barColor, borderRadius: 3 }} />
                          </div>
                          <span style={{ fontSize: '0.75rem', color: barColor,
                            width: 42, textAlign: 'right', flexShrink: 0 }}>
                            {isRisk ? '+' : '-'}{Math.abs(f.impact).toFixed(3)}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                )}

                {result.auto_filled?.length > 0 && (
                  <p style={{ marginTop: '0.75rem', fontSize: '0.78rem',
                    color: 'rgba(255,255,255,0.4)' }}>
                    Auto-filled: {result.auto_filled.map(f => f.field).join(', ')}
                  </p>
                )}
              </div>
            )
          })()}

        </div>

        {/* Chat panel column */}
        <div style={{
          flex: 1,
          borderLeft: '1px solid rgba(255,255,255,0.06)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          background: 'var(--cream)'
        }}>
          {exifResult && (
            <div style={{
              margin: '0.75rem',
              padding: '0.85rem 1rem',
              borderRadius: 8,
              background: exifResult.has_metadata ? 'rgba(34,197,94,0.08)' : 'rgba(251,191,36,0.08)',
              border: `1px solid ${exifResult.has_metadata ? 'rgba(34,197,94,0.3)' : 'rgba(251,191,36,0.3)'}`,
              fontSize: '0.82rem',
              color: 'var(--text-dark)',
            }}>
              <p style={{ margin: '0 0 0.4rem', fontWeight: 600, color: exifResult.has_metadata ? '#4ade80' : '#fbbf24' }}>
                {exifResult.has_metadata ? '✅ Metadata found' : '⚠️ No metadata found'}
              </p>
              {exifResult.has_metadata ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2, color: 'rgba(255,255,255,0.65)' }}>
                  {exifResult.details?.timestamp && (
                    <span>Taken: {exifResult.details.timestamp_readable}</span>
                  )}
                  {exifResult.details?.device && (
                    <span>Device: {exifResult.details.manufacturer} {exifResult.details.device}</span>
                  )}
                  <span>Location data: {exifResult.details?.gps_present ? 'present' : 'not found'}</span>
                  <span>Editing software: {exifResult.details?.edit_detected ? exifResult.details.software : 'none detected'}</span>
                </div>
              ) : (
                <p style={{ margin: 0, color: 'rgba(255,255,255,0.55)' }}>{exifResult.message}</p>
              )}
            </div>
          )}
          <FraudChatPanel
            ensureThreadId={ensureThreadId}
            onFieldsExtracted={handleFieldsExtracted}
            existingThreadId={urlThreadId || null}
            onExifResult={handleExifResult}
          />
        </div>

      </div>
    </div>
  )
}