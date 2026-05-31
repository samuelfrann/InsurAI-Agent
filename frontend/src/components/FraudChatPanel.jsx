import { useState, useRef, useEffect } from 'react'
import { marked } from 'marked'
import { authFetch } from '../lib/api'

marked.setOptions({ breaks: true })

const BASE = import.meta.env.VITE_API_URL || ''

const EXTRACTION_HINT = `

IMPORTANT INSTRUCTION — DO NOT SKIP:
After your response, you MUST include a JSON code block containing the extracted fields. Do NOT return a markdown table. Do NOT ask for confirmation. Do NOT say "would you like to proceed". Just extract what you can find and return it as JSON.

Return the JSON like this:
\`\`\`json
{
  "Fault": "Third Party",
  "BasePolicy": "All Perils",
  "VehicleCategory": "Sport",
  "Make": "Mercedes",
  "Sex": "Male",
  "Month": "Jan",
  "MonthClaimed": "Jan",
  "DayOfWeek": "Monday",
  "DayOfWeekClaimed": "Tuesday",
  "PoliceReportFiled": "No",
  "Age": 35,
  "Year": 2024,
  "AgeOfVehicle": 2,
  "AgeOfPolicyHolder": 33,
  "PastNumberOfClaims": 3,
  "NumberOfSuppliments": 4,
  "VehicleValueNGN": 200000000,
  "AccidentArea": "Urban",
  "WitnessPresent": "No",
  "AgentType": "External",
  "Days_Policy_Accident": "none",
  "Days_Policy_Claim": "1 to 7",
  "AddressChange_Claim": "under 6 months",
  "DriverRating": 1
}
\`\`\`

Only include fields you actually found. Use the EXACT field names shown above. Numbers for Age, Year, AgeOfVehicle, AgeOfPolicyHolder, PastNumberOfClaims, NumberOfSuppliments, VehicleValueNGN, DriverRating — not strings.`

const TABLE_FIELD_MAP = {
  'fault': 'Fault',
  'cover type': 'BasePolicy',
  'base policy': 'BasePolicy',
  'vehicle category': 'VehicleCategory',
  'vehicle make': 'Make',
  'make': 'Make',
  'accident month': 'Month',
  'claim month': 'MonthClaimed',
  'accident day': 'DayOfWeek',
  'day of week (accident)': 'DayOfWeek',
  'claim day': 'DayOfWeekClaimed',
  'day of week (claimed)': 'DayOfWeekClaimed',
  'police report filed': 'PoliceReportFiled',
  'gender': 'Sex',
  'sex': 'Sex',
  'claimant age': 'Age',
  'claim year': 'Year',
  'age of vehicle': 'AgeOfVehicle',
  'age of policyholder': 'AgeOfPolicyHolder',
  'age of policy holder': 'AgeOfPolicyHolder',
  'past number of claims': 'PastNumberOfClaims',
  'number of supplements': 'NumberOfSuppliments',
  'no. of repair supplements': 'NumberOfSuppliments',
  'vehicle value (₦)': 'VehicleValueNGN',
  'vehicle value': 'VehicleValueNGN',
  'vehicle price': 'VehicleValueNGN',
  'accident area': 'AccidentArea',
  'witness present': 'WitnessPresent',
  'agent type': 'AgentType',
  'days: policy → accident': 'Days_Policy_Accident',
  'days policy → accident': 'Days_Policy_Accident',
  'days: policy → claim': 'Days_Policy_Claim',
  'days policy → claim': 'Days_Policy_Claim',
  'address change': 'AddressChange_Claim',
  'driver rating': 'DriverRating',
  'marital status': 'MaritalStatus',
}

function extractFields(text) {
  // Try JSON block first
  const jsonMatch = text.match(/```json\s*([\s\S]*?)```/i)
  if (jsonMatch) {
    try { return JSON.parse(jsonMatch[1]) } catch {}
  }

  // Fallback: parse markdown table rows
  const extracted = {}
  const rows = text.match(/\|\s*([^|\n]+?)\s*\|\s*([^|\n]+?)\s*\|/g) || []

  for (const row of rows) {
    const match = row.match(/\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|/)
    if (!match) continue
    const [, rawKey, rawValue] = match
    const key = rawKey.trim().toLowerCase()
    if (key === 'field' || key.includes('---') || key.includes('default')) continue

    const formKey = TABLE_FIELD_MAP[key]
    if (!formKey) continue

    let value = rawValue.replace(/\*\*/g, '').trim()

    // Numeric fields — extract first number
    if (['Age','Year','AgeOfVehicle','AgeOfPolicyHolder',
         'PastNumberOfClaims','NumberOfSuppliments','DriverRating'].includes(formKey)) {
      const n = value.match(/\d+/)
      value = n ? Number(n[0]) : value
    }

    // NGN vehicle value — strip symbols
    if (formKey === 'VehicleValueNGN') {
      const n = value.replace(/[₦,NGN\s]/g, '')
      value = isNaN(n) ? value : Number(n)
    }

    extracted[formKey] = value
  }

  return Object.keys(extracted).length > 0 ? extracted : null
}

function readFileAsBase64(f) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload  = () => resolve(reader.result.split(',')[1])
    reader.onerror = reject
    reader.readAsDataURL(f)
  })
}

export default function FraudChatPanel({ ensureThreadId, onFieldsExtracted, existingThreadId, onExifResult }) {
  const [messages, setMessages]   = useState([])
  const [input, setInput]         = useState('')
  const [file, setFile]           = useState(null)
  const [streaming, setStreaming] = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  const endRef                    = useRef(null)
  const fileRef                   = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    setMessages([])
    setFile(null)
    if (!existingThreadId) return
    authFetch(`/sessions/${existingThreadId}/history`)
      .then(r => r.json())
      .then(history => {
        if (!Array.isArray(history) || history.length === 0) return
        setMessages(history.map(m => {
          const isUser = m.role === 'user'
          let content = m.content
          if (isUser) {
            const marker = 'Claim information provided:\n'
            const idx = content.indexOf(marker)
            if (idx !== -1) {
              content = content.slice(idx + marker.length)
              const hintIdx = content.indexOf('\n\nIMPORTANT INSTRUCTION')
              if (hintIdx !== -1) content = content.slice(0, hintIdx)
            }
          } else {
            content = content.replace(/```json[\s\S]*?```/gi, '').trim()
          }
          return { role: isUser ? 'user' : 'ai', content }
        }))
      })
      .catch(() => {})
  }, [existingThreadId])

  function handleFileChange(e) {
    const f = e.target.files[0]
    if (f) {
      setFile(f)
      const isImage = f.type.startsWith('image/') || /\.(jpg|jpeg|png|gif|webp)$/i.test(f.name)
      if (isImage && onExifResult) {
        readFileAsBase64(f).then(base64 => {
          authFetch('/fraud/check-photo-metadata', {
            method: 'POST',
            body: JSON.stringify({ image_data: base64, file_name: f.name }),
          })
            .then(r => r.json())
            .then(result => onExifResult(result))
            .catch(() => {})
        })
      }
    }
    e.target.value = ''
  }

  const handleSend = async () => {
    if (streaming) return
    if (!input.trim() && !file) return

    const userText = input + (file ? `  📎 ${file.name}` : '')
    setMessages(m => [...m, { role: 'user', content: userText }, { role: 'ai', content: '' }])
    setStreaming(true)

    const query   = `You are an insurance fraud analyst assistant. A claims officer has provided claim information below. Your job is to read it, give a brief helpful response, then ALWAYS end with a JSON code block of the extracted fields so the form can be populated automatically.

Claim information provided:
${input || `(see attached document${file ? `: ${file.name}` : ''})`}
${EXTRACTION_HINT}`
    const rawFile = file
    setInput(''); setFile(null)

    let filePayload = {}
    if (rawFile) {
      const base64 = await readFileAsBase64(rawFile)
      filePayload = { file_data: base64, file_type: rawFile.type, file_name: rawFile.name }
    }

    const threadId = await ensureThreadId()
    const token    = localStorage.getItem('insurai_token')

    try {
      const res = await fetch(BASE + '/fraud/chat/stream', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body:    JSON.stringify({ query, thread_id: threadId, ...filePayload }),
      })

      if (!res.ok) throw new Error(`Server error ${res.status}`)

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer     = ''
      let aiContent  = ''
      let firstToken = false

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (raw === '[DONE]') break
          let evt
          try { evt = JSON.parse(raw) } catch { continue }
          if (evt.token) {
            if (!firstToken) { firstToken = true; aiContent = '' }
            aiContent += evt.token
            setMessages(m => {
              const u = [...m]
              u[u.length - 1] = { role: 'ai', content: aiContent }
              return u
            })
          }
        }
      }

      const fields = extractFields(aiContent)
      if (fields && onFieldsExtracted) onFieldsExtracted(fields)

      const displayContent = aiContent.replace(/```json[\s\S]*?```/gi, '').trim()
      setMessages(m => {
        const u = [...m]
        u[u.length - 1] = { role: 'ai', content: displayContent }
        return u
      })

    } catch (e) {
      setMessages(m => {
        const u = [...m]
        u[u.length - 1] = { role: 'ai', content: `Error: ${e.message}` }
        return u
      })
    } finally {
      setStreaming(false)
    }
  }

  return (
    <div style={panelStyle}>
      <div style={headerStyle} onClick={() => setCollapsed(c => !c)}>
        <span>💬 AI Assistant — upload a claim document to auto-fill the form</span>
        <span>{collapsed ? '▸' : '▾'}</span>
      </div>

      {!collapsed && (
        <>
          <div style={messagesStyle}>
            {messages.length === 0 && (
              <p style={hintStyle}>
                Upload a claim PDF or describe the claim — I'll extract the fields and fill the form below.
              </p>
            )}
            {messages.map((m, i) => (
              m.role === 'user' ? (
                <div key={i} style={userBubble}>{m.content}</div>
              ) : (
                <div
                  key={i}
                  style={aiBubble}
                  dangerouslySetInnerHTML={{
                    __html: m.content
                      ? marked.parse(m.content)
                      : (streaming && i === messages.length - 1 ? '…' : '')
                  }}
                />
              )
            ))}
            <div ref={endRef} />
          </div>

          <div style={inputRowStyle}>
            <input
              ref={fileRef}
              type="file"
              accept=".pdf,image/*"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
            <button onClick={() => fileRef.current?.click()} style={iconBtnStyle} title="Attach file">
              📎
            </button>
            {file && (
              <div style={fileChipStyle}>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 120 }}>
                  📄 {file.name}
                </span>
                <button onClick={() => setFile(null)} style={fileChipXStyle} title="Remove">✕</button>
              </div>
            )}
            <input
              type="text"
              placeholder="Describe the claim or upload a document…"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSend()}
              style={textInputStyle}
              disabled={streaming}
            />
            <button onClick={handleSend} disabled={streaming} style={sendBtnStyle}>
              {streaming ? '…' : 'Send'}
            </button>
          </div>
        </>
      )}
    </div>
  )
}

const panelStyle = {
  display: 'flex', flexDirection: 'column', height: '100%',
  background: 'var(--card-hover)', overflow: 'hidden',
}
const headerStyle = {
  display: 'flex', justifyContent: 'space-between', padding: '0.75rem 1rem',
  fontSize: '0.85rem', color: 'var(--text-secondary)', cursor: 'pointer',
  borderBottom: '1px solid var(--card-border)', userSelect: 'none',
}
const messagesStyle = {
  flex: 1, overflowY: 'auto', padding: '0.75rem 1rem',
  display: 'flex', flexDirection: 'column', gap: '0.5rem',
}
const hintStyle = {
  fontSize: '0.8rem', color: 'var(--text-light)', fontStyle: 'italic', textAlign: 'center',
}
const userBubble = {
  alignSelf: 'flex-end', maxWidth: '80%', padding: '0.5rem 0.75rem',
  background: 'var(--gold)', color: '#0a1628', borderRadius: 8,
  fontSize: '0.85rem', whiteSpace: 'pre-wrap',
}
const aiBubble = {
  alignSelf: 'flex-start', maxWidth: '85%', padding: '0.5rem 0.75rem',
  background: 'var(--card-bg)', color: 'var(--text-dark)', borderRadius: 8,
  fontSize: '0.85rem', whiteSpace: 'pre-wrap', border: '1px solid var(--card-border)',
}
const inputRowStyle = {
  display: 'flex', gap: 6, padding: '0.5rem 0.75rem',
  borderTop: '1px solid var(--card-border)',
}
const iconBtnStyle = {
  background: 'none', border: 'none', color: 'var(--text-secondary)',
  fontSize: '1.1rem', cursor: 'pointer', padding: '0 0.25rem',
}
const textInputStyle = {
  flex: 1, padding: '0.45rem 0.6rem', borderRadius: 6, fontSize: '0.85rem',
  background: 'var(--input-bg)', border: '1px solid var(--input-border)', color: 'var(--text-dark)',
  outline: 'none',
}
const sendBtnStyle = {
  padding: '0.45rem 1rem', borderRadius: 6, background: 'var(--gold)',
  color: '#0a1628', fontWeight: 600, border: 'none', cursor: 'pointer', fontSize: '0.85rem',
}
const fileChipStyle = {
  display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0,
  padding: '0.25rem 0.5rem', borderRadius: 6, fontSize: '0.75rem',
  background: 'var(--input-bg)', border: '1px solid var(--input-border)',
  color: 'var(--text-dark)', maxWidth: 180,
}
const fileChipXStyle = {
  background: 'none', border: 'none', color: 'var(--text-secondary)',
  cursor: 'pointer', padding: 0, fontSize: '0.75rem', lineHeight: 1, flexShrink: 0,
}


