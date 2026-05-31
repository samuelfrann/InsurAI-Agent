import { useNavigate } from 'react-router-dom'

export function isFraudResult(text) {
  return /fraud\s*probability/i.test(text) && /risk\s*level/i.test(text)
}

export function parseFraudResult(text) {
  const d = { probability: null, riskLevel: null, classification: null, recommendation: null, autoFilled: [] }
  const clean = text.replace(/\*+/g, '')

  let m = clean.match(/fraud\s*probability[^\d-]*([\d.]+)\s*%?/i)
  if (m) d.probability = parseFloat(m[1])

  m = clean.match(/risk\s*level\s*:?\s*([A-Z][A-Z\s-]{1,20}?)(?=\s*\n|$|\.)/i)
  if (m) d.riskLevel = m[1].trim()

  m = clean.match(/classification\s*:?\s*([^\n.]+)/i)
  if (m) d.classification = m[1].trim()

  m = clean.match(/recommendation\s*:?\s*([^\n]+)/i)
  if (m) d.recommendation = m[1].trim()

  const autoBlock = clean.match(/auto[- ]?filled[\s\S]*?(?=\n\s*\n|$)/i)
  if (autoBlock) {
    const pairs = autoBlock[0].matchAll(/[-•]?\s*([A-Z][A-Za-z]+)\s*[:=]\s*([^\n,]+)/g)
    for (const p of pairs) d.autoFilled.push({ f: p[1].trim(), v: p[2].trim() })
  }

  return d
}

function riskClass(level) {
  if (!level) return 'low'
  const l = level.toLowerCase()
  if (l.includes('high')) return 'high'
  if (l.includes('medium')) return 'medium'
  return 'low'
}

function getTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export default function FraudCard({ data }) {
  const navigate = useNavigate()
  const rc   = riskClass(data.riskLevel)
  const prob = data.probability !== null ? data.probability.toFixed(1) + '%' : '—'
  const cls  = data.classification || (rc === 'high' ? 'Suspicious' : 'Normal')

  const scoreColor = rc === 'high' ? 'var(--danger)' : rc === 'medium' ? 'var(--warning)' : 'var(--success)'
  const recBg      = rc === 'high' ? 'var(--danger-bg)' : rc === 'medium' ? 'var(--warning-bg)' : 'var(--success-bg)'
  const recBorder  = rc === 'high' ? 'rgba(229,62,62,0.2)' : rc === 'medium' ? 'rgba(214,158,46,0.2)' : 'rgba(47,133,90,0.2)'
  const recColor   = rc === 'high' ? 'var(--danger)' : rc === 'medium' ? 'var(--warning)' : 'var(--success)'
  const recIcon    = rc === 'high' ? '⚠️' : rc === 'medium' ? '⚡' : '✅'

  return (
    <div
      className="rounded-2xl overflow-hidden max-w-sm"
      style={{
        background: 'var(--white)',
        border: '0.5px solid var(--cream-dark)',
        borderBottomLeftRadius: 4,
      }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-3 px-4 py-3"
        style={{ background: 'var(--navy)' }}
      >
        <div
          className="w-7 h-7 rounded-lg flex items-center justify-center text-sm flex-shrink-0"
          style={{ background: 'var(--gold-soft)', border: '0.5px solid var(--gold-border)' }}
        >
          🔍
        </div>
        <div>
          <p className="text-sm font-medium text-white">Fraud assessment result</p>
          <p className="text-xs" style={{ color: 'rgba(255,255,255,0.35)' }}>
            Processed · {getTime()}
          </p>
        </div>
      </div>

      {/* Body */}
      <div className="p-4">
        {/* Score grid */}
        <div className="grid grid-cols-3 gap-2 mb-3">
          {[
            { label: 'Probability', value: prob },
            { label: 'Risk', value: data.riskLevel || '—' },
            { label: 'Status', value: cls },
          ].map(({ label, value }) => (
            <div
              key={label}
              className="rounded-lg p-2 text-center"
              style={{ background: 'var(--cream)' }}
            >
              <p className="text-xs mb-1" style={{ color: 'var(--text-light)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                {label}
              </p>
              <p className="text-sm font-semibold" style={{ color: scoreColor }}>{value}</p>
            </div>
          ))}
        </div>

        <hr style={{ borderColor: 'var(--cream-dark)', marginBottom: 9 }} />

        {/* Recommendation */}
        {data.recommendation && (
          <div
            className="flex gap-2 items-start rounded-lg px-3 py-2 text-xs"
            style={{ background: recBg, border: `0.5px solid ${recBorder}`, color: recColor }}
          >
            {recIcon} {data.recommendation}
          </div>
        )}

        {/* Auto-fill note */}
        {data.autoFilled.length > 0 && (
          <div
            className="rounded-lg px-3 py-2 text-xs mt-2"
            style={{ background: 'var(--warning-bg)', border: '0.5px solid rgba(214,158,46,0.2)' }}
          >
            ⚠️ <strong>{data.autoFilled.length} auto-filled:</strong>{' '}
            {data.autoFilled.map(x => `${x.f}: ${x.v}`).join(' · ')}
          </div>
        )}

        <p className="text-xs mt-2" style={{ color: 'var(--text-light)', lineHeight: 1.5 }}>
          Model precision ~21% · Triage signal only — final decision requires human review.
        </p>

        {/* Actions */}
        <div className="flex gap-2 mt-3 pt-3" style={{ borderTop: '0.5px solid var(--cream-dark)' }}>
          <button
            onClick={() => navigate('/dashboard')}
            className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg text-xs font-medium transition-all"
            style={{ background: 'rgba(201,168,76,0.1)', border: '0.5px solid rgba(201,168,76,0.3)', color: 'var(--gold)' }}
          >
            📊 Dashboard
          </button>
          <button
            onClick={() => navigate('/analytics')}
            className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg text-xs font-medium transition-all"
            style={{ background: 'rgba(201,168,76,0.1)', border: '0.5px solid rgba(201,168,76,0.3)', color: 'var(--gold)' }}
          >
            📈 Analytics
          </button>
        </div>
      </div>
    </div>
  )
}
