import { marked } from 'marked'
import FraudCard, { isFraudResult, parseFraudResult } from './FraudCard'

marked.setOptions({ breaks: true })

function getTime(ts) {
  const d = ts ? new Date(ts) : new Date()
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function AiBubble({ content, streaming }) {
  if (isFraudResult(content)) {
    return <FraudCard data={parseFraudResult(content)} />
  }
  if (streaming && !content) {
    return (
      <div
        className="px-3.5 py-2.5 rounded-2xl"
        style={{
          background: 'var(--white)',
          border: '0.5px solid var(--cream-dark)',
          borderBottomLeftRadius: 4,
          display: 'flex',
          alignItems: 'center',
          gap: 5,
          height: 38,
        }}
      >
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="typing-dot" />
      </div>
    )
  }
  return (
    <div
      className="px-3.5 py-2.5 rounded-2xl text-sm leading-relaxed ai-content"
      style={{
        background: 'var(--white)',
        color: 'var(--text-dark)',
        border: '0.5px solid var(--cream-dark)',
        borderBottomLeftRadius: 4,
        wordBreak: 'break-word',
      }}
      dangerouslySetInnerHTML={{ __html: streaming ? marked.parse(content) + '<span class="inline-block w-0.5 h-3.5 bg-current ml-0.5 animate-pulse" />' : marked.parse(content) }}
    />
  )
}

export default function MessageBubble({ message, username }) {
  const { role, content, timestamp, streaming } = message
  const isUser = role === 'user'

  return (
    <div className={`flex gap-2 items-end anim-msg ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className="w-7 h-7 rounded-lg flex-shrink-0 flex items-center justify-center text-xs font-semibold"
        style={isUser
          ? { background: 'var(--gold-soft)', border: '0.5px solid var(--gold-border)', color: '#7a5c10' }
          : { background: 'var(--navy)', color: 'var(--gold)' }
        }
      >
        {isUser ? (username || 'U').slice(0, 2).toUpperCase() : 'AI'}
      </div>

      {/* Content */}
      <div className={`flex flex-col gap-1 max-w-[72%] ${isUser ? 'items-end' : ''}`}>
        {isUser ? (
          <div
            className="px-3.5 py-2.5 rounded-2xl text-sm leading-relaxed"
            style={{
              background: 'var(--navy)',
              color: 'rgba(255,255,255,0.92)',
              borderBottomRightRadius: 4,
              wordBreak: 'break-word',
            }}
          >
            {content}
          </div>
        ) : (
          <AiBubble content={content} streaming={streaming} />
        )}

        <span className="text-xs px-1" style={{ color: 'var(--text-light)' }}>
          {getTime(timestamp)}
        </span>
      </div>
    </div>
  )
}
