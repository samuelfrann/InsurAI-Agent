import { useRef, useState, useEffect } from 'react'

export default function ChatInput({ onSend, disabled, onFraudClick }) {
  const [text, setText]           = useState('')
  const [file, setFile]           = useState(null)
  const textareaRef               = useRef(null)
  const fileInputRef              = useRef(null)

  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 140) + 'px'
  }, [text])

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  function submit() {
    const trimmed = text.trim()
    if (!trimmed && !file) return
    onSend(trimmed, file)
    setText('')
    setFile(null)
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  function handleFileChange(e) {
    const f = e.target.files[0]
    if (!f) return
    const reader = new FileReader()
    reader.onload = () => {
      const base64 = reader.result.split(',')[1]
      setFile({ data: base64, type: f.type, name: f.name })
    }
    reader.readAsDataURL(f)
    e.target.value = ''
  }

  return (
    <div
      className="px-5 pb-5 pt-3 flex-shrink-0"
      style={{ background: 'var(--white)', borderTop: '0.5px solid var(--cream-dark)' }}
    >
      {/* File preview */}
      {file && (
        <div
          className="flex items-center gap-2 rounded-lg px-3 py-2 mb-2 text-xs"
          style={{ background: 'var(--gold-soft)', border: '0.5px solid var(--gold-border)', color: 'var(--text-dark)' }}
        >
          <span className="flex-1 truncate">📎 {file.name}</span>
          <button
            onClick={() => setFile(null)}
            style={{ color: 'var(--text-light)', lineHeight: 1 }}
            className="text-sm hover:text-red-500 transition-colors"
          >
            ✕
          </button>
        </div>
      )}

      {/* Input row */}
      <div
        className="flex items-end gap-2 rounded-xl px-4 py-2.5 transition-all"
        style={{
          background: 'var(--cream)',
          border: '1.5px solid var(--cream-dark)',
        }}
        onFocus={e => { e.currentTarget.style.borderColor = 'var(--gold)'; e.currentTarget.style.boxShadow = '0 0 0 3px rgba(201,168,76,0.1)' }}
        onBlur={e => { e.currentTarget.style.borderColor = 'var(--cream-dark)'; e.currentTarget.style.boxShadow = 'none' }}
      >
        <textarea
          ref={textareaRef}
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKey}
          rows={1}
          disabled={disabled}
          placeholder="Ask about a policy, or paste claim details…"
          className="flex-1 border-none bg-transparent outline-none resize-none text-sm leading-relaxed"
          style={{
            color: 'var(--text-dark)',
            minHeight: 22,
            maxHeight: 140,
            fontFamily: 'DM Sans, sans-serif',
          }}
        />

        <div className="flex items-center gap-1 flex-shrink-0">
          {/* Attach */}
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".pdf,.png,.jpg,.jpeg,.gif,.webp,.txt,.csv"
            onChange={handleFileChange}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
            className="w-7 h-7 flex items-center justify-center rounded-lg transition-colors"
            style={{ color: 'var(--text-light)' }}
            title="Attach file"
          >
            <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
            </svg>
          </button>

          {/* Send */}
          <button
            onClick={submit}
            disabled={disabled || (!text.trim() && !file)}
            className="w-9 h-9 flex items-center justify-center rounded-xl transition-all"
            style={{
              background: 'var(--navy)',
              opacity: (disabled || (!text.trim() && !file)) ? 0.5 : 1,
              cursor: (disabled || (!text.trim() && !file)) ? 'not-allowed' : 'pointer',
            }}
          >
            <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="var(--gold)" strokeWidth="2.2" strokeLinecap="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
      </div>

      <div className="flex justify-between mt-2 px-1" style={{ color: 'var(--text-light)', fontSize: 11 }}>
        <span>Enter to send · Shift+Enter for new line</span>
        <span>{text.length} / 500</span>
      </div>
    </div>
  )
}
