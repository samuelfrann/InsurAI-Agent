import { useState, useEffect, useRef, useCallback } from 'react'
import { useOutletContext } from 'react-router-dom'
import { authFetch, getUser, newThreadId, logout } from '../lib/api'
import { isFraudResult, parseFraudResult } from '../components/FraudCard'

import MessageBubble from '../components/MessageBubble'
import ChatInput from '../components/ChatInput'

const BASE = import.meta.env.VITE_API_URL || ''

function mkId() { return Date.now() + Math.random() }

const WELCOME = {
  id: 0,
  role: 'ai',
  content: 'Hello! I am your AI insurance assistant. How can I help you today?',
  timestamp: new Date().toISOString(),
}

export default function Chat() {
  const username = getUser()
  const { currentThreadId, setCurrentThreadId, fetchSessions, darkMode, setDarkMode } = useOutletContext()

  const [messages, setMessages]       = useState([WELCOME])
  const [isSending, setIsSending]     = useState(false)
  const [status, setStatus]           = useState('Online · Ready to assist')
  const [showSugg, setShowSugg]       = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const messagesEndRef = useRef(null)

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function startNewChat() {
    setCurrentThreadId(null)
    setMessages([{ ...WELCOME, id: mkId(), timestamp: new Date().toISOString() }])
    setShowSugg(true)
    setStatus('Online · Ready to assist')
    setSidebarOpen(false)
  }

  const loadSession = useCallback(async (tid, title) => {
    setShowSugg(false)
    setSidebarOpen(false)
    setMessages([{ id: mkId(), role: 'ai', content: `Loading "${title || 'chat'}"…`, timestamp: new Date().toISOString() }])
    try {
      const res  = await authFetch(`/sessions/${tid}/history`)
      const data = await res.json()
      if (!data.length) {
        setMessages([{ id: mkId(), role: 'ai', content: 'No messages in this chat yet.', timestamp: new Date().toISOString() }])
        return
      }
      setMessages(data.map((m, i) => ({
        id: i,
        role: m.role === 'user' ? 'user' : 'ai',
        content: m.content,
        timestamp: m.timestamp,
      })))
    } catch {
      setMessages([{ id: mkId(), role: 'ai', content: 'Could not load chat history.', timestamp: new Date().toISOString() }])
    }
  }, [])

  useEffect(() => {
    if (currentThreadId) {
      loadSession(currentThreadId)
      return
    }
    setMessages([{ ...WELCOME, id: mkId(), timestamp: new Date().toISOString() }])
    setShowSugg(true)
    setStatus('Online · Ready to assist')
  }, [currentThreadId, loadSession])

  async function sendMessageFixed(text, file) {
    if (isSending) return
    setIsSending(true)
    setShowSugg(false)

    const query = text || (file ? 'Please analyse this file.' : '')

    const userMsg = {
      id: mkId(),
      role: 'user',
      content: file && !text ? `📎 ${file.name}` : text + (file ? ` 📎 ${file.name}` : ''),
      timestamp: new Date().toISOString(),
    }
    setMessages(prev => [...prev, userMsg])

    const aiId = mkId()
    setMessages(prev => [...prev, { id: aiId, role: 'ai', content: '', streaming: true, timestamp: new Date().toISOString() }])
    setStatus('Thinking…')

    const activeThreadId = currentThreadId || newThreadId()
    if (!currentThreadId) setCurrentThreadId(activeThreadId)

    const payload = {
      query,
      thread_id: activeThreadId,
      ...(file ? { file_data: file.data, file_type: file.type, file_name: file.name } : {}),
    }

    try {
      const token = localStorage.getItem('insurai_token')
      const res   = await fetch(BASE + '/chat/stream', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(payload),
      })

      if (res.status === 401) { logout(); return }
      if (!res.ok) throw new Error(`Server error ${res.status}`)

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let   buffer  = ''
      let   accumulatedContent = ''
      let   firstToken = false

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

          if (evt.status) {
            setStatus(evt.status + '…')
            if (!firstToken) {
              setMessages(prev => prev.map(m =>
                m.id === aiId ? { ...m, content: evt.status + '…' } : m
              ))
            }
          }

          if (evt.token) {
            if (!firstToken) {
              firstToken = true
              accumulatedContent = ''
              setStatus('Online · Ready to assist')
            }
            accumulatedContent += evt.token
            setMessages(prev => prev.map(m =>
              m.id === aiId ? { ...m, content: accumulatedContent } : m
            ))
          }

          if (evt.error) throw new Error(evt.error)
        }
      }

      setMessages(prev => prev.map(m =>
        m.id === aiId ? { ...m, streaming: false } : m
      ))
      fetchSessions()

    } catch (err) {
      setMessages(prev => prev.map(m =>
        m.id === aiId
          ? { ...m, content: `Error: ${err.message.includes('fetch') ? 'Server unreachable. Is the API running?' : err.message}`, streaming: false }
          : m
      ))
    } finally {
      setStatus('Online · Ready to assist')
      setIsSending(false)
    }
  }

  function triggerFraud() {
    sendMessageFixed('I want to run a fraud detection assessment', null)
  }

  return (
    <div
      className="flex h-screen overflow-hidden"
      style={{ background: 'var(--cream)' }}
    >
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 md:hidden"
          style={{ background: 'rgba(0,0,0,0.5)' }}
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Chat panel */}
      <div className="flex flex-col flex-1 min-w-0 h-full" style={{ background: 'var(--cream)' }}>

        {/* Header */}
        <div
          className="flex items-center gap-3 px-5 py-3.5 flex-shrink-0"
          style={{ background: 'var(--white)', borderBottom: '0.5px solid var(--cream-dark)' }}
        >
          {/* Hamburger (mobile) */}
          <button
            className="md:hidden w-8 h-8 flex items-center justify-center rounded-lg"
            style={{ border: '0.5px solid var(--cream-dark)', background: 'var(--cream)', color: 'var(--text-mid)' }}
            onClick={() => setSidebarOpen(v => !v)}
          >
            <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
              <line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>

          {/* AI avatar */}
          <div
            className="relative w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
            style={{ background: 'var(--navy)' }}
          >
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none">
              <path d="M12 2L4 5.5V11c0 5 3.5 9.5 8 11 4.5-1.5 8-6 8-11V5.5L12 2z" fill="#c9a84c"/>
              <text x="12" y="15.5" textAnchor="middle" fontSize="7" fontWeight="700" fontFamily="sans-serif" fill="#0a1628">AI</text>
            </svg>
            <span
              className="absolute w-2.5 h-2.5 rounded-full"
              style={{ background: '#2ecc71', border: '2px solid var(--white)', bottom: -1, right: -1 }}
            />
          </div>

          <div>
            <p className="text-sm font-medium" style={{ color: 'var(--text-dark)' }}>InsurAI Copilot</p>
            <p className="text-xs" style={{ color: 'var(--text-light)' }}>{status}</p>
          </div>

          <div className="ml-auto flex gap-1.5">
            {/* New chat */}
            <button
              onClick={startNewChat}
              className="w-8 h-8 flex items-center justify-center rounded-lg transition-colors"
              style={{ border: '0.5px solid var(--cream-dark)', background: 'var(--cream)', color: 'var(--text-mid)' }}
              title="New chat"
            >
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" />
                <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" />
              </svg>
            </button>
            {/* Theme */}
            <button
              onClick={() => setDarkMode(v => !v)}
              className="w-8 h-8 flex items-center justify-center rounded-lg transition-colors"
              style={{ border: '0.5px solid var(--cream-dark)', background: 'var(--cream)', color: 'var(--text-mid)' }}
              title="Toggle theme"
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
          </div>
        </div>

        {/* Messages */}
        <div
          className="flex-1 overflow-y-auto px-5 py-5 flex flex-col gap-4"
          style={{ scrollBehavior: 'smooth' }}
        >
          {/* Date divider */}
          <div
            className="flex items-center gap-2.5 text-xs"
            style={{ color: 'var(--text-light)' }}
          >
            <div className="flex-1 h-px" style={{ background: 'var(--cream-dark)' }} />
            Today · {new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })}
            <div className="flex-1 h-px" style={{ background: 'var(--cream-dark)' }} />
          </div>

          {messages.map(msg => (
            <MessageBubble key={msg.id} message={msg} username={username} />
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Suggestions */}
        {showSugg && (
          <div className="flex gap-2 px-5 pb-2 overflow-x-auto" style={{ scrollbarWidth: 'none' }}>
            {['How do I file a claim?', 'Policy coverage info', 'Documents needed'].map(s => (
              <button
                key={s}
                onClick={() => sendMessageFixed(s, null)}
                className="flex-shrink-0 px-3.5 py-2 rounded-full text-xs font-medium transition-all whitespace-nowrap"
                style={{
                  background: 'var(--white)',
                  border: '0.5px solid var(--cream-dark)',
                  color: 'var(--text-dark)',
                }}
              >
                {s}
              </button>
            ))}
          </div>
        )}

        {/* Input */}
        <ChatInput
          onSend={sendMessageFixed}
          disabled={isSending}
        />
      </div>
    </div>
  )
}
