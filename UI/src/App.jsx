import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import './App.css'

const INGEST_API_URL = import.meta.env.VITE_INGEST_API_URL ?? 'http://127.0.0.1:8000/ingest'
const QUERY_API_URL = import.meta.env.VITE_QUERY_API_URL ?? 'http://127.0.0.1:8000/query'
const THREADS_API_URL = import.meta.env.VITE_THREADS_API_URL ?? 'http://127.0.0.1:8000/threads'
const MESSAGES_API_BASE =
  import.meta.env.VITE_MESSAGES_API_BASE ?? 'http://127.0.0.1:8000/messages'

const createPendingThread = () => ({
  threadId: crypto.randomUUID(),
  title: '',
  file: null,
  ingesting: false,
  error: '',
})

const getAnswerText = (payload) => {
  if (!payload) return 'No response returned from API.'
  return (
    payload.answer ??
    payload.response ??
    payload.result ??
    payload.output ??
    payload.text ??
    payload.message ??
    'Response received, but no text field was found.'
  )
}

function App() {
  const [threads, setThreads] = useState([])
  const [activeThreadId, setActiveThreadId] = useState(null)
  const [pendingThread, setPendingThread] = useState(createPendingThread())
  const [messages, setMessages] = useState([])
  const [messagesLoading, setMessagesLoading] = useState(false)
  const [messagesError, setMessagesError] = useState('')
  const [threadsLoading, setThreadsLoading] = useState(false)
  const [threadsError, setThreadsError] = useState('')
  const [draft, setDraft] = useState('')
  const [sending, setSending] = useState(false)
  const messagesListRef = useRef(null)

  const scrollToBottom = (smooth = true) => {
    const list = messagesListRef.current
    if (list) {
      list.scrollTo({
        top: list.scrollHeight,
        behavior: smooth ? 'smooth' : 'auto',
      })
    }
  }

  const activeThread = useMemo(
    () => threads.find((thread) => thread.threadId === activeThreadId) ?? null,
    [threads, activeThreadId],
  )

  const fetchThreads = useCallback(async (preferredId) => {
    setThreadsLoading(true)
    setThreadsError('')
    try {
      const response = await fetch(THREADS_API_URL)
      const payload = await response.json().catch(() => null)
      if (!response.ok) {
        throw new Error(payload?.detail ?? payload?.message ?? 'Threads API failed.')
      }
      const data = Array.isArray(payload) ? payload : []
      setThreads(data)
      setActiveThreadId((current) => {
        if (preferredId && data.some((thread) => thread.threadId === preferredId)) {
          return preferredId
        }
        if (current && data.some((thread) => thread.threadId === current)) {
          return current
        }
        return data[0]?.threadId ?? null
      })
    } catch (error) {
      setThreadsError(error.message || 'Unable to load threads.')
    } finally {
      setThreadsLoading(false)
    }
  }, [])

  const fetchMessages = useCallback(async (threadId) => {
    if (!threadId) return
    setMessagesLoading(true)
    setMessagesError('')
    try {
      const response = await fetch(`${MESSAGES_API_BASE}/${threadId}`)
      const payload = await response.json().catch(() => null)
      if (!response.ok) {
        throw new Error(payload?.detail ?? payload?.message ?? 'Messages API failed.')
      }
      const data = Array.isArray(payload) ? payload : []
      setMessages(
        data.map((item) => ({
          id: crypto.randomUUID(),
          role: item.role,
          text: item.content,
        })),
      )
    } catch (error) {
      setMessagesError(error.message || 'Unable to load messages.')
    } finally {
      setMessagesLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchThreads()
  }, [fetchThreads])

  useEffect(() => {
    if (!activeThreadId) {
      setMessages([])
      setMessagesError('')
      return
    }
    setMessages([])
    setMessagesError('')
    fetchMessages(activeThreadId)
  }, [activeThreadId, fetchMessages])

  useLayoutEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleCreateChat = () => {
    setActiveThreadId(null)
    setMessages([])
    setPendingThread(createPendingThread())
    setDraft('')
  }

  const handleTitleChange = (value) => {
    setPendingThread((prev) => ({
      ...prev,
      title: value,
      error: '',
    }))
  }

  const handleFileSelect = (event) => {
    const selected = event.target.files?.[0]
    if (!selected) return
    if (selected.type !== 'application/pdf') {
      setPendingThread((prev) => ({
        ...prev,
        error: 'Only PDF files are accepted.',
      }))
      return
    }
    setPendingThread((prev) => ({
      ...prev,
      file: selected,
      error: '',
      title: prev.title?.trim() ? prev.title : selected.name.replace(/\.pdf$/i, '') || '',
    }))
  }

  const handleIngest = async () => {
    if (!pendingThread.file) {
      setPendingThread((prev) => ({ ...prev, error: 'Please choose a PDF first.' }))
      return
    }
    if (!pendingThread.title?.trim()) {
      setPendingThread((prev) => ({ ...prev, error: 'Chat name is required before ingesting.' }))
      return
    }

    setPendingThread((prev) => ({ ...prev, ingesting: true, error: '' }))
    const threadIdToSelect = pendingThread.threadId
    try {
      const formData = new FormData()
      formData.append('file', pendingThread.file)
      formData.append('threadId', pendingThread.threadId)
      formData.append('threadName', pendingThread.title.trim())

      const response = await fetch(INGEST_API_URL, {
        method: 'POST',
        body: formData,
      })

      const payload = await response.json().catch(() => null)
      if (!response.ok) {
        throw new Error(payload?.detail ?? payload?.message ?? 'Ingest API failed.')
      }

      await fetchThreads(threadIdToSelect)
      setPendingThread(createPendingThread())
      setDraft('')
    } catch (error) {
      setPendingThread((prev) => ({
        ...prev,
        ingesting: false,
        error: error.message || 'Unable to ingest PDF.',
      }))
    } finally {
      setPendingThread((prev) => ({ ...prev, ingesting: false }))
    }
  }

  const handleSend = async () => {
    if (!activeThread || !draft.trim() || sending) return

    const text = draft.trim()
    const userMessage = { id: crypto.randomUUID(), role: 'user', text }
    const assistantId = crypto.randomUUID()
    const assistantMessage = { id: assistantId, role: 'assistant', text: '' }
    setDraft('')
    setSending(true)

    setMessages((prev) => [...prev, userMessage, assistantMessage])

    try {
      const body = {
        threadId: activeThread.threadId,
        question: text,
        isStream: true,
      }

      const response = await fetch(QUERY_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (!response.ok) {
        const payload = await response.json().catch(() => null)
        throw new Error(payload?.detail ?? payload?.message ?? 'Query API failed.')
      }

      if (!response.body) {
        const textPayload = await response.text().catch(() => '')
        const finalText = textPayload || 'Response received, but no text was streamed.'
        setMessages((prev) =>
          prev.map((msg) => (msg.id === assistantId ? { ...msg, text: finalText } : msg)),
        )
      } else {
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let done = false
        while (!done) {
          const { value, done: chunkDone } = await reader.read()
          done = chunkDone
          if (value) {
            buffer += decoder.decode(value, { stream: true })
            setMessages((prev) =>
              prev.map((msg) => (msg.id === assistantId ? { ...msg, text: buffer } : msg)),
            )
          }
        }
        const finalChunk = decoder.decode()
        if (finalChunk) {
          buffer += finalChunk
          setMessages((prev) =>
            prev.map((msg) => (msg.id === assistantId ? { ...msg, text: buffer } : msg)),
          )
        }
      }
    } catch (error) {
      const friendlyMessage =
        error instanceof TypeError
          ? 'Cannot reach query API. Check backend is running on http://127.0.0.1:8000 and CORS is enabled.'
          : error.message || 'Unable to get response from API.'

      setMessages((prev) =>
        prev.map((msg) => (msg.id === assistantId ? { ...msg, text: friendlyMessage } : msg)),
      )
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="new-chat-btn" onClick={handleCreateChat}>
          + New Chat
        </button>
        {threadsLoading ? (
          <p className="status-text">Loading threads...</p>
        ) : threadsError ? (
          <p className="error-text">{threadsError}</p>
        ) : null}
        <div className="chat-list">
          {threads.map((thread) => (
            <button
              key={thread.threadId}
              className={`chat-item ${thread.threadId === activeThreadId ? 'active' : ''}`}
              onClick={() => setActiveThreadId(thread.threadId)}
            >
              <span className="chat-item-title">{thread.threadName}</span>
              <span className="chat-item-status">
                {thread.threadId === activeThreadId ? 'Active' : new Date(thread.createdAt).toLocaleString()}
              </span>
            </button>
          ))}
        </div>
      </aside>

      <main className="chat-panel">
        <header className="chat-header">
          <button className="menu-btn" onClick={handleCreateChat}>
            Menu
          </button>
          <h1>RAG Assistant</h1>
        </header>

        {!activeThread ? (
          <section className="ingest-card">
            <h2>Upload a PDF to Start</h2>
            <p>Select one PDF file, ingest it, then begin chatting.</p>
            <label className="chat-name-field">
              <span>Chat name (required)</span>
              <input
                type="text"
                className="chat-name-input"
                value={pendingThread.title}
                onChange={(event) => handleTitleChange(event.target.value)}
                placeholder="Describe this chat"
                required
              />
            </label>
            <label className="file-picker">
              <input type="file" accept="application/pdf,.pdf" onChange={handleFileSelect} />
              <span>{pendingThread.file?.name ?? 'Choose PDF file'}</span>
            </label>
            <button
              className="primary-btn"
              disabled={
                !pendingThread.file || pendingThread.ingesting || !pendingThread.title?.trim()
              }
              onClick={handleIngest}
            >
              {pendingThread.ingesting ? 'Ingesting...' : 'Ingest PDF'}
            </button>
            {pendingThread.error ? <p className="error-text">{pendingThread.error}</p> : null}
          </section>
        ) : (
          <section className="messages-area">
        {messagesLoading ? (
          <p className="status-text">Loading messages...</p>
        ) : (
          <div className="messages-list" ref={messagesListRef}>
                {messages.length === 0 ? (
                  <div className="empty-state">
                    <h2>{activeThread.threadName}</h2>
                    <p>PDF indexed. Ask questions about your document.</p>
                  </div>
                ) : (
                  messages.map((msg) => (
                    <article key={msg.id} className={`message-row ${msg.role}`}>
                      <div className="bubble">{msg.text}</div>
                    </article>
                  ))
                )}
              </div>
            )}

            <div className="composer">
              <textarea
                rows={1}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder={`Ask something about ${activeThread.threadName}...`}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault()
                    handleSend()
                  }
                }}
              />
              <button
                className="primary-btn"
                onClick={handleSend}
                disabled={sending || !draft.trim()}
              >
                {sending ? 'Thinking...' : 'Send'}
              </button>
            </div>
            {messagesError ? <p className="error-text chat-error">{messagesError}</p> : null}
          </section>
        )}
      </main>
    </div>
  )
}

export default App
