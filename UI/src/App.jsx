import { useMemo, useState } from 'react'
import './App.css'

const INGEST_API_URL = import.meta.env.VITE_INGEST_API_URL ?? 'http://127.0.0.1:8000/ingest'
const QUERY_API_URL = import.meta.env.VITE_QUERY_API_URL ?? 'http://127.0.0.1:8000/query'

const createChat = () => {
  const id = crypto.randomUUID()
  return {
    id,
    threadId: id,
    title: '',
    createdAt: new Date().toISOString(),
    file: null,
    ingesting: false,
    isIngested: false,
    ingestMeta: null,
    messages: [],
    error: '',
  }
}

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
  const [chats, setChats] = useState([createChat()])
  const [activeChatId, setActiveChatId] = useState(null)
  const [draft, setDraft] = useState('')
  const [sending, setSending] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const currentChatId = activeChatId ?? chats[0]?.id ?? null
  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === currentChatId) ?? null,
    [chats, currentChatId],
  )

  const updateChat = (chatId, updater) => {
    setChats((prev) => prev.map((chat) => (chat.id === chatId ? updater(chat) : chat)))
  }

  const handleCreateChat = () => {
    const newChat = createChat()
    setChats((prev) => [newChat, ...prev])
    setActiveChatId(newChat.id)
    setDraft('')
  }

  const handleTitleChange = (chatId, nextTitle) => {
    if (!chatId) return
    updateChat(chatId, (chat) => ({
      ...chat,
      title: nextTitle,
      error: '',
    }))
  }

  const handleFileSelect = (event, chatId) => {
    const selected = event.target.files?.[0]
    if (!selected) return
    if (selected.type !== 'application/pdf') {
      updateChat(chatId, (chat) => ({
        ...chat,
        error: 'Only PDF files are accepted.',
      }))
      return
    }
    updateChat(chatId, (chat) => ({
      ...chat,
      file: selected,
      error: '',
      title: chat.title?.trim() ? chat.title : selected.name.replace(/\.pdf$/i, '') || '',
    }))
  }

  const handleIngest = async (chatId) => {
    const chat = chats.find((item) => item.id === chatId)
    if (!chat || !chat.file) {
      updateChat(chatId, (item) => ({ ...item, error: 'Please choose a PDF first.' }))
      return
    }
    if (!chat.title?.trim()) {
      updateChat(chatId, (item) => ({
        ...item,
        error: 'Chat name is required before ingesting.',
      }))
      return
    }

    updateChat(chatId, (item) => ({ ...item, ingesting: true, error: '' }))

    try {
      const formData = new FormData()
      formData.append('file', chat.file)
      formData.append('threadId', chat.threadId)
      formData.append('threadName', chat.title.trim())

      const response = await fetch(INGEST_API_URL, {
        method: 'POST',
        body: formData,
      })

      const payload = await response.json().catch(() => null)
      if (!response.ok) {
        throw new Error(payload?.detail ?? payload?.message ?? 'Ingest API failed.')
      }

      updateChat(chatId, (item) => ({
        ...item,
        ingesting: false,
        isIngested: true,
        ingestMeta: payload,
        error: '',
      }))
    } catch (error) {
      updateChat(chatId, (item) => ({
        ...item,
        ingesting: false,
        error: error.message || 'Unable to ingest PDF.',
      }))
    }
  }

  const handleSend = async () => {
    if (!activeChat || !activeChat.isIngested || !draft.trim() || sending) return

    const text = draft.trim()
    const userMessage = { id: crypto.randomUUID(), role: 'user', text }
    setDraft('')
    setSending(true)

    updateChat(activeChat.id, (chat) => ({
      ...chat,
      messages: [...chat.messages, userMessage],
      error: '',
    }))

    try {
      const body = {
        threadId: activeChat.threadId,
        question: text,
      }

      const response = await fetch(QUERY_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      const payload = await response.json().catch(() => null)
      if (!response.ok) {
        throw new Error(payload?.detail ?? payload?.message ?? 'Query API failed.')
      }

      const assistantMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        text: getAnswerText(payload),
      }

      updateChat(activeChat.id, (chat) => ({
        ...chat,
        messages: [...chat.messages, assistantMessage],
      }))
    } catch (error) {
      const friendlyMessage =
        error instanceof TypeError
          ? 'Cannot reach query API. Check backend is running on http://127.0.0.1:8000 and CORS is enabled.'
          : error.message || 'Unable to get response from API.'
      updateChat(activeChat.id, (chat) => ({
        ...chat,
        error: friendlyMessage,
      }))
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <button className="new-chat-btn" onClick={handleCreateChat}>
          + New Chat
        </button>
        <div className="chat-list">
          {chats.map((chat) => (
            <button
              key={chat.id}
              className={`chat-item ${chat.id === currentChatId ? 'active' : ''}`}
              onClick={() => setActiveChatId(chat.id)}
            >
              <span className="chat-item-title">{chat.title || 'New Chat'}</span>
              <span className="chat-item-status">
                {chat.isIngested ? 'Ready' : chat.file ? 'Upload pending' : 'No PDF'}
              </span>
            </button>
          ))}
        </div>
      </aside>

      <main className="chat-panel">
        <header className="chat-header">
          <button className="menu-btn" onClick={() => setSidebarOpen((prev) => !prev)}>
            {sidebarOpen ? 'Hide' : 'Menu'}
          </button>
          <h1>RAG Assistant</h1>
        </header>

        {!activeChat?.isIngested ? (
        <section className="ingest-card">
          <h2>Upload a PDF to Start</h2>
          <p>Select one PDF file, ingest it, then begin chatting.</p>
          <label className="chat-name-field">
            <span>Chat name (required)</span>
            <input
              type="text"
              className="chat-name-input"
              value={activeChat?.title ?? ''}
              onChange={(event) => handleTitleChange(activeChat?.id, event.target.value)}
              placeholder="Describe this chat"
              required
            />
          </label>
          <label className="file-picker">
            <input
              type="file"
              accept="application/pdf,.pdf"
              onChange={(event) => handleFileSelect(event, activeChat?.id)}
            />
            <span>{activeChat?.file?.name ?? 'Choose PDF file'}</span>
          </label>
          <button
            className="primary-btn"
            disabled={
              !activeChat?.file || activeChat?.ingesting || !activeChat?.title?.trim()
            }
            onClick={() => handleIngest(activeChat.id)}
          >
              {activeChat?.ingesting ? 'Ingesting...' : 'Ingest PDF'}
            </button>
            {activeChat?.error ? <p className="error-text">{activeChat.error}</p> : null}
          </section>
        ) : (
          <section className="messages-area">
            <div className="messages-list">
              {activeChat.messages.length === 0 ? (
                <div className="empty-state">
                  <h2>{activeChat.title || 'New Chat'}</h2>
                  <p>PDF indexed. Ask questions about your document.</p>
                </div>
              ) : (
                activeChat.messages.map((msg) => (
                  <article key={msg.id} className={`message-row ${msg.role}`}>
                    <div className="bubble">{msg.text}</div>
                  </article>
                ))
              )}
            </div>

            <div className="composer">
              <textarea
                rows={1}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder="Ask something about the uploaded PDF..."
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault()
                    handleSend()
                  }
                }}
              />
              <button className="primary-btn" onClick={handleSend} disabled={sending || !draft.trim()}>
                {sending ? 'Sending...' : 'Send'}
              </button>
            </div>
            {activeChat.error ? <p className="error-text chat-error">{activeChat.error}</p> : null}
          </section>
        )}
      </main>
    </div>
  )
}

export default App
