import { useCallback, useEffect, useRef, useState } from 'react'
import Header from './components/Header.jsx'
import ChatArea from './components/ChatArea.jsx'
import InputBar from './components/InputBar.jsx'
import SettingsPanel from './components/SettingsPanel.jsx'
import CallScreen from './components/CallScreen.jsx'
import MoodJournal from './components/MoodJournal.jsx'
import { api, streamChat } from './api.js'
import { sleep, typingDelay, stripEmotionTag } from './utils/format.js'

// One shared session across web + WhatsApp + proactive messages, so every
// surface shows the same continuous history (backend uses "main" too).
function getSessionId() {
  return 'main'
}

const IDLE_TO_LASTSEEN_MS = 25000

export default function App() {
  const sessionId = useRef(getSessionId()).current
  const [persona, setPersona] = useState(null)
  const [messages, setMessages] = useState([])
  const [status, setStatus] = useState({ mode: 'online', lastSeen: null }) // online | typing | lastseen
  const [view, setView] = useState('chat') // chat | call
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [journalOpen, setJournalOpen] = useState(false)
  const [sending, setSending] = useState(false)
  const [sourceFilter, setSourceFilter] = useState('all')
  const idleTimer = useRef(null)

  const reloadHistory = useCallback(() => {
    return api
      .getHistory(sessionId)
      .then((data) => {
        setMessages(
          (data.messages || []).map((m, i) => ({
            id: `h${i}`,
            role: m.role,
            // Clean any emotion tag left in older stored replies.
            content: m.role === 'assistant' ? stripEmotionTag(m.content) : m.content,
            timestamp: m.timestamp,
            audio_url: m.audio_url || null,
            sticker_url: m.sticker_url || null,
            image_url: m.image_url || null,
            status: 'read',
            source: m.source || 'webapp_chat',
          }))
        )
      })
      .catch(() => {})
  }, [sessionId])

  // --- initial load: persona + history ---
  useEffect(() => {
    api.getPersona().then(setPersona).catch(() => {})
    reloadHistory()
  }, [sessionId, reloadHistory])

  // --- status: after activity, go online, then drift to "last seen" ---
  const goOnlineThenIdle = useCallback(() => {
    setStatus({ mode: 'online', lastSeen: null })
    clearTimeout(idleTimer.current)
    idleTimer.current = setTimeout(() => {
      setStatus({ mode: 'lastseen', lastSeen: new Date().toISOString() })
    }, IDLE_TO_LASTSEEN_MS)
  }, [])

  useEffect(() => {
    goOnlineThenIdle()
    return () => clearTimeout(idleTimer.current)
  }, [goOnlineThenIdle])

  const appendMessage = useCallback((msg) => {
    setMessages((prev) => [...prev, msg])
  }, [])

  // --- send a text message ---
  const handleSend = useCallback(
    async (text) => {
      const trimmed = text.trim()
      if (!trimmed || sending) return
      setSending(true)
      clearTimeout(idleTimer.current)

      const now = new Date().toISOString()
      const myId = 'm' + Date.now()
      appendMessage({
        id: myId,
        role: 'user',
        content: trimmed,
        timestamp: now,
        status: 'sent',
      })
      // Quickly promote to delivered/read for the double-tick feel.
      setTimeout(() => updateStatus(myId, 'delivered'), 300)
      setTimeout(() => updateStatus(myId, 'read'), 900)

      setStatus({ mode: 'typing', lastSeen: null })

      let reply = ''
      const startedAt = Date.now()
      await new Promise((resolve) => {
        streamChat(trimmed, sessionId, {
          onToken: (tok) => {
            reply += tok
          },
          // final === false is an intermediate "reaction" bubble (see
          // react-then-deliver in main.py: a DEEP question gets a genuine
          // in-character reaction while the slow cloud-brain answer is still
          // cooking, instead of the chat sitting silent for the round trip).
          // Flush it as its own bubble now and keep listening for the rest.
          onDone: (payload) => {
            if (payload && payload.final === false) {
              const text = stripEmotionTag(reply)
              if (text) {
                appendMessage({
                  id: 'r' + Date.now() + Math.random().toString(36).slice(2),
                  role: 'assistant',
                  content: text,
                  timestamp: new Date().toISOString(),
                  audio_url: null,
                  status: 'read',
                })
              }
              reply = ''
              return
            }
            resolve()
          },
          onError: (err) => {
            reply = reply || `⚠️ ${err}`
            resolve()
          },
          // Song / drawn-image bubbles land mid-stream (after the text reply
          // was already generated, e.g. a "sing"/"draw" tool result) - each
          // gets its own bubble the moment the backend has it ready, instead
          // of only showing up after a page reload.
          onMedia: (payload) => {
            appendMessage({
              id: 'media' + Date.now() + Math.random().toString(36).slice(2),
              role: 'assistant',
              content: '',
              timestamp: new Date().toISOString(),
              audio_url: payload.kind === 'song' ? payload.url : null,
              image_url: payload.kind === 'image' ? payload.url : null,
              status: 'read',
            })
          },
        })
      })

      // Defensively remove any stray emotion tag before showing/speaking.
      reply = stripEmotionTag(reply)

      // Human-like delay: hold "typing…" until it feels proportional to length.
      const elapsed = Date.now() - startedAt
      const wait = typingDelay(reply) - elapsed
      if (wait > 0) await sleep(wait)

      // Text bubble only. Chat is text; her voice lives in Call mode - so we
      // never send a duplicate text + voice-note pair here.
      appendMessage({
        id: 'r' + Date.now(),
        role: 'assistant',
        content: reply,
        timestamp: new Date().toISOString(),
        audio_url: null,
        status: 'read',
      })
      setSending(false)
      goOnlineThenIdle()
    },
    [appendMessage, goOnlineThenIdle, sending, sessionId]
  )

  function updateStatus(id, s) {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, status: s } : m)))
  }

  const visibleMessages = messages.filter((m) => sourceFilter === 'all' || m.source === sourceFilter)

  const handleClearChat = useCallback(async () => {
    await api.clearChat(sessionId).catch(() => {})
    setMessages([])
  }, [sessionId])

  const handlePersonaChange = useCallback((p) => {
    setPersona(p)
  }, [])

  if (view === 'call') {
    return (
      <CallScreen
        persona={persona}
        sessionId={sessionId}
        onEnd={() => {
          setView('chat')
          reloadHistory() // pick up the "call ended" bubble + call turns
        }}
      />
    )
  }

  return (
    <div className="app">
      <Header
        persona={persona}
        status={status}
        onCall={() => setView('call')}
        onMenu={() => setSettingsOpen(true)}
      />
      <ChatArea messages={visibleMessages} persona={persona} typing={status.mode === 'typing'} />
      <InputBar onSend={handleSend} disabled={sending} />
      {settingsOpen && (
        <SettingsPanel
          persona={persona}
          onClose={() => setSettingsOpen(false)}
          onPersonaChange={handlePersonaChange}
          onClearChat={handleClearChat}
          sourceFilter={sourceFilter}
          onSourceFilterChange={setSourceFilter}
          onOpenJournal={() => {
            setSettingsOpen(false)
            setJournalOpen(true)
          }}
        />
      )}
      {journalOpen && <MoodJournal onClose={() => setJournalOpen(false)} />}
    </div>
  )
}
