import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble.jsx'
import { dateLabel, dayKey } from '../utils/format.js'

export default function ChatArea({ messages, persona, typing }) {
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, typing])

  // Group into date sections for WhatsApp-style separators.
  let lastDay = null

  return (
    <main className="chat-area">
      <div className="chat-scroll">
        {messages.length === 0 && (
          <div className="empty-hint">
            <span>Say hi to {persona?.name || 'your companion'} 👋</span>
          </div>
        )}

        {messages.map((m) => {
          const key = dayKey(m.timestamp)
          const showSep = key !== lastDay
          lastDay = key
          return (
            <div key={m.id}>
              {showSep && (
                <div className="date-sep">
                  <span>{dateLabel(m.timestamp)}</span>
                </div>
              )}
              {m.role === 'event' ? (
                <div className="event-pill">
                  <span>📞 {m.content}</span>
                </div>
              ) : (
                <MessageBubble message={m} />
              )}
            </div>
          )
        })}

        {typing && (
          <div className="row her">
            <div className="bubble her typing-bubble">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
            </div>
          </div>
        )}

        <div ref={endRef} />
      </div>
    </main>
  )
}
