import VoiceNote from './VoiceNote.jsx'
import { timeHHMM } from '../utils/format.js'

// Blue double-tick when read, grey otherwise. Single tick when just "sent".
function Ticks({ status }) {
  const read = status === 'read'
  const single = status === 'sent'
  return (
    <span className={`ticks ${read ? 'read' : ''}`}>
      <svg viewBox="0 0 18 14" width="16" height="12" fill="currentColor">
        <path d="M6.4 11.2 2.6 7.4l-1 1L6.4 13.2 15 4.6l-1-1z" />
        {!single && <path d="M10.4 11.2 6.6 7.4l-1 1 4.8 4.8L19 4.6l-1-1z" opacity="0.95" />}
      </svg>
    </span>
  )
}

// webapp_chat is the default channel: no marker at all. Other sources get a
// tiny colored dot next to the timestamp + a faint edge accent — visible if
// you look, invisible if you don't.
const SOURCE_META = {
  webapp_call: { label: 'Voice call', color: '#a78bfa' },
  whatsapp: { label: 'WhatsApp', color: '#22c55e' },
  tablet: { label: 'Tablet', color: '#60a5fa' },
  iot: { label: 'Smart device', color: '#fb923c' },
}

export default function MessageBubble({ message }) {
  const mine = message.role === 'user'
  const sourceMeta = SOURCE_META[message.source] // undefined for webapp_chat

  // Stickers render WhatsApp-style: bare image, no bubble background.
  if (message.sticker_url) {
    return (
      <div className={`row ${mine ? 'me' : 'her'}`}>
        <div className="sticker-msg">
          <img src={message.sticker_url} alt="sticker" draggable="false" />
          <span className="time sticker-time">
            {sourceMeta && <span className="src-dot" style={{ background: sourceMeta.color }} title={sourceMeta.label} />}
            {timeHHMM(message.timestamp)}
          </span>
        </div>
      </div>
    )
  }

  return (
    <div className={`row ${mine ? 'me' : 'her'}`}>
      <div
        className={`bubble ${mine ? 'me' : 'her'} ${sourceMeta ? 'tinted' : ''}`}
        style={sourceMeta ? { [mine ? 'borderRight' : 'borderLeft']: `2px solid ${sourceMeta.color}55` } : undefined}
        title={sourceMeta ? `via ${sourceMeta.label}` : undefined}
      >
        {message.audio_url ? (
          <VoiceNote src={message.audio_url} mine={mine} />
        ) : (
          <span className="bubble-text">{message.content}</span>
        )}
        <span className="meta">
          {sourceMeta && <span className="src-dot" style={{ background: sourceMeta.color }} title={sourceMeta.label} />}
          <span className="time">{timeHHMM(message.timestamp)}</span>
          {mine && <Ticks status={message.status} />}
        </span>
      </div>
    </div>
  )
}
