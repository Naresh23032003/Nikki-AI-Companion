import { timeHHMM } from '../utils/format.js'
import Avatar from './Avatar.jsx'

function statusText(status) {
  if (status.mode === 'typing') return 'typing…'
  if (status.mode === 'online') return 'online'
  if (status.mode === 'lastseen') return `last seen today at ${timeHHMM(status.lastSeen)}`
  return ''
}

export default function Header({ persona, status, onCall, onMenu }) {
  return (
    <header className="header">
      <Avatar persona={persona} size={40} />
      <div className="header-text">
        <div className="header-name">{persona?.name || 'Companion'}</div>
        <div className={`header-status ${status.mode}`}>{statusText(status)}</div>
      </div>
      <div className="header-actions">
        <button className="icon-btn" aria-label="Voice call" onClick={onCall}>
          <PhoneIcon />
        </button>
        <button className="icon-btn" aria-label="Menu" onClick={onMenu}>
          <DotsIcon />
        </button>
      </div>
    </header>
  )
}

function PhoneIcon() {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="currentColor">
      <path d="M6.6 10.8c1.4 2.8 3.8 5.1 6.6 6.6l2.2-2.2c.3-.3.7-.4 1-.2 1.1.4 2.3.6 3.6.6.6 0 1 .4 1 1V20c0 .6-.4 1-1 1C10.5 21 3 13.5 3 4c0-.6.4-1 1-1h3.5c.6 0 1 .4 1 1 0 1.2.2 2.4.6 3.6.1.4 0 .8-.3 1l-2.2 2.2z" />
    </svg>
  )
}

function DotsIcon() {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="currentColor">
      <circle cx="12" cy="5" r="2" />
      <circle cx="12" cy="12" r="2" />
      <circle cx="12" cy="19" r="2" />
    </svg>
  )
}
