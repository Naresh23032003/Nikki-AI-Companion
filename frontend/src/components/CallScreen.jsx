import { useEffect, useRef } from 'react'
import AnimatedAvatar from './AnimatedAvatar.jsx'
import { useCall } from '../call/useCall.js'
import { makeRingtone } from '../call/ringtone.js'

// Full-screen, WhatsApp-video-call-style Call mode. Hands-free: VAD drives the
// turn-taking; the avatar animates and lip-syncs to her streamed speech.
export default function CallScreen({ persona, sessionId, onEnd }) {
  const call = useCall(sessionId, { onEnded: () => onEnd() })
  const ringtoneRef = useRef(null)

  // Ringtone while connecting/ringing (before she picks up).
  useEffect(() => {
    if (!ringtoneRef.current) ringtoneRef.current = makeRingtone()
    const rt = ringtoneRef.current
    if (call.phase === 'connecting' || call.phase === 'ringing') rt.start()
    else rt.stop()
    return () => rt.stop()
  }, [call.phase])

  const statusText = {
    connecting: 'connecting…',
    ringing: 'ringing…',
    idle: formatTimer(call.elapsed),
    listening: 'listening…',
    thinking: 'thinking…',
    speaking: formatTimer(call.elapsed),
    ended: 'call ended',
  }[call.phase]

  const connecting = call.phase === 'connecting' || call.phase === 'ringing'

  return (
    <div className={`call-screen video ${call.phase}`}>
      <div className={`call-avatar-stage ${connecting ? 'ringing' : ''}`}>
        <AnimatedAvatar
          persona={persona}
          emotion={call.emotion}
          speaking={call.phase === 'speaking'}
          getViseme={call.currentViseme}
          getAmplitude={call.amplitude}
        />
        {call.phase === 'listening' && (
          <div className="listening-pill">
            <span className="ldot" />
            <span className="ldot" />
            <span className="ldot" />
            listening
          </div>
        )}
      </div>

      <div className="call-overlay-top">
        <div className="call-name">{persona?.name || 'Companion'}</div>
        <div className="call-sub">{statusText}</div>
        {call.error && <div className="notice call-notice">{call.error}</div>}
      </div>

      {call.caption && <div className="call-caption">{call.caption}</div>}

      <div className="call-actions">
        <button
          className={`call-btn muted ${call.muted ? 'on' : ''}`}
          aria-label={call.muted ? 'Unmute' : 'Mute'}
          onClick={call.toggleMute}
        >
          {call.muted ? <MicOffIcon /> : <MicIcon />}
        </button>
        <button className="call-btn end" aria-label="End call" onClick={call.endCall}>
          <PhoneDownIcon />
        </button>
      </div>
      <div className="call-hint">
        {call.micReady ? 'Just talk — tap the mic to mute' : 'Allow microphone to talk'}
      </div>
    </div>
  )
}

function formatTimer(sec) {
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return `${m}:${String(s).padStart(2, '0')}`
}

function MicIcon() {
  return (
    <svg viewBox="0 0 24 24" width="26" height="26" fill="currentColor">
      <path d="M12 15a3 3 0 003-3V6a3 3 0 00-6 0v6a3 3 0 003 3zm5-3a5 5 0 01-10 0H5a7 7 0 006 6.9V21h2v-2.1A7 7 0 0019 12h-2z" />
    </svg>
  )
}
function MicOffIcon() {
  return (
    <svg viewBox="0 0 24 24" width="26" height="26" fill="currentColor">
      <path d="M15 10.6V6a3 3 0 00-5.9-.8L15 10.6zM4.3 3 3 4.3l6 6V12a3 3 0 004.6 2.5l1.4 1.4A5 5 0 017 12H5a7 7 0 006 6.9V21h2v-2.1c.9-.13 1.7-.42 2.4-.85L19.7 21 21 19.7 4.3 3z" />
    </svg>
  )
}
function PhoneDownIcon() {
  return (
    <svg viewBox="0 0 24 24" width="28" height="28" fill="currentColor">
      <path d="M12 9c-1.6 0-3.2.25-4.7.7v3.1c0 .4-.24.75-.6.9-1 .45-1.9 1.05-2.7 1.75-.18.18-.43.28-.7.28-.28 0-.53-.1-.7-.29L.3 13.2a.98.98 0 01-.3-.7c0-.28.1-.53.29-.7C3.34 8.78 7.46 7 12 7s8.66 1.78 11.71 4.8c.19.17.29.42.29.7 0 .28-.1.53-.29.71l-2.48 2.44c-.17.19-.42.29-.7.29-.27 0-.52-.1-.7-.28-.8-.7-1.7-1.3-2.7-1.75a.99.99 0 01-.6-.9v-3.1C15.2 9.25 13.6 9 12 9z" />
    </svg>
  )
}
