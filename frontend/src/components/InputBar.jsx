import { useRef, useState } from 'react'
import EmojiPicker from './EmojiPicker.jsx'
import { api } from '../api.js'

export default function InputBar({ onSend, disabled }) {
  const [text, setText] = useState('')
  const [emojiOpen, setEmojiOpen] = useState(false)
  const [recording, setRecording] = useState(false)
  const [transcribing, setTranscribing] = useState(false)
  const [notice, setNotice] = useState('')
  const inputRef = useRef(null)
  const recorderRef = useRef(null)
  const chunksRef = useRef([])

  const hasText = text.trim().length > 0

  const submit = () => {
    if (!hasText || disabled) return
    onSend(text)
    setText('')
    setEmojiOpen(false)
    inputRef.current?.focus()
  }

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const insertEmoji = (emoji) => {
    setText((t) => t + emoji)
    inputRef.current?.focus()
  }

  const flash = (msg) => {
    setNotice(msg)
    setTimeout(() => setNotice(''), 2600)
  }

  // --- hold-to-record: press mic, speak, release -> /stt -> send as message ---
  const startRecording = async () => {
    if (recording || disabled) return
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const rec = new MediaRecorder(stream)
      chunksRef.current = []
      rec.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data)
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        if (blob.size < 800) return // too short - ignore accidental taps
        setTranscribing(true)
        try {
          const res = await api.stt(blob)
          const spoken = (res?.text || '').trim()
          if (spoken) onSend(spoken) // send the transcribed message straight away
          else flash("Didn't catch that - try again.")
        } catch (err) {
          flash(
            err.message === 'STT_NOT_AVAILABLE'
              ? 'Speech-to-text is unavailable (install faster-whisper).'
              : 'Transcription failed.'
          )
        } finally {
          setTranscribing(false)
        }
      }
      rec.start()
      recorderRef.current = rec
      setRecording(true)
    } catch {
      flash('Mic permission denied.')
    }
  }

  const stopRecording = () => {
    if (!recording) return
    recorderRef.current?.stop()
    setRecording(false)
  }

  return (
    <div className="inputbar-wrap">
      {notice && <div className="notice">{notice}</div>}
      {emojiOpen && <EmojiPicker onPick={insertEmoji} />}
      <div className="inputbar">
        <button
          className="icon-btn ghost"
          aria-label="Emoji"
          onClick={() => setEmojiOpen((o) => !o)}
        >
          <EmojiIcon />
        </button>
        <input
          ref={inputRef}
          className="text-input"
          type="text"
          placeholder={
            recording ? 'Recording… release to send' : transcribing ? 'Transcribing…' : 'Message'
          }
          value={text}
          disabled={recording || transcribing}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
        />
        {hasText ? (
          <button className="send-btn" aria-label="Send" onClick={submit} disabled={disabled}>
            <SendIcon />
          </button>
        ) : (
          <button
            className={`send-btn ${recording ? 'recording' : ''}`}
            aria-label="Hold to record a voice message"
            title="Hold to talk"
            disabled={transcribing || disabled}
            onMouseDown={startRecording}
            onMouseUp={stopRecording}
            onMouseLeave={stopRecording}
            onTouchStart={(e) => {
              e.preventDefault()
              startRecording()
            }}
            onTouchEnd={(e) => {
              e.preventDefault()
              stopRecording()
            }}
          >
            {recording ? <StopIcon /> : <MicIcon />}
          </button>
        )}
      </div>
    </div>
  )
}

function EmojiIcon() {
  return (
    <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
      <path d="M12 2a10 10 0 100 20 10 10 0 000-20zm0 18a8 8 0 110-16 8 8 0 010 16zM8.5 11a1.5 1.5 0 100-3 1.5 1.5 0 000 3zm7 0a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM12 17.5c2.3 0 4.2-1.5 4.9-3.5H7.1c.7 2 2.6 3.5 4.9 3.5z" />
    </svg>
  )
}
function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="currentColor">
      <path d="M2 21l21-9L2 3v7l15 2-15 2z" />
    </svg>
  )
}
function MicIcon() {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="currentColor">
      <path d="M12 15a3 3 0 003-3V6a3 3 0 00-6 0v6a3 3 0 003 3zm5-3a5 5 0 01-10 0H5a7 7 0 006 6.9V21h2v-2.1A7 7 0 0019 12h-2z" />
    </svg>
  )
}
function StopIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  )
}
