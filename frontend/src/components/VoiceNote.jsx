import { useEffect, useMemo, useRef, useState } from 'react'

// WhatsApp-style voice note. `src` is a TTS audio URL (Phase 4) or a local blob
// URL for a recorded message. Renders a play/pause button, a waveform whose
// bars fill with playback progress, and a duration counter.
export default function VoiceNote({ src, mine }) {
  const audioRef = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [progress, setProgress] = useState(0) // 0..1
  const [duration, setDuration] = useState(0)
  const [current, setCurrent] = useState(0)

  // Deterministic pseudo-waveform from the src string so it's stable.
  const bars = useMemo(() => makeBars(src, 34), [src])

  useEffect(() => {
    const a = audioRef.current
    if (!a) return
    const onTime = () => {
      setCurrent(a.currentTime)
      if (a.duration) setProgress(a.currentTime / a.duration)
    }
    const onMeta = () => setDuration(a.duration || 0)
    const onEnd = () => {
      setPlaying(false)
      setProgress(0)
      setCurrent(0)
    }
    a.addEventListener('timeupdate', onTime)
    a.addEventListener('loadedmetadata', onMeta)
    a.addEventListener('ended', onEnd)
    return () => {
      a.removeEventListener('timeupdate', onTime)
      a.removeEventListener('loadedmetadata', onMeta)
      a.removeEventListener('ended', onEnd)
    }
  }, [])

  const toggle = () => {
    const a = audioRef.current
    if (!a) return
    if (playing) {
      a.pause()
      setPlaying(false)
    } else {
      a.play().then(() => setPlaying(true)).catch(() => {})
    }
  }

  const shown = current || duration
  return (
    <div className={`voicenote ${mine ? 'me' : 'her'}`}>
      <audio ref={audioRef} src={src} preload="metadata" />
      <button className="vn-play" onClick={toggle} aria-label={playing ? 'Pause' : 'Play'}>
        {playing ? <PauseIcon /> : <PlayIcon />}
      </button>
      <div className="vn-wave">
        {bars.map((h, i) => {
          const filled = i / bars.length <= progress
          return (
            <span
              key={i}
              className={`vn-bar ${filled ? 'filled' : ''}`}
              style={{ height: `${h}%` }}
            />
          )
        })}
      </div>
      <span className="vn-time">{fmt(shown)}</span>
    </div>
  )
}

function makeBars(seed, n) {
  let s = 0
  for (let i = 0; i < (seed || '').length; i++) s = (s * 31 + seed.charCodeAt(i)) % 9973
  const out = []
  for (let i = 0; i < n; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff
    out.push(30 + (s % 70)) // 30%..100%
  }
  return out
}

function fmt(sec) {
  if (!sec || Number.isNaN(sec)) return '0:00'
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

function PlayIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
      <path d="M8 5v14l11-7z" />
    </svg>
  )
}
function PauseIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
      <path d="M6 5h4v14H6zM14 5h4v14h-4z" />
    </svg>
  )
}
