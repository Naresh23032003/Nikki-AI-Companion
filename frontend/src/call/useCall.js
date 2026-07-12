import { useCallback, useEffect, useRef, useState } from 'react'
import { callSocketUrl } from '../api.js'
import { AudioEngine } from './audioEngine.js'
import { float32ToWav, bytesToBase64, concatFloat32 } from './wav.js'

// Orchestrates a hands-free voice call:
//   VAD (mic) -> /stt -> LLM -> streamed TTS -> Web Audio playback
// with barge-in (interrupt her by speaking) and a backchannel grace window
// (pause mid-thought and it waits for you to continue).
//
// Phases: connecting -> ringing -> (she greets) -> listening/speaking/thinking

const GRACE_MS = 1300 // wait this long after you stop for you to continue

export function useCall(sessionId, { onEnded } = {}) {
  const [phase, setPhase] = useState('connecting')
  const [caption, setCaption] = useState('')
  const [emotion, setEmotion] = useState('neutral')
  const [muted, setMuted] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [micReady, setMicReady] = useState(false)
  const [error, setError] = useState('')

  const wsRef = useRef(null)
  const engineRef = useRef(null)
  const vadRef = useRef(null)
  const mutedRef = useRef(false)
  const speakingRef = useRef(false) // is SHE speaking right now
  const pendingRef = useRef([]) // buffered speech fragments (backchannel)
  const graceTimer = useRef(null)
  const startedAtRef = useRef(Date.now())

  const setPhaseSafe = useCallback((p) => setPhase((prev) => (prev === 'ended' ? prev : p)), [])

  // ---- send a (possibly merged) utterance to the server ----
  const flushPending = useCallback(() => {
    const frags = pendingRef.current
    pendingRef.current = []
    if (!frags.length) {
      setPhaseSafe('idle') // nothing captured -> never stay stuck on "listening"
      return
    }
    const merged = concatFloat32(frags)
    if (merged.length < 4800) {
      // < 0.3s @16k: a blip/noise, not speech. Reset the UI instead of hanging.
      setPhaseSafe('idle')
      return
    }
    const b64 = bytesToBase64(float32ToWav(merged, 16000))
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      setPhaseSafe('thinking')
      ws.send(JSON.stringify({ type: 'user_audio', audio: b64, session_id: sessionId }))
    } else {
      setPhaseSafe('idle')
    }
  }, [sessionId, setPhaseSafe])

  // ---- VAD callbacks ----
  const listenWatchdog = useRef(null)

  const onSpeechStart = useCallback(() => {
    if (mutedRef.current) return
    clearTimeout(graceTimer.current)
    // Barge-in: if she's talking, cut her off instantly.
    if (speakingRef.current) {
      engineRef.current?.flush()
      speakingRef.current = false
      wsRef.current?.send(JSON.stringify({ type: 'cancel' }))
    }
    setPhaseSafe('listening')
    // Watchdog: if VAD never fires speech-end (noisy room, speaker bleed),
    // force-flush whatever we have after 12s so "listening" can't hang forever.
    clearTimeout(listenWatchdog.current)
    listenWatchdog.current = setTimeout(() => {
      clearTimeout(graceTimer.current)
      flushPending()
    }, 12000)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setPhaseSafe])

  const onSpeechEnd = useCallback(
    (audio) => {
      clearTimeout(listenWatchdog.current)
      if (mutedRef.current) return
      pendingRef.current.push(audio)
      // Backchannel: wait a beat in case you're just pausing mid-thought.
      clearTimeout(graceTimer.current)
      graceTimer.current = setTimeout(flushPending, GRACE_MS)
    },
    [flushPending]
  )

  // ---- setup: websocket, audio engine, VAD ----
  useEffect(() => {
    let disposed = false
    const engine = new AudioEngine()
    engineRef.current = engine
    engine.onStart = () => {
      speakingRef.current = true
      setPhaseSafe('speaking')
    }
    engine.onDrain = () => {
      speakingRef.current = false
      // She finished talking; go idle so VAD listening reads as the active state.
      setPhase((prev) => (prev === 'speaking' || prev === 'thinking' ? 'idle' : prev))
    }

    const ws = new WebSocket(callSocketUrl())
    wsRef.current = ws

    ws.onopen = async () => {
      if (disposed) return
      await engine.ensure().catch(() => {})
      setPhaseSafe('ringing')
      // She "answers" after a short ring.
      setTimeout(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'start_call', session_id: sessionId }))
        }
      }, 1600)
      startVad(engine)
    }
    ws.onerror = () => setError('Call connection error.')
    ws.onmessage = async (ev) => {
      let msg
      try {
        msg = JSON.parse(ev.data)
      } catch {
        return
      }
      switch (msg.type) {
        case 'stt':
          if (msg.text) setCaption(`you: ${msg.text}`)
          break
        case 'reply_start':
          setPhaseSafe('thinking')
          break
        case 'emotion':
          setEmotion(msg.emotion || 'neutral')
          break
        case 'chunk': {
          setCaption(msg.text || '')
          const bytes = base64ToBytes(msg.audio)
          engine.enqueue(bytes.buffer, msg.timings).catch(() => {})
          break
        }
        case 'reply_end':
          if (msg.emotion) setEmotion(msg.emotion)
          break
        case 'cancelled':
          engine.flush()
          speakingRef.current = false
          break
        case 'error':
          setError(msg.message || 'Call error')
          // Without this, a turn that failed server-side left phase stuck on
          // 'thinking'/'listening' forever — the mic looked dead with no way
          // to recover short of ending the call.
          setPhaseSafe('idle')
          break
        default:
          break
      }
    }

    async function startVad(eng) {
      try {
        const { MicVAD } = await import('@ricky0123/vad-web')
        if (disposed) return
        const vad = await MicVAD.new({
          baseAssetPath: '/vad/',
          onnxWASMBasePath: '/vad/',
          model: 'v5',
          // Echo-cancel so her voice from the speakers doesn't trigger the mic
          // (this is what made "listening" start and never stop).
          additionalAudioConstraints: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
          // Tolerate short pauses so we don't chop mid-sentence.
          redemptionFrames: 16,
          minSpeechFrames: 5,
          positiveSpeechThreshold: 0.6,
          negativeSpeechThreshold: 0.4,
          onSpeechStart,
          onSpeechEnd,
        })
        vadRef.current = vad
        vad.start()
        setMicReady(true)
        // don't override an in-progress greeting
        setPhase((prev) => (prev === 'ringing' || prev === 'connecting' ? prev : 'idle'))
      } catch (e) {
        setError('Mic / VAD unavailable — check microphone permission.')
      }
    }

    return () => {
      disposed = true
      clearTimeout(graceTimer.current)
      clearTimeout(listenWatchdog.current)
      try {
        vadRef.current?.destroy?.()
      } catch {
        /* ignore */
      }
      engine.close()
      try {
        ws.close()
      } catch {
        /* ignore */
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId])

  // ---- call timer ----
  useEffect(() => {
    startedAtRef.current = Date.now()
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAtRef.current) / 1000))
    }, 1000)
    return () => clearInterval(id)
  }, [])

  const toggleMute = useCallback(() => {
    setMuted((m) => {
      const next = !m
      mutedRef.current = next
      if (next) vadRef.current?.pause?.()
      else vadRef.current?.start?.()
      return next
    })
  }, [])

  const endCall = useCallback(async () => {
    setPhase('ended')
    clearTimeout(graceTimer.current)
    try {
      vadRef.current?.destroy?.()
    } catch {
      /* ignore */
    }
    engineRef.current?.close()
    try {
      wsRef.current?.close()
    } catch {
      /* ignore */
    }
    const seconds = Math.floor((Date.now() - startedAtRef.current) / 1000)
    // Drop the "call ended" event bubble + store a summary memory.
    try {
      await fetch('/call/end', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, duration_seconds: seconds }),
      })
    } catch {
      /* ignore */
    }
    onEnded && onEnded(seconds)
  }, [sessionId, onEnded])

  const amplitude = useCallback(() => engineRef.current?.amplitude() ?? 0, [])
  const currentViseme = useCallback(() => engineRef.current?.currentViseme() ?? 'closed', [])

  return {
    phase,
    caption,
    emotion,
    muted,
    elapsed,
    micReady,
    error,
    toggleMute,
    endCall,
    amplitude,
    currentViseme,
    isSpeaking: () => speakingRef.current,
  }
}

function base64ToBytes(b64) {
  const bin = atob(b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return bytes
}
