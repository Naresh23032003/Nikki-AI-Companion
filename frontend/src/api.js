// Thin API client. Same-origin relative URLs - the FastAPI server hosts both
// the API and these static files in production; Vite proxies them in dev.

// LAN auth: when the server has COMPANION_AUTH_TOKEN set, non-localhost
// clients must send it. Open the PWA once as http://<host>:8000/?token=XYZ -
// it's remembered in localStorage and attached to every request after that.
const urlToken = new URLSearchParams(location.search).get('token')
if (urlToken) localStorage.setItem('companion_token', urlToken)
export function authToken() {
  return localStorage.getItem('companion_token') || ''
}

const _fetch = window.fetch.bind(window)
window.fetch = (input, init = {}) => {
  const t = authToken()
  if (t) init.headers = { ...(init.headers || {}), 'X-Auth-Token': t }
  return _fetch(input, init)
}

async function json(res) {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const api = {
  getPersona: () => fetch('/persona').then(json),
  getPersonas: () => fetch('/personas').then(json),
  setActivePersona: (id) =>
    fetch('/personas/active', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    }).then(json),

  uploadPhoto: (file) => {
    const fd = new FormData()
    fd.append('file', file)
    return fetch('/persona/photo', { method: 'POST', body: fd }).then(json)
  },

  getHistory: (sessionId) => fetch(`/history/${encodeURIComponent(sessionId)}`).then(json),
  clearChat: (sessionId) =>
    fetch(`/history/${encodeURIComponent(sessionId)}`, { method: 'DELETE' }).then(json),

  getMemories: () => fetch('/memories').then(json),
  addMemory: (fact, category) =>
    fetch('/memories', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fact, category }),
    }).then(json),
  deleteMemory: (id) => fetch(`/memories/${id}`, { method: 'DELETE' }).then(json),
  completeMemory: (id) => fetch(`/memories/${id}/complete`, { method: 'POST' }).then(json),

  // Speech-to-text. Degrades gracefully if the endpoint/model isn't there.
  stt: async (blob) => {
    const fd = new FormData()
    fd.append('file', blob, 'voice.webm')
    const res = await fetch('/stt', { method: 'POST', body: fd })
    if (res.status === 404 || res.status === 405 || res.status === 503) {
      throw new Error('STT_NOT_AVAILABLE')
    }
    return json(res)
  },

  // Voice system.
  voiceStatus: () => fetch('/voice/status').then(json),
  voiceBench: (emotions) =>
    fetch('/voice/bench', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ emotions }),
    }).then(json),

  // Two-brain status + her day (dev).
  brainStatus: () => fetch('/brain/status').then(json),
  dayState: () => fetch('/day-state').then(json),
  regenerateDayState: () => fetch('/day-state/regenerate', { method: 'POST' }).then(json),

  // Relationship progression.
  getRelationship: () => fetch('/relationship').then(json),
  overrideRelationship: (payload) =>
    fetch('/relationship/override', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(json),

  // Proactive messaging controls.
  proactiveStatus: () => fetch('/proactive/status').then(json),
  proactivePause: (hours) =>
    fetch('/proactive/pause', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hours }),
    }).then(json),

  // Mood journal.
  getJournal: (since, mood) => {
    const params = new URLSearchParams()
    if (since) params.set('since', since)
    if (mood) params.set('mood', mood)
    const qs = params.toString()
    return fetch(`/journal${qs ? `?${qs}` : ''}`).then(json)
  },
  updateJournalEntry: (id, fields) =>
    fetch(`/journal/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(fields),
    }).then(json),
  deleteJournalEntry: (id) => fetch(`/journal/${id}`, { method: 'DELETE' }).then(json),

  // Text-to-speech (file mode) -> { audio_url, duration, timings }.
  // Pass message_id to persist the voice note onto that stored message.
  tts: async (text, messageId) => {
    const res = await fetch('/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, message_id: messageId ?? null }),
    })
    if (res.status === 503) throw new Error('TTS_NOT_AVAILABLE')
    return json(res)
  },
}

// WebSocket URL for Call mode (same host, ws/wss to match the page).
// WebSockets can't send custom headers from the browser, so the auth token
// rides as a query param instead (validated server-side the same way).
export function callSocketUrl() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const t = authToken()
  return `${proto}://${location.host}/ws/call${t ? `?token=${encodeURIComponent(t)}` : ''}`
}

// Record a short utterance from the mic; resolves with a webm/opus Blob.
// Returns a controller with stop() to end recording.
export function startRecorder(onReady) {
  let recorder
  let chunks = []
  navigator.mediaDevices
    .getUserMedia({ audio: true })
    .then((stream) => {
      recorder = new MediaRecorder(stream, { mimeType: pickMime() })
      recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data)
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        onReady(new Blob(chunks, { type: recorder.mimeType || 'audio/webm' }))
      }
      recorder.start()
    })
    .catch(() => onReady(null))
  return {
    stop: () => {
      try {
        recorder && recorder.state !== 'inactive' && recorder.stop()
      } catch {
        /* ignore */
      }
    },
  }
}

function pickMime() {
  const opts = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus']
  for (const o of opts) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(o)) return o
  }
  return ''
}

export function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const r = new FileReader()
    r.onerror = reject
    r.onload = () => resolve(String(r.result).split(',')[1]) // strip data: prefix
    r.readAsDataURL(blob)
  })
}

// Stream a chat reply via Server-Sent Events over fetch.
// Calls onToken(token) for each chunk, onDone() when finished, onError(msg).
// Returns an AbortController so the caller can cancel.
export function streamChat(message, sessionId, { onToken, onDone, onError }) {
  const controller = new AbortController()

  ;(async () => {
    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: sessionId }),
        signal: controller.signal,
      })
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // SSE frames are separated by a blank line.
        let idx
        while ((idx = buffer.indexOf('\n\n')) !== -1) {
          const frame = buffer.slice(0, idx)
          buffer = buffer.slice(idx + 2)
          handleFrame(frame, { onToken, onDone, onError })
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') onError?.(err.message || String(err))
    }
  })()

  return controller
}

function handleFrame(frame, { onToken, onDone, onError }) {
  let event = 'message'
  const dataLines = []
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (!dataLines.length) return
  let payload
  try {
    payload = JSON.parse(dataLines.join('\n'))
  } catch {
    return
  }
  if (event === 'token') onToken?.(payload.token || '')
  else if (event === 'done') onDone?.(payload) // { message_id, text }
  else if (event === 'error') onError?.(payload.error || 'stream error')
}
