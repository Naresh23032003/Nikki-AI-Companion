// WhatsApp bridge for the local AI companion.
//
// - Logs into a DEDICATED WhatsApp number via whatsapp-web.js (LocalAuth keeps
//   the session; a QR code prints in the terminal on first run).
// - Forwards incoming messages from YOUR number to the backend
//   (POST /whatsapp/incoming) and delivers the reply as text or a real
//   WhatsApp voice note (WAV -> ogg/opus via ffmpeg, sendAudioAsVoice).
// - Auto-rejects incoming WhatsApp calls and sends the backend's in-character
//   "can't pick up" text.
// - Exposes a local HTTP API (default :3001) the backend uses for proactive
//   messages: POST /send-text, POST /send-voice.
//
// Config via environment variables:
//   WA_TARGET     your personal number, digits only with country code
//                 (e.g. 919876543210)  [REQUIRED]
//   BACKEND_URL   companion backend (default http://localhost:8000)
//   PORT          bridge HTTP port (default 3001)
//
// See README.md — this uses an unofficial client; use a second number.

const { Client, LocalAuth, MessageMedia } = require('whatsapp-web.js')
const qrcode = require('qrcode-terminal')
const express = require('express')
const { execFile } = require('child_process')
const fs = require('fs')
const os = require('os')
const path = require('path')

const TARGET = (process.env.WA_TARGET || '').replace(/[^0-9]/g, '')
const BACKEND = (process.env.BACKEND_URL || 'http://localhost:8000').replace(/\/$/, '')
const PORT = parseInt(process.env.PORT || '3001', 10)
const ALLOW_ANY_SENDER = ['1', 'true', 'yes', 'on'].includes((process.env.WA_ALLOW_ANY_SENDER || '').toLowerCase())

if (!TARGET) {
  console.error('ERROR: set WA_TARGET to your personal number (digits only, with country code).')
  process.exit(1)
}
const TARGET_CHAT = `${TARGET}@c.us`

// ---------------------------------------------------------------------------
// WhatsApp client
// ---------------------------------------------------------------------------
const client = new Client({
  authStrategy: new LocalAuth({ dataPath: path.join(__dirname, '.wwebjs_auth') }),
  puppeteer: {
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  },
})

let ready = false

client.on('qr', (qr) => {
  console.log('\nScan this QR with the COMPANION WhatsApp account (Linked devices):\n')
  qrcode.generate(qr, { small: true })
})

client.on('ready', () => {
  ready = true
  console.log(`WhatsApp bridge ready. Forwarding messages from +${TARGET} to ${BACKEND}`)
})

client.on('disconnected', (reason) => {
  ready = false
  console.warn('WhatsApp disconnected:', reason)
})

// ---------------------------------------------------------------------------
// Incoming messages -> backend -> reply (text, voice note, photo, or her
// voice note in return). Shared by the text path and the media path below.
// ---------------------------------------------------------------------------
async function deliverReply(reply, chat) {
  // Human-ish typing delay proportional to TOTAL reply length (capped) —
  // texts is every bubble she'd send (see app/persona.py's "texted twice"
  // rule); a single-bubble reply is just a one-element array.
  const texts = (reply.texts && reply.texts.length ? reply.texts : [reply.text]).filter(Boolean)
  const totalLen = texts.reduce((n, t) => n + t.length, 0)
  const delay = Math.min(6000, 800 + totalLen * 40)
  await sleep(delay)
  chat.clearState().catch(() => {})

  if (reply.mode === 'sticker_only' && reply.sticker_path && fs.existsSync(reply.sticker_path)) {
    // Sometimes the sticker IS the reply.
    await sendSticker(reply.sticker_path)
    console.log('[out] (sticker only)')
    return
  }
  if (reply.mode === 'voice' && reply.wav_path && fs.existsSync(reply.wav_path)) {
    // The voice note IS the last bubble (spoken). Any earlier bubbles still
    // go out as text first — dropping them lost real content ("she only
    // sends voice messages") whenever a multi-bubble reply rolled voice.
    for (let i = 0; i < texts.length - 1; i++) {
      await client.sendMessage(TARGET_CHAT, texts[i])
      console.log(`[out] ${texts[i].slice(0, 80)}`)
      await sleep(500 + Math.random() * 800)
    }
    await sendVoice(reply.wav_path)
    console.log(`[out] (voice) ${(reply.text || '').slice(0, 60)}…`)
  } else {
    // Text mode: one message per bubble, with a re-typing pause between
    // them so a 2-3 part reply reads as "texted twice", not a wall of text.
    for (let i = 0; i < texts.length; i++) {
      if (i > 0) {
        await sleep(500 + Math.random() * 800)
        chat.sendStateTyping().catch(() => {})
        await sleep(400 + Math.random() * 500)
      }
      await client.sendMessage(TARGET_CHAT, texts[i])
      console.log(`[out] ${texts[i].slice(0, 80)}`)
    }
  }
  if (reply.sticker_path && fs.existsSync(reply.sticker_path)) {
    await sleep(700)
    await sendSticker(reply.sticker_path)
    console.log('[out] (sticker)')
  }
}

// ---------------------------------------------------------------------------
// Burst batching: real people fire 2-4 short texts in a row; replying to
// each one separately reads as pure bot. Debounce instead: hold incoming
// texts while the user is still typing, and only when they go quiet
// (WA_DEBOUNCE_MS without a new message) send the whole batch to the
// backend for ONE reply. Escape hatches: urgent-looking messages flush
// immediately, and a burst can never wait longer than WA_DEBOUNCE_MAX_MS
// total — a real person jumps in eventually too.
// ---------------------------------------------------------------------------
const DEBOUNCE_MS = parseInt(process.env.WA_DEBOUNCE_MS || '9000', 10)
const DEBOUNCE_MAX_MS = parseInt(process.env.WA_DEBOUNCE_MAX_MS || '30000', 10)
// Mirrors the backend's _URGENT regex — an urgent message must never sit
// in the buffer waiting for a quiet period.
const URGENT = /\b(urgent|emergency|help me|asap|right now|crying|scared|panic|hurt|hospital)\b|!{2,}/i

let pendingTexts = []
let debounceTimer = null
let firstPendingAt = 0

async function flushPending() {
  clearTimeout(debounceTimer)
  debounceTimer = null
  if (!pendingTexts.length) return
  const texts = pendingTexts
  pendingTexts = []
  firstPendingAt = 0
  try {
    const chat = await client.getChatById(TARGET_CHAT)
    // She "starts typing" only now — showing the indicator during the whole
    // debounce window would look like she's typing while you still are.
    chat.sendStateTyping().catch(() => {})
    const res = await fetch(`${BACKEND}/whatsapp/incoming`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: texts.join('\n'), texts, from_number: TARGET, source: 'whatsapp',
      }),
    })
    if (!res.ok) throw new Error(`backend ${res.status}`)
    await deliverReply(await res.json(), chat)
  } catch (err) {
    console.error('batch flush failed:', err.message)
  }
}

function queueText(body) {
  pendingTexts.push(body)
  if (!firstPendingAt) firstPendingAt = Date.now()
  clearTimeout(debounceTimer)
  if (URGENT.test(body)) return flushPending()
  // The waited-so-far time counts against the cap, so a long burst still
  // gets answered within DEBOUNCE_MAX_MS of its first message.
  const remaining = DEBOUNCE_MAX_MS - (Date.now() - firstPendingAt)
  if (remaining <= 0) return flushPending()
  debounceTimer = setTimeout(() => {
    flushPending().catch((err) => console.error('flush failed:', err.message))
  }, Math.min(DEBOUNCE_MS, remaining))
  return null
}

client.on('message', async (msg) => {
  try {
    const sender = msg.from || ''
    const isTargetChat = sender === TARGET_CHAT || ALLOW_ANY_SENDER
    if (!isTargetChat) return // only my human (or any sender in debug mode)

    // Photos and voice notes -> a local vision/STT model on the backend,
    // then the SAME reply pipeline as text (routing/guards/stickers/etc).
    if (msg.hasMedia && (msg.type === 'image' || msg.type === 'ptt' || msg.type === 'audio')) {
      // Any buffered text burst gets answered first so the media reply
      // doesn't arrive before the texts it followed.
      await flushPending()
      const media = await msg.downloadMedia().catch(() => null)
      if (!media) return
      const kind = msg.type === 'image' ? 'image' : 'voice'
      console.log(`[in ] (${kind} message) from ${sender}`)
      const chat = await msg.getChat()
      chat.sendStateTyping().catch(() => {})
      const res = await fetch(`${BACKEND}/whatsapp/incoming-media`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          kind, data_b64: media.data, mimetype: media.mimetype, caption: msg.body || null,
        }),
      })
      if (!res.ok) throw new Error(`backend ${res.status}`)
      await deliverReply(await res.json(), chat)
      return
    }

    if (!msg.body || msg.type !== 'chat') return // other unsupported message types

    console.log(`[in ] ${msg.body} from ${sender} (batching)`)
    await queueText(msg.body)
  } catch (err) {
    console.error('incoming handler failed:', err.message)
  }
})

// ---------------------------------------------------------------------------
// Incoming CALLS: cannot be answered by web clients -> reject + excuse text
// ---------------------------------------------------------------------------
client.on('call', async (call) => {
  try {
    console.log(`[call] incoming ${call.isVideo ? 'video' : 'voice'} call — rejecting`)
    await call.reject()
    const res = await fetch(`${BACKEND}/whatsapp/call-rejected`, { method: 'POST' })
    const { text } = res.ok
      ? await res.json()
      : { text: "ahh I can't pick up on here 😭 call me on our app??" }
    await sleep(1200)
    await client.sendMessage(TARGET_CHAT, text)
    console.log(`[out] ${text}`)
  } catch (err) {
    console.error('call handler failed:', err.message)
  }
})

// ---------------------------------------------------------------------------
// Voice notes: WAV -> ogg/opus with ffmpeg, sent as a real voice message
// ---------------------------------------------------------------------------
// Resolve ffmpeg explicitly instead of relying on ambient PATH: this process
// is often started from a different shell/session than the one it was set up
// in (a plain terminal without Anaconda/conda's PATH additions, a scheduled
// task, etc), and a bare 'ffmpeg' then silently fails with an opaque
// "Command failed" error. FFMPEG_PATH env var overrides; otherwise try common
// install locations before falling back to PATH lookup.
const FFMPEG_CANDIDATES = [
  process.env.FFMPEG_PATH,
  'ffmpeg', // ambient PATH — works if this shell has it
  `${process.env.USERPROFILE || ''}\\anaconda3\\Library\\bin\\ffmpeg.exe`,
  `${process.env.USERPROFILE || ''}\\miniconda3\\Library\\bin\\ffmpeg.exe`,
  'C:\\ffmpeg\\bin\\ffmpeg.exe',
].filter(Boolean)

let _ffmpegPath = null

function resolveFfmpeg() {
  if (_ffmpegPath) return _ffmpegPath
  for (const candidate of FFMPEG_CANDIDATES) {
    try {
      require('child_process').execFileSync(
        candidate, ['-version'], { stdio: 'ignore' })
      _ffmpegPath = candidate
      console.log(`ffmpeg resolved: ${candidate}`)
      return candidate
    } catch {
      // try the next candidate
    }
  }
  throw new Error(
    'ffmpeg not found (tried: ' + FFMPEG_CANDIDATES.join(', ') + '). ' +
    'Set FFMPEG_PATH to its full exe path, or add it to PATH and restart the bridge.')
}

function ffmpegConvert(wavPath, oggPath) {
  return new Promise((resolve, reject) => {
    let exe
    try {
      exe = resolveFfmpeg()
    } catch (e) {
      reject(e)
      return
    }
    execFile(
      exe,
      ['-y', '-i', wavPath, '-c:a', 'libopus', '-b:a', '32k', '-vbr', 'on', oggPath],
      (err, _stdout, stderr) => (err ? reject(new Error(`${err.message}\n${stderr || ''}`)) : resolve())
    )
  })
}

async function sendVoice(wavPath) {
  const ogg = path.join(os.tmpdir(), `vn_${Date.now()}.ogg`)
  await ffmpegConvert(wavPath, ogg)
  try {
    const media = MessageMedia.fromFilePath(ogg)
    media.mimetype = 'audio/ogg; codecs=opus'
    await client.sendMessage(TARGET_CHAT, media, { sendAudioAsVoice: true })
  } finally {
    fs.unlink(ogg, () => {})
  }
}

// Stickers: whatsapp-web.js converts the image to a real WhatsApp sticker.
async function sendSticker(imgPath) {
  const media = MessageMedia.fromFilePath(imgPath)
  await client.sendMessage(TARGET_CHAT, media, { sendMediaAsSticker: true })
}

// ---------------------------------------------------------------------------
// Local HTTP API (used by the backend's proactive scheduler)
// ---------------------------------------------------------------------------
const app = express()
app.use(express.json())

app.get('/status', (_req, res) => res.json({ ready, target: TARGET }))

app.post('/send-text', async (req, res) => {
  try {
    if (!ready) return res.status(503).json({ error: 'whatsapp not ready' })
    const { text } = req.body || {}
    if (!text) return res.status(400).json({ error: 'text required' })
    await client.sendMessage(TARGET_CHAT, text)
    console.log(`[out] (proactive) ${text.slice(0, 80)}`)
    res.json({ ok: true })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

app.post('/send-sticker', async (req, res) => {
  try {
    if (!ready) return res.status(503).json({ error: 'whatsapp not ready' })
    const { path: imgPath } = req.body || {}
    if (!imgPath || !isAllowedPath(imgPath) || !fs.existsSync(imgPath)) {
      return res.status(400).json({ error: 'path missing, not found, or not allowed' })
    }
    await sendSticker(imgPath)
    console.log('[out] (proactive sticker)')
    res.json({ ok: true })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

app.post('/send-voice', async (req, res) => {
  try {
    if (!ready) return res.status(503).json({ error: 'whatsapp not ready' })
    const { wav_path } = req.body || {}
    if (!wav_path || !isAllowedPath(wav_path) || !fs.existsSync(wav_path)) {
      return res.status(400).json({ error: 'wav_path missing, not found, or not allowed' })
    }
    await sendVoice(wav_path)
    res.json({ ok: true })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms))
}

// Path allowlist for /send-voice and /send-sticker: these endpoints have no
// auth (the bridge binds 127.0.0.1, trusting anything local), but they used
// to accept ANY absolute path and read+send it over WhatsApp — any local
// process could exfiltrate arbitrary files on disk that way. Legitimate
// callers only ever pass paths under these three directories.
const PROJECT_ROOT = path.resolve(__dirname, '..')
const ALLOWED_DIRS = [
  path.join(PROJECT_ROOT, 'media', 'tts'),
  path.join(PROJECT_ROOT, 'songs', 'library'),
  path.join(PROJECT_ROOT, 'stickers'),
].map((p) => path.resolve(p))

function isAllowedPath(p) {
  const resolved = path.resolve(p)
  return ALLOWED_DIRS.some((dir) => resolved === dir || resolved.startsWith(dir + path.sep))
}

app.listen(PORT, '127.0.0.1', () => {
  console.log(`Bridge HTTP API on http://127.0.0.1:${PORT}`)
})

client.initialize()
