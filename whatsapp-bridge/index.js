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
// See README.md - this uses an unofficial client; use a second number.

const { Client, LocalAuth, MessageMedia } = require('whatsapp-web.js')
const qrcode = require('qrcode-terminal')
const express = require('express')
const { execFile } = require('child_process')
const fs = require('fs')
const os = require('os')
const path = require('path')

const digits = (s) => String(s || '').replace(/[^0-9]/g, '')

// Multi-person: every allowed number gets its own persona on the backend (see
// app/profiles.py). WA_TARGETS is a comma-separated allowlist; WA_TARGET stays
// supported as the single-number form so existing setups keep working.
//   WA_TARGETS=919876543210,919999999999
const TARGET = digits(process.env.WA_TARGET)
const TARGETS = [
  ...new Set(
    [...(process.env.WA_TARGETS || '').split(','), TARGET].map(digits).filter(Boolean)
  ),
]
const BACKEND = (process.env.BACKEND_URL || 'http://localhost:8000').replace(/\/$/, '')
const PORT = parseInt(process.env.PORT || '3001', 10)
const ALLOW_ANY_SENDER = ['1', 'true', 'yes', 'on'].includes((process.env.WA_ALLOW_ANY_SENDER || '').toLowerCase())

if (!TARGETS.length) {
  console.error('ERROR: set WA_TARGET (or WA_TARGETS=a,b) to the allowed numbers, digits only with country code.')
  process.exit(1)
}
const TARGET_CHAT = `${TARGETS[0]}@c.us`

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

// WhatsApp has been migrating personal chats from phone-number JIDs (@c.us)
// to opaque "linked ID" JIDs (@lid) for privacy - once a contact migrates,
// every getChatById/sendMessage call using the old @c.us id starts failing
// (with an unhelpful minified internal error, not a clear one).
//
// With multiple people we need BOTH directions:
//   chatIdFor:  number -> the JID that currently works for sending
//   numberFor:  JID (possibly @lid) -> the phone number, to pick the persona
const chatIdFor = new Map() // '919876543210' -> '1153...@lid' | '9198...@c.us'
const numberFor = new Map() // '1153...@lid'  -> '919876543210'

function rememberMapping(number, chatId) {
  if (!number || !chatId) return
  chatIdFor.set(number, chatId)
  numberFor.set(chatId, number)
}

client.on('qr', (qr) => {
  console.log('\nScan this QR with the COMPANION WhatsApp account (Linked devices):\n')
  qrcode.generate(qr, { small: true })
})

client.on('ready', async () => {
  ready = true
  console.log(`WhatsApp bridge ready. Forwarding ${TARGETS.length} number(s) to ${BACKEND}`)
  for (const number of TARGETS) {
    const jid = `${number}@c.us`
    rememberMapping(number, jid) // sane default until/unless a lid shows up
    try {
      const [info] = await client.getContactLidAndPhone([jid])
      if (info?.lid) {
        rememberMapping(number, info.lid)
        console.log(`  +${number} -> lid ${info.lid}`)
      } else {
        console.log(`  +${number} -> ${jid}`)
      }
    } catch (e) {
      console.warn(`  +${number}: lid resolution failed, staying on @c.us (${e.message})`)
    }
  }
})

client.on('disconnected', (reason) => {
  ready = false
  console.warn('WhatsApp disconnected:', reason)
})

/**
 * Which allowed person sent this? Returns their phone number (digits), or
 * null if they're not on the allowlist. Handles @lid senders by asking
 * whatsapp-web.js for the underlying phone number once, then caching it.
 */
async function senderNumber(jid) {
  if (!jid) return null
  if (numberFor.has(jid)) return numberFor.get(jid)

  const bare = digits(jid.split('@')[0])
  // Plain @c.us sender: the JID already IS the number.
  if (jid.endsWith('@c.us') && TARGETS.includes(bare)) {
    rememberMapping(bare, jid)
    return bare
  }
  // @lid sender: resolve to a phone number and check the allowlist.
  try {
    const [info] = await client.getContactLidAndPhone([jid])
    const pn = digits(info?.pn)
    if (pn && (TARGETS.includes(pn) || ALLOW_ANY_SENDER)) {
      rememberMapping(pn, jid)
      return pn
    }
  } catch {
    // fall through
  }
  if (ALLOW_ANY_SENDER) return bare || null
  return null
}

/** The JID to send to for a given number (falls back to @c.us). */
function chatIdForNumber(number) {
  const n = digits(number)
  return chatIdFor.get(n) || (n ? `${n}@c.us` : null)
}

// ---------------------------------------------------------------------------
// Incoming messages -> backend -> reply (text, voice note, photo, or her
// voice note in return). Shared by the text path and the media path below.
// ---------------------------------------------------------------------------
// getChatById() does an extra internal step beyond just finding the chat
// (building a full serializable model, incl. contact metadata) - that's the
// step actually crashing for @lid-addressed chats WhatsApp's own web client
// hasn't cached yet ("No LID for user" - a known whatsapp-web.js gap, see
// github.com/pedroslopez/whatsapp-web.js/issues/3834). client.sendMessage()
// skips that step entirely, so the real send can succeed even when this
// can't - typing-indicator state is cosmetic and must never block a reply.
async function bestEffortChatState(chatId, fn) {
  try {
    const chat = await client.getChatById(chatId)
    await fn(chat)
  } catch {
    // cosmetic only - see comment above
  }
}

async function deliverReply(reply, chatId) {
  // Human-ish typing delay proportional to TOTAL reply length (capped) -
  // texts is every bubble she'd send (see app/persona.py's "texted twice"
  // rule); a single-bubble reply is just a one-element array.
  const texts = (reply.texts && reply.texts.length ? reply.texts : [reply.text]).filter(Boolean)
  const totalLen = texts.reduce((n, t) => n + t.length, 0)
  const delay = Math.min(6000, 800 + totalLen * 40)
  await sleep(delay)
  await bestEffortChatState(chatId, (chat) => chat.clearState())

  if (reply.mode === 'sticker_only' && reply.sticker_path && fs.existsSync(reply.sticker_path)) {
    // Sometimes the sticker IS the reply.
    await sendSticker(reply.sticker_path, chatId)
    console.log('[out] (sticker only)')
    return
  }
  if (reply.mode === 'voice' && reply.wav_path && fs.existsSync(reply.wav_path)) {
    // The voice note IS the last bubble (spoken). Any earlier bubbles still
    // go out as text first - dropping them lost real content ("she only
    // sends voice messages") whenever a multi-bubble reply rolled voice.
    for (let i = 0; i < texts.length - 1; i++) {
      await client.sendMessage(chatId, texts[i])
      console.log(`[out] ${texts[i].slice(0, 80)}`)
      await sleep(500 + Math.random() * 800)
    }
    await sendVoice(reply.wav_path, chatId)
    console.log(`[out] (voice) ${(reply.text || '').slice(0, 60)}…`)
  } else {
    // Text mode: one message per bubble, with a re-typing pause between
    // them so a 2-3 part reply reads as "texted twice", not a wall of text.
    for (let i = 0; i < texts.length; i++) {
      if (i > 0) {
        await sleep(500 + Math.random() * 800)
        await bestEffortChatState(chatId, (chat) => chat.sendStateTyping())
        await sleep(400 + Math.random() * 500)
      }
      await client.sendMessage(chatId, texts[i])
      console.log(`[out] ${texts[i].slice(0, 80)}`)
    }
  }
  if (reply.sticker_path && fs.existsSync(reply.sticker_path)) {
    await sleep(700)
    await sendSticker(reply.sticker_path, chatId)
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
// total - a real person jumps in eventually too.
// ---------------------------------------------------------------------------
const DEBOUNCE_MS = parseInt(process.env.WA_DEBOUNCE_MS || '9000', 10)
const DEBOUNCE_MAX_MS = parseInt(process.env.WA_DEBOUNCE_MAX_MS || '30000', 10)
// How long to wait before re-trying a batch the backend couldn't take (it's
// usually just restarting - a few seconds of downtime shouldn't eat a text).
const RETRY_MS = parseInt(process.env.WA_RETRY_MS || '15000', 10)

/**
 * `TypeError: fetch failed` on its own is useless - undici hides the real
 * reason (ECONNREFUSED / socket hang up / timeout) in err.cause. Surface it.
 */
function describeErr(err) {
  const cause = err && err.cause
  if (cause) {
    const detail = cause.code || cause.message || String(cause)
    return `${err.message} (cause: ${detail})`
  }
  return err && (err.stack || err.message) ? err.stack || err.message : String(err)
}

/** POST JSON with one retry - survives a backend restart mid-conversation. */
async function postJson(url, body, attempts = 2) {
  let lastErr
  for (let i = 0; i < attempts; i++) {
    try {
      return await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
    } catch (err) {
      lastErr = err
      if (i < attempts - 1) {
        console.warn(`  backend unreachable (${describeErr(err)}) - retrying once...`)
        await sleep(2000)
      }
    }
  }
  throw lastErr
}
// Mirrors the backend's _URGENT regex - an urgent message must never sit
// in the buffer waiting for a quiet period.
const URGENT = /\b(urgent|emergency|help me|asap|right now|crying|scared|panic|hurt|hospital)\b|!{2,}/i

// One buffer PER SENDER - with two people on the same account a single shared
// buffer would merge their bursts into one batch and hand one person's texts
// to the other's persona.
const pending = new Map() // number -> { texts: [], timer, firstAt }

function bufferFor(number) {
  let b = pending.get(number)
  if (!b) {
    b = { texts: [], timer: null, firstAt: 0 }
    pending.set(number, b)
  }
  return b
}

async function flushPending(number) {
  const b = pending.get(number)
  if (!b) return
  clearTimeout(b.timer)
  b.timer = null
  if (!b.texts.length) return
  const texts = b.texts
  b.texts = []
  b.firstAt = 0
  const chatId = chatIdForNumber(number)
  try {
    // She "starts typing" only now - showing the indicator during the whole
    // debounce window would look like she's typing while you still are.
    // Best-effort: must never block the actual reply (see bestEffortChatState).
    await bestEffortChatState(chatId, (chat) => chat.sendStateTyping())
    const res = await postJson(`${BACKEND}/whatsapp/incoming`, {
      text: texts.join('\n'), texts, from_number: number, source: 'whatsapp',
    })
    if (!res.ok) throw new Error(`backend ${res.status}`)
    await deliverReply(await res.json(), chatId)
  } catch (err) {
    // The texts were already taken out of the buffer, so without this they'd
    // be silently lost and she'd just ghost - which is exactly how a backend
    // restart used to look from the phone. Put them back at the FRONT so the
    // next message (or the retry below) answers the whole thread in order.
    b.texts = [...texts, ...b.texts]
    console.error(`batch flush failed (+${number}): ${describeErr(err)}`)
    console.error(`  -> ${texts.length} message(s) re-queued; retrying in ${RETRY_MS / 1000}s`)
    clearTimeout(b.timer)
    b.timer = setTimeout(() => {
      flushPending(number).catch(() => {})
    }, RETRY_MS)
  }
}

function queueText(number, body) {
  const b = bufferFor(number)
  b.texts.push(body)
  if (!b.firstAt) b.firstAt = Date.now()
  clearTimeout(b.timer)
  if (URGENT.test(body)) return flushPending(number)
  // The waited-so-far time counts against the cap, so a long burst still
  // gets answered within DEBOUNCE_MAX_MS of its first message.
  const remaining = DEBOUNCE_MAX_MS - (Date.now() - b.firstAt)
  if (remaining <= 0) return flushPending(number)
  b.timer = setTimeout(() => {
    flushPending(number).catch((err) =>
      console.error('flush failed:', describeErr(err)))
  }, Math.min(DEBOUNCE_MS, remaining))
  return null
}

client.on('message', async (msg) => {
  try {
    const sender = msg.from || ''
    // Which allowed person is this? null = not on the allowlist, ignore.
    const number = await senderNumber(sender)
    if (!number) return
    // Remember whichever id form this message actually arrived on - if
    // WhatsApp migrated this contact to @lid mid-session, this keeps sends
    // pointed at a chat id that still resolves instead of a stale @c.us one.
    rememberMapping(number, sender)

    // Photos and voice notes -> a local vision/STT model on the backend,
    // then the SAME reply pipeline as text (routing/guards/stickers/etc).
    if (msg.hasMedia && (msg.type === 'image' || msg.type === 'ptt' || msg.type === 'audio')) {
      // This sender's buffered text burst gets answered first so the media
      // reply doesn't arrive before the texts it followed.
      await flushPending(number)
      const media = await msg.downloadMedia().catch(() => null)
      if (!media) return
      const kind = msg.type === 'image' ? 'image' : 'voice'
      console.log(`[in ] (${kind} message) from +${number}`)
      // msg.getChat() is just client.getChatById() under the hood - same
      // @lid resolution gap, so it's best-effort here too (see bestEffortChatState).
      await bestEffortChatState(sender, (chat) => chat.sendStateTyping())
      const res = await postJson(`${BACKEND}/whatsapp/incoming-media`, {
        kind, data_b64: media.data, mimetype: media.mimetype,
        caption: msg.body || null, from_number: number,
      })
      if (!res.ok) throw new Error(`backend ${res.status}`)
      await deliverReply(await res.json(), sender)
      return
    }

    if (!msg.body || msg.type !== 'chat') return // other unsupported message types

    console.log(`[in ] (+${number}) ${msg.body} (batching)`)
    await queueText(number, msg.body)
  } catch (err) {
    console.error('incoming handler failed:', describeErr(err))
  }
})

// ---------------------------------------------------------------------------
// Incoming CALLS: cannot be answered by web clients -> reject + excuse text
// ---------------------------------------------------------------------------
client.on('call', async (call) => {
  try {
    // Whoever called gets HER excuse, from her own persona - not the default's.
    const number = await senderNumber(call.from || '')
    if (!number) return
    console.log(`[call] incoming ${call.isVideo ? 'video' : 'voice'} call from +${number} - rejecting`)
    await call.reject()
    const res = await postJson(`${BACKEND}/whatsapp/call-rejected`, { from_number: number })
    const { text } = res.ok
      ? await res.json()
      : { text: "ahh I can't pick up on here 😭 call me on our app??" }
    await sleep(1200)
    await client.sendMessage(chatIdForNumber(number), text)
    console.log(`[out] ${text}`)
  } catch (err) {
    console.error('call handler failed:', describeErr(err))
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
  'ffmpeg', // ambient PATH - works if this shell has it
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

// All media senders take an explicit chatId - with two people on one account
// there is no such thing as "the" chat any more.
async function sendVoice(wavPath, chatId) {
  const ogg = path.join(os.tmpdir(), `vn_${Date.now()}.ogg`)
  await ffmpegConvert(wavPath, ogg)
  try {
    const media = MessageMedia.fromFilePath(ogg)
    media.mimetype = 'audio/ogg; codecs=opus'
    await client.sendMessage(chatId, media, { sendAudioAsVoice: true })
  } finally {
    fs.unlink(ogg, () => {})
  }
}

// Stickers: whatsapp-web.js converts the image to a real WhatsApp sticker.
async function sendSticker(imgPath, chatId) {
  const media = MessageMedia.fromFilePath(imgPath)
  await client.sendMessage(chatId, media, { sendMediaAsSticker: true })
}

// Drawn/generated images: sent as a normal photo attachment, not a sticker.
async function sendImage(imgPath, chatId) {
  const media = MessageMedia.fromFilePath(imgPath)
  await client.sendMessage(chatId, media)
}

// Default send target for the HTTP API when no `to` is given: preserves
// single-number behavior for older callers.
const defaultChatId = () => chatIdForNumber(TARGETS[0])

// ---------------------------------------------------------------------------
// Local HTTP API (used by the backend's proactive scheduler)
// ---------------------------------------------------------------------------
const app = express()
app.use(express.json())

app.get('/status', (_req, res) => res.json({ ready, targets: TARGETS }))

app.post('/send-text', async (req, res) => {
  try {
    if (!ready) return res.status(503).json({ error: 'whatsapp not ready' })
    const { text, to } = req.body || {}
    if (!text) return res.status(400).json({ error: 'text required' })
    // `to` picks the person (the backend sends the profile's number); without
    // it we fall back to the first target = old single-number behavior.
    const chatId = to ? chatIdForNumber(to) : defaultChatId()
    if (!chatId) return res.status(400).json({ error: `unknown recipient ${to}` })
    await client.sendMessage(chatId, text)
    console.log(`[out] (proactive -> +${digits(to) || TARGETS[0]}) ${text.slice(0, 60)}`)
    res.json({ ok: true })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

app.post('/send-sticker', async (req, res) => {
  try {
    if (!ready) return res.status(503).json({ error: 'whatsapp not ready' })
    const { path: imgPath, to } = req.body || {}
    if (!imgPath || !isAllowedPath(imgPath) || !fs.existsSync(imgPath)) {
      return res.status(400).json({ error: 'path missing, not found, or not allowed' })
    }
    const chatId = to ? chatIdForNumber(to) : defaultChatId()
    if (!chatId) return res.status(400).json({ error: `unknown recipient ${to}` })
    await sendSticker(imgPath, chatId)
    console.log(`[out] (sticker -> +${digits(to) || TARGETS[0]})`)
    res.json({ ok: true })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

app.post('/send-voice', async (req, res) => {
  try {
    if (!ready) return res.status(503).json({ error: 'whatsapp not ready' })
    const { wav_path, to } = req.body || {}
    if (!wav_path || !isAllowedPath(wav_path) || !fs.existsSync(wav_path)) {
      return res.status(400).json({ error: 'wav_path missing, not found, or not allowed' })
    }
    const chatId = to ? chatIdForNumber(to) : defaultChatId()
    if (!chatId) return res.status(400).json({ error: `unknown recipient ${to}` })
    await sendVoice(wav_path, chatId)
    console.log(`[out] (voice -> +${digits(to) || TARGETS[0]})`)
    res.json({ ok: true })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

app.post('/send-image', async (req, res) => {
  try {
    if (!ready) return res.status(503).json({ error: 'whatsapp not ready' })
    const { path: imgPath, to } = req.body || {}
    if (!imgPath || !isAllowedPath(imgPath) || !fs.existsSync(imgPath)) {
      return res.status(400).json({ error: 'path missing, not found, or not allowed' })
    }
    const chatId = to ? chatIdForNumber(to) : defaultChatId()
    if (!chatId) return res.status(400).json({ error: `unknown recipient ${to}` })
    await sendImage(imgPath, chatId)
    console.log(`[out] (drawn image -> +${digits(to) || TARGETS[0]})`)
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
// to accept ANY absolute path and read+send it over WhatsApp - any local
// process could exfiltrate arbitrary files on disk that way. Legitimate
// callers only ever pass paths under these three directories.
const PROJECT_ROOT = path.resolve(__dirname, '..')
const ALLOWED_DIRS = [
  path.join(PROJECT_ROOT, 'media', 'tts'),
  path.join(PROJECT_ROOT, 'media', 'images'),
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
