# WhatsApp bridge

Connects the companion backend to WhatsApp so she can text (and send voice
notes) on a real WhatsApp number, and so your WhatsApp messages flow through
the same persona + memory pipeline as the web app.

## Read this first

- **This uses `whatsapp-web.js`, an UNOFFICIAL WhatsApp client.** WhatsApp's
  terms prohibit automation, and **accounts using unofficial clients can be
  permanently banned.**
- **Use a dedicated second number** (cheap SIM / eSIM), never your main
  account. Treat the number as disposable.
- Keep it personal-scale: one chat, human-ish pacing. Bulk/spam behavior gets
  numbers banned fast.

## Prerequisites

- Node.js 18+
- `ffmpeg` on PATH (voice-note conversion to ogg/opus)
- The companion backend running on port 8000
- A phone with the **companion's** WhatsApp account (the second number)

## Setup

```bash
cd whatsapp-bridge
npm install          # first time only (downloads Chromium for puppeteer)

# Windows PowerShell
$env:WA_TARGET = "919876543210"   # your personal number, digits only
node index.js

# macOS/Linux
WA_TARGET=919876543210 node index.js
```

On first run a **QR code** prints in the terminal - scan it from the
companion's phone (WhatsApp → Linked devices). LocalAuth persists the session
in `.wwebjs_auth/`, so subsequent runs log in automatically.

## What it does

| Direction | Behavior |
|---|---|
| You → her | Your WhatsApp texts POST to the backend (`/whatsapp/incoming`); the reply comes back through the same persona + memory pipeline (session `main`, shared with the web app) |
| Her → you | Replies are sent as text, or (~30%, `whatsapp.voice_reply_ratio` in `config.yaml`) as a **real voice note** (Kokoro WAV → opus via ffmpeg → `sendAudioAsVoice`) |
| You call her on WhatsApp | The bridge **cannot answer calls** - it auto-rejects, logs `Missed WhatsApp call` into shared history, stores a memory ("he tried to call me…"), and texts you an in-character excuse pointing you to the app's Call mode |
| Proactive | The backend's scheduler pushes her check-in texts through `POST /send-text` |

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `WA_TARGET` | - (required) | Your personal number, digits only with country code |
| `BACKEND_URL` | `http://localhost:8000` | Companion backend |
| `PORT` | `3001` | Bridge HTTP port (localhost only) |
