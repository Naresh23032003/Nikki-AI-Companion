# Nikki AI Companion 💬

A **private, self-hosted AI companion** that lives entirely on your own machine.
She texts, sends voice notes, takes hands-free voice calls with an animated
avatar, messages you first on WhatsApp, remembers your life across sessions,
and grows a real relationship with you over time all running locally on a
single 6 GB GPU laptop.

**One FastAPI server** talks to a local [Ollama](https://ollama.com) LLM and
serves a **WhatsApp-style React PWA**, so you can install her on your phone
from your own WiFi and no conversation ever has to leave your house.

```
Python · FastAPI · Ollama · React (Vite PWA) · SQLite + ChromaDB
faster-whisper · Kokoro TTS · RVC · XTTS-v2 · whatsapp-web.js
```

---

## Table of contents

- [Feature demos](#feature-demos)
 - [1. WhatsApp-style chat](#1-whatsapp-style-chat)
 - [2. Long-term memory](#2-long-term-memory)
 - [3. Voice notes (TTS) + speech input (STT)](#3-voice-notes-tts--speech-input-stt)
 - [4. Call mode hands-free voice calls](#4-call-mode--hands-free-voice-calls)
 - [5. Her own cloned voice (RVC + studio TTS)](#5-her-own-cloned-voice-rvc--studio-tts)
 - [6. Singing covers](#6-singing-covers)
 - [7. WhatsApp integration](#7-whatsapp-integration)
 - [8. Proactive messaging she texts first](#8-proactive-messaging--she-texts-first)
 - [9. Relationship progression](#9-relationship-progression)
 - [10. Her inner life (day simulation)](#10-her-inner-life-day-simulation)
 - [11. Two-brain architecture (local + cloud)](#11-two-brain-architecture-local--cloud)
 - [12. Local tools](#12-local-tools)
 - [13. Mood journal](#13-mood-journal)
 - [14. Stickers](#14-stickers)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Install on your phone (PWA)](#install-on-your-phone-pwa)
- [Configuration](#configuration)
- [Adding a persona](#adding-a-persona)
- [API reference](#api-reference)
- [VRAM budget (6 GB GPU)](#vram-budget-6-gb-gpu)
- [Testing & evals](#testing--evals)
- [Privacy & ethics](#privacy--ethics)

---

## Feature demos

> The transcripts below are representative examples of each feature in action.

### 1. WhatsApp-style chat

A dark-theme, mobile-first chat UI with everything you expect: chat bubbles,
timestamps, date separators, double-tick read receipts, emoji picker, a live
status line (`online` / `typing…` / `last seen today at 23:14`), and
human-like typing delays proportional to reply length.

```
┌──────────────────────────────────────────────┐
│ ◉ Nikki             📞 ⋮   │
│  typing…                  │
├──────────────────────────────────────────────┤
│        ── today ──          │
│                       │
│          ┌───────────────────────┐ │
│          │ just got home, so   │ │
│          │ tired 😩    23:02 ✓✓│ │
│          └───────────────────────┘ │
│ ┌────────────────────────┐          │
│ │ awww long day?? go   │          │
│ │ shower and come tell  │          │
│ │ me everything  23:02 │          │
│ └────────────────────────┘          │
│ ┌────────────────────────┐          │
│ │ i'll wait 🤍   23:02 │          │
│ └────────────────────────┘          │
├──────────────────────────────────────────────┤
│ 😊 [ Type a message…      ] 🎤 ➤  │
└──────────────────────────────────────────────┘
```

Replies stream token-by-token over SSE, and she often sends **a couple of
short texts instead of one paragraph** like a real person typing.

### 2. Long-term memory

After every exchange, a background task extracts durable facts (JSON-constrained
LLM pass), embeds them with `nomic-embed-text`, and stores them in SQLite +
ChromaDB. On every turn the most relevant memories (plus the freshest ones) are
retrieved (< 200 ms target) and injected into her context so she remembers
**across sessions, days, and channels**.

```
── Tuesday ─────────────────────────────────────
You:  interview at 10am tomorrow, wish me luck
Nikki: omg it's finally happening!! you're gonna
    be great. sleep early tonight ok?

── Wednesday (new session) ─────────────────────
Nikki: [proactive, 11:42] okay it's been long
    enough HOW DID THE INTERVIEW GO ??
```

Memories are deduplicated by embedding similarity (near-duplicates update the
existing fact instead of piling up) and are fully inspectable:

```bash
curl http://localhost:8000/memories        # list everything she knows
curl -X POST http://localhost:8000/memories \
 -H "Content-Type: application/json" \
 -d '{"fact": "User is allergic to peanuts", "category": "personal_info"}'
curl -X DELETE http://localhost:8000/memories/3  # make her forget
```

You can also view and delete memories from **Settings → Memories** in the app.

### 3. Voice notes (TTS) + speech input (STT)

- **Her voice notes** every reply can arrive as both a text bubble and a
 playable **voice-note bubble** (waveform, duration, play button). Synthesis
 is `Kokoro-82M` on CPU; emoji/markdown are stripped before synthesis so she
 never says "winking face" out loud.
- **Your voice** hold the mic button, talk, release: `faster-whisper`
 transcribes it (GPU if available, CPU fallback) and sends it as your message.

```
┌────────────────────────────────┐
│ ▶ ‖‖‖‖‖‖‖‖‖‖‖‖‖‖‖‖ 0:07   │  ← her reply as a voice note
└────────────────────────────────┘
```

Every synthesized chunk carries **per-word timing metadata**, which drives the
avatar's lip-sync in Call mode:

```json
{
 "source": "kokoro-word",
 "sample_rate": 24000,
 "audio_duration": 2.15,
 "units": [ { "t": 0.00, "d": 0.18, "s": "Hey" }, ... ]
}
```

### 4. Call mode hands-free voice calls

Tap 📞 and you get a full-screen, WhatsApp-video-call-style experience:

```
  ring… ring… → "heyy! i was literally just
          thinking about you. how did
          the interview go?"

  you talk ──► VAD detects you stopped ──► STT
  ──► LLM streams reply ──► sentence-by-sentence
  TTS ──► first audio in ~1.5 s
```

- **Hands-free** browser VAD detects when you start/stop speaking. No buttons.
- **Barge-in** interrupt her mid-sentence and playback stops instantly; a
 `cancel` aborts generation over the WebSocket, and she reacts naturally next
 turn ("oh sorry, go ahead").
- **Backchannel** pause mid-thought and she waits ~1.3 s and merges your
 fragments instead of answering half a sentence.
- **Animated 2D avatar** layered sprite with idle breathing, randomized
 blinking, **lip-sync** (6 mouth visemes driven by the TTS timing metadata),
 and **emotion reactions** (eye variants, head tilt, floating hearts). Personas
 without sprite art fall back to a photo with an audio-reactive glow.
- Ending the call drops a `📞 Call ended · 4:32` bubble into the chat and stores
 a one-line call summary as a memory.

### 5. Her own cloned voice (RVC + studio TTS)

Give her a consistent, unique voice built from voice recordings
(**with the informed consent of the voice's owner** see
[Privacy & ethics](#privacy--ethics)):

1. `tools/process_songs.py` Demucs-separates vocals from instruments.
2. `tools/prepare_rvc_dataset.py` slices recordings into a training dataset.
3. Train an RVC model headlessly (`tools/train_rvc_headless.py`, or via
  [Applio](https://github.com/IAHispano/Applio)'s GUI) full walkthrough with
  RTX 3060 settings in [docs/RVC_TRAINING.md](docs/RVC_TRAINING.md).
4. **Calls**: `voice.call_voice: kokoro_rvc` pipes every Kokoro chunk through
  the RVC layer at **~300 ms/chunk** on an RTX 3060 (a warm worker keeps the
  pitch model + FAISS index cached, and the client reuses one persistent HTTP
  connection). `kokoro_raw` is the instant rollback.
5. **Voice notes**: a studio TTS engine (XTTS-v2 or Chatterbox) clones from
  reference clips and picks a **per-emotion reference** matching the message's
  emotion metadata, with an optional "speakable" rewrite so the audio sounds
  spoken (the text bubble stays unchanged).

All renders share one GPU job queue (voice notes outrank covers), with VRAM
guards so a studio render can never starve the chat model. Consistency bench:
`POST /voice/bench` renders the same lines through both voices per emotion.

### 6. Singing covers

Drop any song into `songs/inbox/` and the pipeline splits the vocals, converts
them to her voice (auto pitch-shift into her range), remixes with light reverb,
and files it in `songs/library/` with mood tags.

```
You:  sing something for me?
Nikki: okay okay… i've been practicing this one 🎵
    [voice note 0:52]
```

Library hits play instantly; misses queue a render and **she delivers it
later herself** ("that song you wanted… i recorded it 🙈") she never
pretends to sing live on calls.

### 7. WhatsApp integration

A separate Node bridge ([whatsapp-bridge/](whatsapp-bridge/README.md)) links a
**dedicated second WhatsApp number** to the same brain:

- Your WhatsApp texts run through the same persona + memory pipeline the web
 app and WhatsApp share **one continuous history**.
- A configurable fraction of her replies arrive as **real WhatsApp voice notes**
 (Kokoro WAV → opus via ffmpeg).
- Incoming photos are captioned by a local vision model (`moondream`) so she can
 react to what you send.
- She **can't answer WhatsApp calls** the bridge auto-rejects, stores a memory
 of the missed call, and texts an in-character excuse pointing you to the
 app's Call mode.

> ⚠️ Uses `whatsapp-web.js`, an **unofficial client accounts can be banned.**
> Use a disposable second number, never your main one. Details in
> [whatsapp-bridge/README.md](whatsapp-bridge/README.md).

### 8. Proactive messaging she texts first

```
09:04 Nikki: good morning 🤍 dreamt something
       ridiculous, remind me to tell you
13:37 Nikki: lunch update? or are you being bad
       and skipping again
22:51 Nikki: okay you've been quiet all day.
       blink twice if you're alive
```

Configured per persona:

```yaml
proactive:
 enabled: true
 messages_per_day: "2-5"   # random within range each day
 active_hours: "08:00-23:00" # never messages outside this window
 clinginess: 0.6       # 0 = chill, 1 = full Nikki
 escalate_on_silence: true  # follow up if left on read
```

Texts are **grounded** the prompt includes the time of day, hours since you
last talked, your last message, and retrieved memories so they're specific
("how did the interview go??"), not generic. With high clinginess, being left
on read triggers a needier follow-up after 2–4 h (max 2), then she quietly
**sulks** stored as a memory she may bring up later. Pausable anytime from
**Settings → Her check-in texts**.

### 9. Relationship progression

She doesn't start as your girlfriend. She starts as a **stranger**:

```
stranger → acquaintance → friend → close → girlfriend
```

- Affection (0–100) moves slowly: each exchange is rated −2…+2 by the memory
 pass. Small talk is 0; deep conversations and calls move it.
- Promotion needs affection **AND** days known **AND** stored memories (e.g.
 girlfriend = 80 affection + 21 days + 45 memories) it can't be speedrun
 in one night.
- Each stage injects behavior rules: strangers are polite and guarded (no pet
 names), friends joke and remember details, close unlocks missing you, and
 **girlfriend** unlocks the full persona including "I love you", which she
 deflects sweetly at every earlier stage and never says first.
- Proactive frequency and sticker probability scale with the stage; stage
 changes are acknowledged naturally in her next reply.

Track it in **Settings → Relationship** (read-only affection meter, plus a
dev-only override endpoint for testing).

### 10. Her inner life (day simulation)

Persona YAML defines her `life:` occupation, weekday/weekend rhythm, friends,
recurring plans, ongoing threads. Each day a hidden **day state** is generated
(mood, energy, morning/afternoon/evening activities, one thing on her mind,
thread progress, occasional small events) and injected into **every channel**,
so she's never in two places at once.

```
You:  what are you up to?
Nikki: studio day 😮‍💨 we're shooting the campaign
    teasers and the light keeps dying on us.
    wednesday dinner with emma tho, so i'll
    survive. what about you?
```

Mood drifts within bounds based on conversation warmth; notable events become
memories she can reference weeks later. During busy slots her replies slow
down a little (`behavior.schedule_realism`) but urgent messages always get an
immediate real response. Dev view + reroll: **Settings → Her day**.

### 11. Two-brain architecture (local + cloud)

Nikki herself is **always local** persona, memory, everything personal. An
optional cloud "big brain" (Groq → Gemini → Cerebras → local-only fallback)
handles genuinely complex work, wrapped so its output is never shown raw:

```
You:  compare fixed vs variable home loan rates
    for 20yrs and just tell me which
Nikki: ok nerd hat on 🤓 give me a sec…
    [big brain runs, output rewritten in her voice]
Nikki: short version: fixed. you hate surprises
    and the spread's tiny right now. i did the
    math, wanna see it?
```

- **Router**: deterministic rules first (deep verbs, message length), then a
 constrained local classifier. **Mention vs request** is enforced everywhere 
 "i'm hungry" is conversation, "order me food" is a request.
- **Guards (code-enforced)**: replies claiming un-run actions get regenerated;
 price/news claims with no tool behind them are stripped; assistant-speak
 (bullet lists, "As an AI…", multi-questions) is banned outright.
- **Honest failure**: rate-limited → "gimme ~10 min" and the task is queued in
 SQLite for the proactive system to deliver later; all providers down →
 brain-fog lines. She never invents an answer for a failed task.
- **Budget-aware**: daily request/token usage is persisted against free-tier
 budgets; above 80% the routing threshold rises so casual chat stays local.
- **Auditable**: every outbound cloud payload is appended to
 `cloud_audit.jsonl`. Master kill switch: `brain.cloud_enabled: false`.

Dashboard: **Settings → Brain** (usage vs budget, provider status, routing
stats, guard hits, pending deferred tasks).

### 12. Local tools

Simple requests run fully locally with schema-validated argument extraction 
no cloud involved:

| Tool | Example |
|---|---|
| Reminders | "remind me to call mom at 6" → she texts you at 6 |
| Weather | "do I need an umbrella tomorrow?" |
| Todos | "add milk to my list" |
| Expenses | "note down 450 for lunch" |
| Events | "what do I have on friday?" |
| Vision | send a photo on WhatsApp → she reacts to what's in it |
| Singing | "sing for me" → covers library ([demo #6](#6-singing-covers)) |

### 13. Mood journal

A nightly, **local-only** pass reads the day's conversations and extracts your
mood grounded in concrete evidence, never invented on quiet days (a larger
local model, `llama3.1:8b`, runs this batch job since latency doesn't matter at
23:45). A weekly pass looks for patterns she can gently bring up
("you've been stressed every sunday night lately… work dread?").

### 14. Stickers

Drop WebP/PNG stickers into `stickers/<emotion>/` (`happy`, `laughing`, `shy`,
`love`, `sad`, `miss_you`, `good_morning`, `good_night`, `angry_cute`). After a
WhatsApp reply she may send one matching the reply's **emotion metadata** as
a real WhatsApp sticker. Probability scales with relationship stage (stranger
0% → girlfriend ~25%), occasionally the sticker *is* the reply, and never two
in a row. Placeholders included; see [stickers/README.md](stickers/README.md).

---

## Architecture

```
            ┌───────────────────────────────┐
    Phone (PWA)   │    FastAPI backend     │
  ┌────────────────┐  │                │   ┌─────────────┐
  │ React chat UI │◄──┤ /chat (SSE)  persona system  ├─────►│  Ollama  │
  │ Call screen  │◄──┤ /ws/call   memory (Chroma) │   │ llama3.2:3b │
  │ (VAD, avatar) │  │ /stt /tts   relationship   │   │ nomic-embed │
  └────────────────┘  │ scheduler   day simulation  │   │ moondream  │
            │ router+guards journal     │   └─────────────┘
  ┌────────────────┐  │                │   ┌─────────────┐
  │ WhatsApp bridge│◄─►┤ /whatsapp/*  SQLite: sessions,│─────►│ cloud brain │
  │ (Node, 2nd nr) │  │        messages,    │   │ (optional, │
  └────────────────┘  │        memories, state │   │ audited)  │
            └───────┬───────────────┬────────┘   └─────────────┘
                │        │
            ┌───────▼──────┐ ┌──────▼───────┐
            │ STT: whisper │ │ TTS: Kokoro │
            │ (GPU, int8) │ │ (+RVC/XTTS) │
            └──────────────┘ └──────────────┘
```

Everything personal chat history, memories, voice, persona lives in local
SQLite/ChromaDB files. The only thing that can leave the machine is an
explicitly routed, fully audited "big brain" request (and you can turn that off).

## Quick start

**Prerequisites**

1. **Python 3.10+** and **Node.js 18+**
2. **[Ollama](https://ollama.com)** running, with models pulled:
  ```bash
  ollama pull llama3.2:3b    # chat (or your preferred model see config.yaml)
  ollama pull nomic-embed-text  # memory embeddings
  ollama pull moondream     # optional: photo captioning
  ```
3. **espeak-ng** (needed by Kokoro TTS):
  Windows [releases](https://github.com/espeak-ng/espeak-ng/releases) ·
  macOS `brew install espeak-ng` · Linux `sudo apt-get install espeak-ng`
4. *(Optional)* CUDA-enabled PyTorch for GPU speech-to-text (auto-falls back to CPU)

**Install & run**

```bash
git clone https://github.com/Naresh23032003/Nikki-AI-Companion.git
cd Nikki-AI-Companion

# backend
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# frontend (built once, then served by FastAPI)
npm --prefix frontend install
npm --prefix frontend run build

# secrets
copy .env.example .env       # then fill in your keys (all optional)

# run everything on one server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** done. Health check:

```bash
curl http://localhost:8000/health
# {"status":"ok","ollama_reachable":true}
```

> First run downloads models: faster-whisper `small` (~0.5 GB) and Kokoro-82M.
> One-time delay of a minute or two.

Optional extras:

- **Voice cloning / covers**: `pip install -r requirements-voice.txt` +
 [docs/RVC_TRAINING.md](docs/RVC_TRAINING.md)
- **WhatsApp**: [whatsapp-bridge/README.md](whatsapp-bridge/README.md)
- **Frontend dev mode** (hot reload): `npm --prefix frontend run dev` → :5173,
 proxies API calls to :8000

## Install on your phone (PWA)

The server listens on `0.0.0.0`, so any device on your WiFi can reach it.

1. Find your machine's LAN IP (`ipconfig` / `ipconfig getifaddr en0` / `hostname -I`).
2. On your phone (same WiFi): open `http://<LAN_IP>:8000`.
3. **Android/Chrome**: ⋮ → *Add to Home screen* → *Install*.
  **iOS/Safari**: Share → *Add to Home Screen*.

She launches full-screen with her own icon indistinguishable from a real
messaging app. If the page doesn't load, allow port 8000 through your firewall:

```powershell
New-NetFirewallRule -DisplayName "Companion 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

> Mic features need a secure context they work on `localhost`; some browsers
> block the mic on plain-HTTP LAN IPs (use HTTPS or localhost if so).

## Configuration

Everything lives in [config.yaml](config.yaml) heavily commented. Highlights:

| Section | What it controls |
|---|---|
| `ollama` | Base URL, chat/embedding/extraction models, sampling, keep-alive |
| `persona` | Active persona + personas folder |
| `memory` | ChromaDB path, retrieval top-k, thresholds, dedup |
| `stt` / `tts` | Whisper size, Kokoro voice + pace |
| `voice` | Call voice (raw vs RVC), studio engine, VRAM guards, timeouts |
| `brain` | Cloud providers, daily budgets, audit log, master switch |
| `router` | Deep-routing verbs, length threshold |
| `behavior` | Eagerness, schedule realism, quiet hours |
| `journal` | Nightly mood extraction model + times |
| `whatsapp` | Bridge URL, voice-note ratio |

Secrets go in `.env` (see [.env.example](.env.example)) never in config.

## Adding a persona

Drop a YAML file in `personas/` ([personas/luna.yaml](personas/luna.yaml) is a
complete example):

```yaml
name: "Aria"
age: 26
personality: >
 Warm, playful, a little teasing…
speaking_style: >
 Casual and lowercase-leaning. Short, natural texts…
backstory: >
 …
relationship_context: >
 …
avatar_id: "aria_default"
profile_pic: "media/avatars/aria.png"
voice: "af_heart"     # Kokoro voice
life:           # feeds the daily state generator
 occupation: { ... }
 weekday: …
 friends: [ ... ]
 ongoing_threads: [ ... ]
proactive:
 enabled: true
 messages_per_day: "2-5"
```

Switch personas from **Settings → Persona** in the app. Set the profile photo
from Settings (uploads are stored as a DB override; the YAML default stays).
Avatar sprite specs for Call mode are in
[frontend/public/avatars/README.md](frontend/public/avatars/README.md).

## API reference

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/chat` | Stream a reply (SSE) |
| GET | `/persona` | Active persona info |
| GET / POST | `/personas`, `/personas/active` | List / switch personas |
| POST | `/persona/photo` | Upload profile photo |
| GET / DELETE | `/history/{session_id}` | Fetch / clear chat history |
| GET / POST / DELETE | `/memories` | List / add / delete memories |
| POST | `/stt` | Transcribe audio → `{text}` |
| POST | `/tts` | Synthesize text → `{audio_url, duration, timings}` |
| WS | `/ws/call` | Call mode: streamed TTS, barge-in |
| POST | `/call/end` | End call → event bubble + summary memory |
| POST | `/whatsapp/incoming` | Bridge → chat pipeline |
| GET / POST | `/proactive/status`, `/proactive/pause` | Scheduler state / pause |
| GET | `/brain/status` | Cloud usage, routing stats, guard hits |
| POST | `/voice/bench` | Voice consistency benchmark |
| GET | `/health` | Liveness + Ollama reachability |

<details>
<summary><code>/chat</code> SSE example</summary>

```bash
curl -N -X POST http://localhost:8000/chat \
 -H "Content-Type: application/json" \
 -d '{"message": "hi!", "session_id": "abc"}'
```

```
event: token
data: {"token": "hey"}

event: token
data: {"token": " you"}

event: done
data: {"done": true}
```
</details>

## VRAM budget (6 GB GPU)

The whole pipeline is arranged so chat + speech coexist on one 6 GB card:

| Component | Where | Approx. VRAM |
|---|---|---|
| `llama3.2:3b` (Ollama, 4-bit) | GPU | ~3.0–3.5 GB |
| `faster-whisper` small, int8 | GPU | ~0.5–1.0 GB |
| `nomic-embed-text` | GPU | ~0.3 GB (transient) |
| **Kokoro-82M TTS** | **CPU** | **0 GB** |
| **Total on GPU** | | **~4–5 GB** |

Keeping TTS on CPU is the key decision it frees VRAM for Whisper + the LLM.
Tight on VRAM? Drop Whisper to `base`/`tiny` in `config.yaml` (or let it
auto-fall-back to CPU). VRAM guards prevent the studio voice engine from ever
starving the chat model.

## Testing & evals

```bash
python test_chat.py        # end-to-end memory recall across sessions
python test_backend.py      # backend smoke test
python tests/routing_eval.py   # 50 routing + 10 tone cases (run after any model/prompt change)
python tests/extraction_eval.py  # memory extraction quality
python tools/bench_tools.py    # tool latency benchmarks
```

## Privacy & ethics

- **Local-first by design.** Chat history, memories, voice, persona all in
 local files. The optional cloud brain is off-switchable
 (`brain.cloud_enabled: false`) and every outbound payload is logged to
 `cloud_audit.jsonl` so you can audit exactly what left your machine.
- **Voice cloning requires informed consent** from the voice's owner, and the
 output is for **personal use only** never distribute renders or covers, and
 never present generated audio as real recordings. If consent is withdrawn,
 `python cleanup_voice.py` permanently deletes every recording, dataset,
 model, and rendered file derived from that voice.
- **No personal data ships with this repo** voice datasets, trained models,
 databases, logs, and WhatsApp sessions are all gitignored.
- **WhatsApp**: unofficial client, use a disposable second number, keep it
 personal-scale.

---

*Built for one user, one GPU, and zero subscriptions.*
