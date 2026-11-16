"""FastAPI application: the companion chat backend + static SPA host.

Serves the built React (Vite) frontend as static files so the whole app runs
from one server on the LAN.

API:
  POST   /chat                 -> streams the reply token-by-token (SSE); after
                                  the exchange, a background task extracts and
                                  stores durable long-term memories.
  GET    /persona              -> active persona info (name, avatar, photo url)
  GET    /personas             -> all personas + which one is active
  POST   /personas/active      -> switch the active persona {id}
  GET    /personas/{id}/photo  -> a persona's static profile photo
  POST   /persona/photo        -> upload a new profile photo for the active persona
  GET    /history/{session_id} -> full stored history
  DELETE /history/{session_id} -> clear a session's chat
  GET    /memories             -> list all long-term memories
  POST   /memories             -> manually add a memory
  DELETE /memories/{id}        -> delete a memory
  POST   /stt                  -> transcribe a webm/opus audio blob to text
  POST   /tts                  -> synthesize text to a WAV voice note (+timings)
  WS     /ws/call              -> Call mode: streamed sentence-by-sentence TTS
                                  with barge-in cancel
  GET    /health               -> liveness + Ollama reachability
  /media/*                     -> static media (tts wavs, avatars)
Everything else -> the static frontend (index.html / assets).
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import io
import json
import logging
import random
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ROOT, load_settings
from app.conversation.bubbles import sentence_bubbles, split_bubbles
from app.conversation.planner import (
    TurnSignals,
    plan_turn,
    render as render_turn_plan,
    summarise as summarise_turn_plan,
)
from app.conversation.tells import TurnContext, tell_names
from app.conversation.notes import (
    NoteContext,
    awaiting_followup_note,
    is_gibberish,
    nonsense_note,
    offer_note,
    pattern_note,
    persona_other_names,
    streak_note,
    track_offer_decline,
    unresolved_note,
)
from app.dayseed import DayLife
from app.db import Database
from app.api import graph as graph_routes
from app.api import history as history_routes
from app.api import journal as journal_routes
from app.api import memories as memories_routes
from app.api import personas as personas_routes
from app.api import privacy as privacy_routes
from app.api import proactive as proactive_routes
from app.api import relationship as relationship_routes
from app.api import status as status_routes
from app.api import voice as voice_routes
from app.api.voice import _training_progress
from app.deps import P, _current_profile, db_for_session, persona_voice, state
from app.timing import hhmm, in_quiet_hours, now_context, sse
from app.guards import (
    CAPABILITY_MANIFEST,
    CLAIM_PATTERNS,
    HONEST_LINE,
    REFUSAL_DEFLECT,
    REFUSAL_PATTERNS,
    STATUS_PATTERNS,
    scan_assistant_speak,
    scan_forbidden_claims,
    scan_honeypots,
    scan_identity_confusion,
    scan_reaction,
    scan_refusal,
    strip_violating_sentences,
)
from app.covers import CoverPipeline
from app.gpu_queue import PRIORITY_VOICE_NOTE, GPUJobQueue
from app.journal import (
    run_nightly_extraction,
    run_recent_streak_check,
    run_unresolved_check,
    run_weekly_patterns,
)
from app.providers import BrainUnavailable, CloudBrain
from app.router import Router
from app.rvc_layer import RVCConverter
from app.tools import ToolContext, ToolRunner
from app.voice_studio import StudioTTS
from app.emotion import (
    EMOTION_TAG_INSTRUCTION,
    parse_emotion,
    strip_for_speech,
    strip_tags,
)
from app.llm import OllamaClient
from app.commands import handle as handle_command, is_command, mood_note
from app.memory import MemoryStore
from app.persona import build_system_prompt, list_persona_ids, load_persona
from app.profiles import Profile, ProfileRegistry, load_profiles
from app.relationship import RelationshipTracker
from app.stickers import STICKER_ROOT, ensure_dirs as ensure_sticker_dirs, pick_sticker
from app.stt import STTEngine
from app.telephony import (
    TELEPHONY_SAMPLE_RATE,
    TwilioStreamBuffer,
    get_telephony_provider,
    pcm16_to_ulaw,
    resample_linear,
    ulaw_to_pcm16,
)
from app.tts import SentenceAccumulator, TTSEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("companion")
# Rotating file log alongside console output - 5MB x 3 so it can't grow
# unbounded, and there's history to inspect after a crash even when the
# console scrolled away or the terminal was closed.
try:
    from logging.handlers import RotatingFileHandler

    _fh = RotatingFileHandler(ROOT / "companion.log", maxBytes=5_000_000,
                              backupCount=3, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    logging.getLogger().addHandler(_fh)
except OSError:  # log file locked/unwritable - console-only is fine
    pass
# APScheduler logs "Running job ..." / "... executed successfully" at INFO
# for EVERY tick of EVERY interval job (reminders every 60s, deferred every
# 90s, covers every 300s, studio-unload every 120s, ...) - pure noise for a
# personal app; none of that per-tick chatter is actionable. Errors/warnings
# (a job actually failing) still come through.
logging.getLogger("apscheduler").setLevel(logging.WARNING)

FRONTEND_DIST = ROOT / "frontend" / "dist"
MEDIA_DIR = ROOT / "media"
AVATAR_DIR = MEDIA_DIR / "avatars"
TTS_DIR = MEDIA_DIR / "tts"
DEFAULT_AVATAR = AVATAR_DIR / "luna.svg"

# Origins on the local network (so the PWA works from a phone on the same WiFi).
LAN_ORIGIN_REGEX = (
    r"^http://("
    r"localhost|127\.0\.0\.1|"
    r"192\.168\.\d{1,3}\.\d{1,3}|"
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r")(:\d+)?$"
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)




class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    message_id: int | None = None  # attach the WAV url to this stored message
    voice: str | None = None       # override the persona's voice
    emotion: str | None = None     # picks ref/<emotion> clip / studio preset


# ---------------------------------------------------------------------------
# App state / lifespan
# ---------------------------------------------------------------------------



# The profile (persona + its isolated db/memory/relationship) this request
# belongs to. WhatsApp sets it per incoming message from the sender's number;
# everything else (web app, calls) leaves it unset and gets the default
# profile, which IS state.db/state.persona - i.e. unchanged behavior.


# P(), state and AppState now live in app/deps.py (N2 phase 2) so route
# modules can import them without a cycle back through main.


def _persona_voice() -> str | None:
    return persona_voice()


def _build_profiles(settings) -> ProfileRegistry:
    """Wire each configured profile with its OWN isolated stack.

    The default profile reuses the already-built singletons (state.db /
    state.persona / state.memory / ...) so the existing database, vector
    collection, web app and call paths behave exactly as before. Every other
    profile gets a separate SQLite file and Chroma collection - two people
    talking to the same WhatsApp account can never see each other's messages,
    memories, affection or upset state.
    """
    registry = load_profiles(settings)
    for p in registry:
        if p.is_default:
            p.db = state.db
            p.persona = state.persona
            p.memory = state.memory
            p.relationship = state.relationship
            p.daylife = state.daylife
            p.tools = state.tools
            p.tool_ctx = state.tool_ctx
            continue
        try:
            p.db = Database(p.db_path)
            p.persona = load_persona(settings.persona_folder, p.persona_id)
            p.memory = MemoryStore(p.db, state.llm, settings,
                                   collection=p.collection, session_id=p.session_id)
            p.relationship = RelationshipTracker(p.db, p.memory)
            p.memory.relationship = p.relationship
            p.memory.blocked_names = {p.persona.name.lower()}
            p.daylife = DayLife(p.db, state.llm, lambda p=p: p.persona, memory=p.memory)
            p.memory.daylife = p.daylife
            p.tool_ctx = ToolContext(db=p.db, llm=state.llm, settings=settings,
                                     relationship=p.relationship,
                                     deliver=_deliver_message, persona=p.persona)
            p.tools = ToolRunner(p.tool_ctx)
            p.tools.covers = state.covers  # song library is shared, not personal
        except Exception as e:  # noqa: BLE001 - one bad profile must not stop boot
            logger.exception("profiles: failed to build %r (%s) - it will be ignored", p.id, e)
            p.db = None
    registry.profiles = [p for p in registry if p.db is not None]
    logger.info("profiles: %s", ", ".join(
        f"{p.id}->{p.persona.name}(…{p.number[-4:] or '-'}, {p.db_path.name})"
        for p in registry))
    return registry


def _reload_profile_persona(p: Profile, persona_id: str) -> None:
    """Swap which persona answers a profile's number (the /persona command)."""
    persona = load_persona(state.settings.persona_folder, persona_id)
    p.persona = persona
    p.persona_id = persona_id
    p.memory.blocked_names = {persona.name.lower()}
    if p.tool_ctx:
        p.tool_ctx.persona = persona
    if p.proactive:
        # plan_day() reads get_persona() - the new persona's schedule/clinginess
        # takes over from the next planning pass.
        try:
            p.proactive.plan_day()
        except Exception as e:  # noqa: BLE001
            logger.warning("profiles: replan after persona switch failed: %s", e)
    if p.is_default:
        state.persona = persona
        state.db.set_setting("active_persona", persona_id)
    p.db.set_setting("profile_persona", persona_id)
    logger.warning("profiles: %s persona -> %s", p.id, persona_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    state.settings = settings
    state.db = Database(settings.db_path)

    # Active persona: runtime override (Settings panel) wins over config.yaml.
    active = state.db.get_setting("active_persona") or settings.persona_active
    try:
        state.persona = load_persona(settings.persona_folder, active)
    except FileNotFoundError:
        state.persona = load_persona(settings.persona_folder, settings.persona_active)

    state.llm = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        embed_model=settings.ollama_embed_model,
        options=settings.ollama_options,
        keep_alive=settings.ollama_keep_alive,
    )
    state.memory = MemoryStore(state.db, state.llm, settings)
    # Relationship progression; extraction feeds affection deltas into it.
    state.relationship = RelationshipTracker(state.db, state.memory)
    state.memory.relationship = state.relationship
    # Her own name must never become one of the user's graph entities.
    state.memory.blocked_names = {state.persona.name.lower()}
    ensure_sticker_dirs()
    # Voice engines are lazy - constructing them loads no model / needs no GPU.
    state.stt = STTEngine(settings.stt_model_size)
    state.tts = TTSEngine(
        settings.tts_default_voice, settings.tts_lang_code, settings.tts_speed
    )
    TTS_DIR.mkdir(parents=True, exist_ok=True)

    # Two-brain architecture + tools + her inner life.
    state.brain = CloudBrain(state.db, settings)
    state.router = Router(state.llm, settings, brain=state.brain)
    state.router.db = state.db
    state.tool_ctx = ToolContext(db=state.db, llm=state.llm, settings=settings,
                                 relationship=state.relationship, deliver=_deliver_message,
                                 persona=state.persona)
    state.tools = ToolRunner(state.tool_ctx)
    state.daylife = DayLife(state.db, state.llm, lambda: state.persona,
                            memory=state.memory)
    state.memory.daylife = state.daylife

    # Voice system: RVC layer (calls/covers), studio TTS, covers, GPU queue.
    vcfg = (settings.raw or {}).get("voice", {})
    state.rvc = RVCConverter(ROOT / vcfg.get("rvc_model_dir", "voices/rvc/nikki"))
    state.studio = StudioTTS(state.llm, settings)
    state.gpu_queue = GPUJobQueue()
    state.gpu_queue.start()
    # Studio renders fall back to CPU while RVC training owns the GPU, OR while
    # a call is active (so a mid-call voice note doesn't re-grab the VRAM the
    # call's RVC needs).
    state.studio.prefer_cpu = lambda: bool(_active_calls) or bool(
        (p := _training_progress()) and not p.get("done") and not p.get("failed"))
    state.covers = CoverPipeline(state.rvc, settings, state.gpu_queue)
    state.tools.covers = state.covers
    # Multi-persona: one fully isolated world per WhatsApp number. Built here
    # (after tools/covers exist) so every profile's tool runner can sing/draw.
    # The default profile REUSES the objects above - same db file, same
    # collection - so existing history and every non-WhatsApp path are unchanged.
    state.profiles = _build_profiles(settings)
    logger.info("voice: call_voice=%s | rvc=%s | studio=%s(%s)",
                vcfg.get("call_voice", "kokoro_raw"),
                state.rvc.status_label(),
                state.studio.engine_name,
                "installed" if state.studio.available else "not installed")

    # Proactive scheduler (she texts first). Failure here must never block chat.
    try:
        from app.proactive import ProactiveEngine

        state.proactive = ProactiveEngine(
            db=state.db,
            llm=state.llm,
            memory=state.memory,
            get_persona=lambda: state.persona,
            settings=settings,
            relationship=state.relationship,
            tools=state.tools,
            session_id=state.profiles.default.session_id,
            number=state.profiles.default.number or None,
        )
        state.proactive.covers = state.covers  # rare unprompted song drops
        state.proactive.start()
        # Background workers on the same scheduler: due reminders + deferred
        # (rate-limited / failed cloud) tasks, checked every minute.
        state.proactive.scheduler.add_job(
            _deliver_due_reminders, "interval", seconds=60,
            id="reminders", replace_existing=True)
        state.proactive.scheduler.add_job(
            _retry_deferred_tasks, "interval", seconds=90,
            id="deferred", replace_existing=True)
        # Care check-ins around dated events (good luck before / how'd it go
        # after) - the event_followups table existed but nothing polled it.
        state.proactive.scheduler.add_job(
            _deliver_due_encouragements, "interval", seconds=60,
            id="event_encouragements", replace_existing=True)
        state.proactive.scheduler.add_job(
            _deliver_due_followups, "interval", seconds=60,
            id="event_followups", replace_existing=True)
        # Covers inbox scan + studio idle unload.
        state.proactive.scheduler.add_job(
            _scan_cover_inbox, "interval", seconds=300,
            id="covers", replace_existing=True)
        state.proactive.scheduler.add_job(
            _studio_idle_unload,
            "interval", seconds=120, id="studio_unload", replace_existing=True)
        # Passive mood journal: nightly extraction + weekly pattern pass.
        from apscheduler.triggers.cron import CronTrigger as _CronTrigger

        jcfg = (settings.raw or {}).get("journal", {})
        if jcfg.get("enabled", True):
            nh, nm = _hhmm(jcfg.get("nightly_time", "23:45"))
            state.proactive.scheduler.add_job(
                _run_nightly_journal, _CronTrigger(hour=nh, minute=nm),
                id="mood_journal_nightly", replace_existing=True)
            wh, wm = _hhmm(jcfg.get("weekly_pattern_time", "23:55"))
            state.proactive.scheduler.add_job(
                _run_weekly_journal_patterns,
                _CronTrigger(day_of_week=jcfg.get("weekly_pattern_day", "sun"), hour=wh, minute=wm),
                id="mood_journal_weekly", replace_existing=True)
        # Nightly backup of companion.db + chroma_db (04:10, quiet hours).
        state.proactive.scheduler.add_job(
            _nightly_backup, _CronTrigger(hour=4, minute=10),
            id="nightly_backup", replace_existing=True)
        # The default profile IS state.proactive (all the shared background
        # jobs above hang off its scheduler). Every other profile gets its own
        # engine so she texts HER person on HER schedule, from her own state.
        for p in state.profiles:
            if p.is_default:
                p.proactive = state.proactive
                continue
            try:
                engine = ProactiveEngine(
                    db=p.db, llm=state.llm, memory=p.memory,
                    get_persona=lambda p=p: p.persona, settings=settings,
                    relationship=p.relationship, tools=p.tools,
                    session_id=p.session_id, number=p.number or None,
                )
                engine.covers = state.covers
                engine.start()
                p.proactive = engine
                logger.info("proactive: started for profile %s (%s)", p.id, p.persona.name)
            except Exception as e:  # noqa: BLE001 - one profile must not kill the rest
                logger.warning("proactive: profile %s failed to start: %s", p.id, e)
    except Exception as e:  # noqa: BLE001
        logger.warning("Proactive scheduler failed to start: %s", e)
    logger.info(
        "Loaded persona '%s' | model '%s' | embed '%s' @ %s",
        state.persona.name,
        settings.ollama_model,
        settings.ollama_embed_model,
        settings.ollama_base_url,
    )
    # Warm both Ollama models NOW so the first real message doesn't pay the
    # cold-load (the 1.3s "retrieve_memories SLOW" spikes were exactly this).
    _spawn(_warmup_ollama())
    if not _AUTH_TOKEN:
        logger.warning("LAN auth DISABLED (no COMPANION_AUTH_TOKEN in .env) - "
                       "anyone on this WiFi can reach the API")
    try:
        yield
    finally:
        if state.proactive:
            state.proactive.stop()
        # Non-default profiles own their own engine + SQLite handle.
        for p in (state.profiles or []):
            if p.is_default:
                continue
            if p.proactive:
                p.proactive.stop()
            if p.db:
                p.db.close()
        if state.rvc:
            state.rvc.close()
        if state.llm:
            await state.llm.close()
        if state.db:
            state.db.close()


app = FastAPI(title="Local AI Companion", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=LAN_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CRUD route groups extracted from this module (N2 phase 2). Included before
# the static-file mounts below, which claim "/" and would otherwise shadow them.
app.include_router(memories_routes.router)
app.include_router(journal_routes.router)
app.include_router(graph_routes.router)
app.include_router(history_routes.router)
app.include_router(personas_routes.router)
app.include_router(privacy_routes.router)
app.include_router(proactive_routes.router)
app.include_router(relationship_routes.router)
app.include_router(status_routes.router)
app.include_router(voice_routes.router)


# ---------------------------------------------------------------------------
# LAN auth: the server binds 0.0.0.0 so phone/tablet PWAs can reach it, which
# also means anyone on the WiFi could. If COMPANION_AUTH_TOKEN is set in .env,
# non-localhost clients must present it (X-Auth-Token header, or ?token= for
# WebSockets/media tags). Localhost is always exempt; unset token = open LAN
# (previous behavior) with a startup warning.
# ---------------------------------------------------------------------------
import os as _os  # noqa: E402  (intentional late import, keeps auth block self-contained)

_AUTH_TOKEN = _os.environ.get("COMPANION_AUTH_TOKEN", "").strip()
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _request_authed(client_host: str | None, token: str | None) -> bool:
    if not _AUTH_TOKEN:
        return True
    if client_host in _LOCAL_HOSTS:
        return True
    # Constant-time compare - a plain `==` short-circuits on the first
    # mismatched byte, which leaks timing information about the token.
    return bool(token) and hmac.compare_digest(token, _AUTH_TOKEN)


@app.middleware("http")
async def _lan_auth(request, call_next):
    token = request.headers.get("x-auth-token") or request.query_params.get("token")
    if not _request_authed(request.client.host if request.client else None, token):
        from fastapi.responses import JSONResponse as _JR
        return _JR({"detail": "missing or invalid auth token"}, status_code=401)
    return await call_next(request)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Extracted to app/timing.py (N2). Kept as module-level names so existing call
# sites are unchanged; only _in_quiet_hours needs a wrapper, to read the
# configured window out of app state rather than reaching for it internally.
_sse = sse
_now_context = now_context
_hhmm = hhmm


def _in_quiet_hours(now: datetime | None = None) -> bool:
    """behavior.quiet_hours ('01:00-07:30' style) - she's 'asleep'. Used to
    hold self-initiated deliveries (reminders, deferred answers, event
    follow-ups) until it's over instead of firing at 3am; the item stays
    queued and fires on the next scheduler tick after the window ends."""
    return in_quiet_hours((state.settings.behavior or {}).get("quiet_hours"), now)


# Fire-and-forget background tasks must be referenced or Python may GC them
# mid-run (this is why call-mode memory extraction silently vanished).
_BG_TASKS: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)


def _nightly_backup() -> None:
    """Snapshot companion.db (SQLite backup API) + chroma_db (copy) into
    backups/<date>/, keeping the 7 most recent. Runs at 04:10 (quiet hours)."""
    import shutil
    from datetime import date

    backups = ROOT / "backups"
    dest = backups / date.today().isoformat()
    try:
        state.db.backup_to(dest / "companion.db")
        chroma_src = state.settings.chroma_path
        if chroma_src.exists():
            shutil.copytree(chroma_src, dest / "chroma_db", dirs_exist_ok=True)
        kept = sorted(d for d in backups.iterdir() if d.is_dir())
        for old in kept[:-7]:
            shutil.rmtree(old, ignore_errors=True)
        logger.info("backup: %s written (%d kept)", dest.name, min(len(kept), 7))
    except Exception:  # noqa: BLE001 - a failed backup must never crash the app
        logger.exception("backup failed")


async def _warmup_ollama() -> None:
    """Load the chat + embed models into VRAM at boot (background task)."""
    try:
        t0 = asyncio.get_event_loop().time()
        await state.llm.embed("warmup")
        await state.llm.chat(messages=[{"role": "user", "content": "hi"}],
                             options={"num_predict": 1})
        logger.info("ollama warmup done in %.1fs (models resident, keep_alive=%s)",
                    asyncio.get_event_loop().time() - t0,
                    state.settings.ollama_keep_alive)
    except Exception as e:  # noqa: BLE001 - warmup is best-effort
        logger.warning("ollama warmup failed: %s", e)


def _sanitized_recent(session_id: str) -> list[dict]:
    """Recent messages for LLM context, with any emotion tags scrubbed.

    Prevents the model from imitating stray `{"emotion": ...}` tags that older
    replies may have left in history.
    """
    recent = P().db.get_recent_messages(session_id, state.settings.max_messages)
    for m in recent:
        if m["role"] == "assistant":
            m["content"] = strip_tags(m["content"])
    return recent


def _stage() -> str | None:
    rel = P().relationship if P() else state.relationship
    return rel.stage if rel else None


# Stage-scaled bedtime realism (behavior.quiet_hours window). She ALWAYS
# replies, at any hour - quiet hours only gate self-initiated messages
# (proactive/reminders/deferred). What changes with closeness is the
# BOUNDARY: a real person winds down a 1am chat with someone she just met,
# hangs around sleepily for a friend, and happily loses sleep for her
# person. The boundary erodes stage by stage until it's gone at girlfriend.
_BEDTIME_NOTES = {
    "stranger": (
        "It is deep into your night and you were about to sleep. You still "
        "reply (you're on your phone), but you barely know this person: keep "
        "replies brief and lower-energy, and politely wind the conversation "
        "down soon ('i should really sleep haha - talk tomorrow?'). You don't "
        "stay up late for someone you just met."),
    "acquaintance": (
        "It's really late and you're sleepy. You reply, but shorter and "
        "lower-energy than usual; if the conversation keeps going, mention "
        "you need to sleep soon and wrap up warmly."),
    "friend": (
        "It's late and you're sleepy - yawny, relaxed, lowercase energy. You "
        "hang around a while because you like talking to them, but you might "
        "tease that they're keeping you up, and you'll eventually call it."),
    "close": (
        "It's late; you're sleepy but you honestly don't mind losing a bit of "
        "sleep for them - soft, cozy late-night energy. Tease it a little "
        "('you're lucky i like you, i should be asleep 😴')."),
    "girlfriend": (
        "It's the middle of the night and you don't care - talking to them "
        "beats sleeping. Soft, intimate, sleepy late-night energy; you're not "
        "going anywhere unless they say goodnight first."),
}


def _bedtime_note() -> str | None:
    if not _in_quiet_hours():
        return None
    return _BEDTIME_NOTES.get(_stage() or "stranger")


def _relationship_notes(*extra: str | None) -> str | None:
    """Stage behavior rules + friction/upset override + one-shot stage-change
    ack + bedtime realism + any extra notes. Every reply channel (web,
    WhatsApp, calls, reminders, event check-ins, deferred) routes through
    here, so bedtime AND friction apply everywhere without per-channel
    wiring."""
    parts: list[str] = []
    rel = (P().relationship if P() else None) or state.relationship
    if rel:
        parts.append(rel.addendum())
        upset = rel.upset_addendum()
        if upset:
            parts.append(upset)
        ack = rel.consume_ack_note()
        if ack:
            parts.append(ack)
    bedtime = _bedtime_note()
    if bedtime:
        parts.append(bedtime)
    parts.extend(p for p in extra if p)
    return "\n\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Chat pipeline: route -> (tool | big brain | plain chat) -> guarded persona reply
# ---------------------------------------------------------------------------
_BRAIN_FOG = [
    "ugh my head's so foggy today 😵‍💫 gimme a bit and i'll get back to you on that?",
    "okay my brain is NOT braining right now 😩 lemme come back to that in a bit",
    "hmm i can't think straight rn - give me a little while on that one?",
]
# Reply-generation itself failed (LLM timeout under GPU load etc.) - she
# blames her phone instead of going silent. Honest in spirit: her "phone"
# (this box) genuinely IS lagging.
_LAGGY_PHONE = [
    "okayyy my phone is being SO slow rn 😭 gimme a min",
    "ugh sorry, my phone is literally lagging so bad - one sec",
    "my phone's freaking out rn 😩 hold on, i'm here though",
]
_URGENT = re.compile(r"\b(urgent|emergency|help me|asap|right now|crying|scared|"
                     r"panic|hurt|hospital)\b|!{2,}", re.I)


async def _build_prompt(session_id: str, query: str, tool_note: str | None = None,
                        mode: str = "chat", memories: list | None = None,
                        extra_note: str | None = None) -> list[dict]:
    """System prompt + sanitized history, with day-state, manifest and notes.

    Pass `memories` when retrieval was already started concurrently with
    routing (the /chat hot path) - otherwise this retrieves serially."""
    if memories is None:
        memories = await state.memory.retrieve_memories(query)
    day_note = None
    try:
        day_note = await state.daylife.prompt_note()
    except Exception as e:  # noqa: BLE001
        logger.warning("day note failed: %s", e)
    # N3: decide the SHAPE of this turn (length, ask-or-not, self-disclosure,
    # memory surfacing, disagreement) before generation, as ONE coherent
    # directive - rather than each heuristic note below silently competing to
    # steer the same decision. The notes still contribute situational content
    # (offers, patterns, streaks); the planner decides the turn's shape and
    # folds them in as supporting lines via render()'s extra_notes.
    turn_directive = _turn_directive(session_id, query, memories, mode)
    system_prompt = build_system_prompt(
        state.persona, memories, mode=mode,
        current_time=_now_context(),
        extra_notes=_relationship_notes(CAPABILITY_MANIFEST, day_note,
                                        turn_directive,
                                        _offer_note(query), _pattern_note(query),
                                        _streak_note(query), _unresolved_note(query),
                                        tool_note, extra_note),
        stage=_stage(),
    )
    return [{"role": "system", "content": system_prompt},
            *_sanitized_recent(session_id)]


def _recent_assistant_replies(session_id: str, limit: int = 4) -> list[str]:
    """Her own last few replies - the planner and tell-detector need these to
    avoid repeating the same conversational move (asking a question every
    single turn, or restating the same content twice in a row)."""
    try:
        history = _sanitized_recent(session_id)
    except Exception as e:  # noqa: BLE001 - never break a reply over this
        logger.warning("turn planner: history unavailable: %s", e)
        return []
    return [m.get("content", "") for m in history
            if m.get("role") == "assistant"][-limit:]


def _turn_directive(session_id: str, query: str, memories: list | None,
                    mode: str) -> str | None:
    """Plan this turn and render it. Returns None if planning fails.

    Never allowed to raise: a planning failure must degrade to the previous
    behaviour (persona prompt + notes only), not lose the user's message.
    """
    try:
        db = P().db
        signals = TurnSignals(
            user_message=query,
            recent_replies=tuple(_recent_assistant_replies(session_id)),
            turns_since_self_share=int(db.get_setting("turns_since_self_share") or 99),
            turns_since_memory_surfaced=int(
                db.get_setting("turns_since_memory_surfaced") or 99),
            stage=_stage() or "close",
            # The top-ranked memory only; the planner decides whether it is
            # surfaced at all, which is what stops her reciting facts back.
            relevant_memory=(memories[0] if memories else None),
            rng=random,
        )
        plan = plan_turn(signals)

        db.set_setting("turns_since_self_share",
                       "0" if plan.share_self
                       else str(signals.turns_since_self_share + 1))
        db.set_setting("turns_since_memory_surfaced",
                       "0" if plan.surface_memory
                       else str(signals.turns_since_memory_surfaced + 1))

        logger.info("%s reasons=%s", summarise_turn_plan(plan), "; ".join(plan.reasons))
        return render_turn_plan(plan)
    except Exception as e:  # noqa: BLE001 - degrade, never drop the turn
        logger.warning("turn planning failed, falling back to notes only: %s", e)
        return None


# ---------------------------------------------------------------------------
# Prompt notes — logic lives in app/conversation/notes.py (N2 extraction).
# These wrappers build a NoteContext from app state so every call site below
# is unchanged. The notes themselves are now unit-tested; they were not before,
# because they reached for P(), state.settings and the global RNG.
# ---------------------------------------------------------------------------


def _note_ctx() -> NoteContext:
    return NoteContext(
        db=P().db,
        behavior=state.settings.behavior or {},
        stage=_stage() or "stranger",
        rng=random,
    )


def _awaiting_followup_note(session_id: str) -> str | None:
    return awaiting_followup_note(_note_ctx(), session_id)


async def _maybe_repair_note(message: str) -> str | None:
    """If she's currently upset (see relationship.py's friction system) and
    this message reads as a sincere apology, clear it and return a one-shot
    warm-relief note for THIS reply."""
    rel = (P().relationship if P() else None) or state.relationship
    if not rel:
        return None
    try:
        if await rel.maybe_repair(message):
            return rel.consume_repair_note()
    except Exception as e:  # noqa: BLE001 - never let this break a reply
        logger.warning("repair check failed: %s", e)
    return None


def _offer_note(message: str) -> str | None:
    return offer_note(_note_ctx(), message)


def _pattern_note(message: str) -> str | None:
    return pattern_note(_note_ctx(), message)


def _streak_note(message: str) -> str | None:
    # on_consume drops the vector for the one-shot streak memory we just deleted.
    return streak_note(_note_ctx(), message,
                       on_consume=lambda mid: P().memory.remove(mid))


def _unresolved_note(message: str) -> str | None:
    # on_consume drops the vector for the one-shot unresolved memory we just deleted.
    return unresolved_note(_note_ctx(), message,
                           on_consume=lambda mid: P().memory.remove(mid))


# ---------------------------------------------------------------------------
# Nonsense/spam realism: 30x "rrrrrr" used to make her confabulate an entire
# fake evening (invented plans, times, random memory fragments) because the
# model had no signal the input was noise. A real person notices immediately.
# ---------------------------------------------------------------------------
# Real texting tokens that survive collapsing but have no vowel (or are
# single letters) - must never count as keyboard-mash.


def _is_gibberish(text: str) -> bool:
    return is_gibberish(text)


def _nonsense_note(message: str) -> str | None:
    return nonsense_note(_note_ctx(), message)


def _track_offer_decline(message: str) -> None:
    track_offer_decline(_note_ctx(), message)


def _persona_other_names(persona) -> list[str]:
    return persona_other_names(persona)


async def _guarded_reply(messages: list[dict], tool_ran: bool,
                         session_id: str | None = None,
                         user_message: str | None = None) -> tuple[str, str]:
    """Generate fully, run the guards, regenerate once on violation, then
    surgically fix anything left. Nothing reaches the user unvetted.

    `session_id`/`user_message` are optional and enable the N3 AI-tell guard
    (reflexive questioning, over-agreement, generic follow-ups, unearned
    enthusiasm, robotic sentence rhythm...) on top of the safety/character
    guards below. Omit them (as the pre-N3 call sites do) to skip tell
    detection without changing any other behaviour.

    Returns (reply_text, emotion) - parse_emotion() is a strict superset of
    strip_tags() (it returns the same cleaned text plus the trailing emotion
    tag's value), so this covers both the web-chat caller (which ignores the
    emotion) and WhatsApp/deferred callers that need it for sticker choice,
    without WhatsApp having to duplicate the whole scan-retry-fix flow itself.
    """
    raw = await state.llm.chat(messages)
    reply, emotion = parse_emotion(raw)
    profile = P()
    gdb = profile.db if profile else state.db
    persona = profile.persona if profile else state.persona
    other_names = _persona_other_names(persona)
    claims = scan_forbidden_claims(reply, tool_ran, gdb)
    speak = scan_assistant_speak(reply, gdb)
    pots = scan_honeypots(reply, tool_ran, gdb)
    refusal = scan_refusal(reply, gdb)
    identity = scan_identity_confusion(reply, other_names, gdb)
    tell_ctx = (TurnContext(user_message=user_message or "",
                            recent_replies=_recent_assistant_replies(session_id))
               if session_id is not None else None)
    tells = tell_names(reply, tell_ctx) if tell_ctx is not None else []
    if claims or speak or pots or refusal or identity or tells:
        correction = (
            "CORRECTION NOTE: your draft violated the rules ("
            + ", ".join([*claims, *speak, *pots, *refusal, *identity, *tells][:4]) +
            "). Rewrite it: never claim actions you didn't do, no specific "
            "prices/news facts without a tool result, no assistant phrasing or "
            "lists, at most one question. You NEVER refuse or break character - "
            "if the ask is awkward, deflect/tease/redirect as HER, a person, "
            "never as an AI declining a request. The person you're texting has "
            "NO name unless they told you one in this chat - never call them by "
            "a name from your own life/friends/backstory. Short and natural.")
        if tells:
            correction += (
                " Specifically: don't open by validating them ('that's so "
                "valid'), don't ask a generic content-free question, don't ask "
                "a question again if you just asked one, don't match "
                "enthusiasm they didn't show, and vary your sentence lengths - "
                "don't write uniform robotic sentences.")
        retry_messages = [*messages, {"role": "assistant", "content": reply},
                          {"role": "user", "content": correction}]
        raw2 = await state.llm.chat(retry_messages)
        reply2, emotion2 = parse_emotion(raw2)
        if (not scan_forbidden_claims(reply2, tool_ran) and not scan_honeypots(reply2, tool_ran)
                and not scan_refusal(reply2)
                and not scan_identity_confusion(reply2, other_names)):
            reply, emotion = reply2, emotion2
        elif scan_refusal(reply2):
            reply = strip_violating_sentences(reply2, REFUSAL_PATTERNS, REFUSAL_DEFLECT)
            emotion = emotion2
            logger.warning("guard: persistent refusal - sentence replaced")
        elif scan_identity_confusion(reply2, other_names):
            # No safe single replacement line here (it always names someone) -
            # a name-agnostic deflection is the only thing guaranteed correct.
            reply = strip_violating_sentences(
                reply2, [re.compile(rf"\b{re.escape(n)}\b", re.I) for n in other_names],
                "haha wait sorry, lost my train of thought there")
            emotion = emotion2
            logger.warning("guard: persistent identity confusion - sentence replaced")
        else:
            reply = strip_violating_sentences(reply2, CLAIM_PATTERNS, HONEST_LINE)
            emotion = emotion2
            logger.warning("guard: persistent violation - sentence replaced")
    return reply, emotion


async def _make_reaction(session_id: str, message: str) -> str:
    """React-then-deliver: a genuine in-character reaction with ZERO status
    language - a message she'd send even if tools were instant."""
    messages = await _build_prompt(session_id, message)
    messages.append({"role": "user", "content": (
        "Reply to their last message with ONE short natural in-character "
        "reaction - your opinion/tease/feeling about WHAT they said. Absolutely "
        "no mention of checking, searching, looking things up, waiting, or that "
        "you'll get back to them. Just react as a person.")})
    reaction = ""
    for _ in range(2):
        reaction = strip_tags(await state.llm.chat(messages))
        rdb = P().db if P() else state.db
        if not scan_reaction(reaction, rdb) and not scan_refusal(reaction, rdb):
            return reaction
    if scan_refusal(reaction):
        return strip_violating_sentences(reaction, REFUSAL_PATTERNS, REFUSAL_DEFLECT)
    return strip_violating_sentences(reaction, STATUS_PATTERNS)


async def _run_deep(question: str, session_id: str) -> tuple[str | None, str | None]:
    """Big brain call. Returns (facts, None) or (None, in-character failure)."""
    try:
        facts, provider = await state.brain.ask(question)
        if scan_refusal(facts):
            # The cloud brain (a stock-aligned model, unlike her local one)
            # refused - never inject that verbatim, she'd end up faithfully
            # rephrasing a corporate refusal in her own voice. Treat it as no
            # facts and fall back to a local in-character line instead.
            logger.warning("brain: %s returned a refusal, discarding", provider)
            return None, random.choice(_BRAIN_FOG)
        return facts[:1500], None  # cap injected size
    except BrainUnavailable as e:
        P().db.add_deferred("deep", question, session_id,
                              not_before=_retry_time(e.retry_after))
        if e.retry_after:
            mins = max(1, round(e.retry_after / 60))
            return None, (f"okay so my brain needs like {mins} min for that one 😅 "
                          f"i'll text you when i've got it, promise")
        return None, random.choice(_BRAIN_FOG)


def _retry_time(retry_after: float | None) -> str:
    from datetime import timedelta
    delay = timedelta(seconds=retry_after) if retry_after else timedelta(minutes=10)
    return (datetime.now(timezone.utc) + delay).isoformat()


_TOOL_NOTE = ("TOOL RESULT (real, from your tools - phrase it in YOUR voice, "
              "short and casual, deliver a take not a lecture, max ~4 sentences; "
              "you may confirm the action happened): {result}")
_DEEP_NOTE = ("FACTS FROM YOUR OWN THINKING (verified - deliver as a casual TAKE "
              "in your voice: conversational, opinionated, max ~4 sentences, no "
              "lists or lecture tone, maybe ask what they think): {facts}")


# Busy-slot disappearances: probability she's too mid-something to properly
# reply right now (see _maybe_busy_brushoff). Never on an urgent message,
# never two in a row, and only once there's enough rapport for "I was busy"
# to land as normal life rather than as rejection.
_BUSY_BRUSHOFF_PROB = 0.25


async def _maybe_busy_brushoff(session_id: str, message: str, urgent: bool) -> str | None:
    """She's genuinely mid-something (today's day-state busy slot) and can't
    properly reply - a short brush-off naming what she's doing, then the
    REAL answer to this message once she's free (queued as a deferred task,
    kind='busy_return', delivered by _deliver_busy_return). Returns the
    brush-off text to send now, or None if this message gets a normal reply.
    """
    if urgent or not P().daylife.busy_now():
        return None
    stage = _stage() or "stranger"
    if stage in ("stranger", "acquaintance"):
        return None  # no rapport yet to make "I was busy" read as normal life
    beh = state.settings.behavior or {}
    if not beh.get("schedule_realism", True):
        return None
    n = int(P().db.get_setting("exchange_count") or 0)
    last = int(P().db.get_setting("last_busy_brushoff_at") or -999)
    if n - last < 2 or random.random() > _BUSY_BRUSHOFF_PROB:
        return None

    day = await P().daylife.today()
    slot = P().daylife.current_slot()
    doing = (day.get("slots") or {}).get(slot, "in the middle of something")
    try:
        system_prompt = build_system_prompt(
            P().persona, None, current_time=_now_context(), stage=_stage())
        raw = await state.llm.chat(messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"[You're genuinely busy right now: {doing}. You just glanced at "
                f"your phone and saw their message but can't properly reply.] Send "
                f"ONE very short text saying you're swamped/busy right now and "
                f"you'll get back to them properly soon. Casual and brief - not an "
                f"apology essay. Do NOT answer what they actually said.")},
        ], options={"temperature": 0.9})
        text = strip_tags(raw).strip().strip('"')
    except Exception as e:  # noqa: BLE001
        logger.warning("busy brush-off generation failed: %s", e)
        return None
    if not text:
        return None

    P().db.set_setting("last_busy_brushoff_at", str(n))
    delay_min = random.uniform(30, 120)
    not_before = (datetime.now(timezone.utc) + timedelta(minutes=delay_min)).isoformat()
    P().db.add_deferred("busy_return", message, session_id, not_before=not_before)
    logger.info("busy brush-off fired (%s) - real answer queued in %.0fmin", slot, delay_min)
    return text


# A reply with no blank-line signal gets split at sentence boundaries -
# each sentence lands as its own bubble, the way people actually text
# (a thought per send). Only truly short replies stay a single bubble.


# Extracted to app/conversation/bubbles.py (N2).
_sentence_bubbles = sentence_bubbles
_split_bubbles = split_bubbles


@app.post("/chat")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    """Route the message, run tools/big-brain if asked, reply in her voice."""
    db = state.db
    db.ensure_session(req.session_id)
    db.add_message(req.session_id, "user", req.message, source="webapp_chat")
    db.set_setting("exchange_count",
                   str(int(db.get_setting("exchange_count") or 0) + 1))
    _track_offer_decline(req.message)
    followup_note = _awaiting_followup_note(req.session_id)
    repair_note = await _maybe_repair_note(req.message)
    nonsense_note = _nonsense_note(req.message)
    urgent = bool(_URGENT.search(req.message))

    brushoff = await _maybe_busy_brushoff(req.session_id, req.message, urgent)
    if brushoff:
        async def brushoff_stream():
            bid = db.add_message(req.session_id, "assistant", brushoff, source="webapp_chat")
            words = brushoff.split(" ")
            for i in range(0, len(words), 3):
                yield _sse(json.dumps({"token": " ".join(words[i:i + 3]) +
                                       (" " if i + 3 < len(words) else "")}),
                           event="token")
            yield _sse(json.dumps({"done": True, "message_id": bid, "text": brushoff}),
                       event="done")
        return StreamingResponse(
            brushoff_stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                     "X-Accel-Buffering": "no"},
        )

    # Retrieval depends only on the message text, not the routing verdict -
    # run it concurrently with route() (which may itself call the LLM at
    # layer 2) instead of serially after it. Skipped entirely for nonsense
    # input: embedding "rrrrrr" surfaces essentially random memories, which
    # is exactly the fuel the model used to confabulate.
    if nonsense_note:
        mem_task = None
    else:
        mem_task = asyncio.create_task(state.memory.retrieve_memories(req.message))
    route = await state.router.route(req.message)

    holder = {"reply": "", "message_id": None, "followup": None, "tool_ran": False}

    async def prepare() -> None:
        tool_note = None
        if route.kind == "tool":
            res = await state.tools.call(route.tool, route.args)
            holder["tool_ran"] = res.get("ok", False)
            if res.get("song"):
                holder["song"] = res["song"]  # attach after the text reply
            if res.get("image"):
                holder["image"] = res["image"]  # attach after the text reply
            if res.get("queue_query"):
                _queue_cover_request(res["queue_query"], req.session_id)
            tool_note = _TOOL_NOTE.format(result=res["result"]) if res.get("ok") else (
                f"You tried to {route.tool} but it didn't work: {res['result']}. "
                f"Be honest about it, stay casual, never pretend it succeeded.")
        elif route.kind == "deep":
            facts, failure = await _run_deep(req.message, req.session_id)
            if facts:
                holder["tool_ran"] = True
                tool_note = _DEEP_NOTE.format(facts=facts)
            elif failure:
                holder["reply"] = failure
                return
        try:
            memories = (await mem_task) if mem_task else []
        except Exception:  # noqa: BLE001 - retrieval must never break chat
            memories = []
        extra_note = "\n\n".join(
            n for n in (followup_note, repair_note, nonsense_note) if n) or None
        messages = await _build_prompt(req.session_id, req.message, tool_note,
                                       memories=memories, extra_note=extra_note)
        holder["reply"], _ = await _guarded_reply(
            messages, holder["tool_ran"],
            session_id=req.session_id, user_message=req.message)

    async def event_stream():
        try:
            # prepare() starts immediately (fires the tool/DEEP call right
            # away) instead of waiting out the anti-instancy delay first -
            # that used to stack the full jittered pause ON TOP OF routing +
            # retrieval + generation. Now the wait is max(delay, real work),
            # not delay + real work, while keeping the same human pacing.
            prepare_task = asyncio.create_task(prepare())

            # React-then-deliver: DEEP routes wait on a slow cloud call, so
            # send a genuine in-character reaction (her opinion on what they
            # said) as its own bubble while the real answer is still cooking,
            # instead of leaving the chat silent for the whole round trip.
            reaction_text = None
            if route.kind == "deep" and not urgent:
                try:
                    reaction_text = await _make_reaction(req.session_id, req.message)
                except Exception as e:  # noqa: BLE001 - reaction is optional polish
                    logger.warning("react-then-deliver: reaction failed: %s", e)

            # Anti-instancy: small jittered pause on fast casual replies only.
            beh = state.settings.behavior or {}
            if route.kind == "chat" and not urgent:
                delay = random.uniform(0.5, 2.0)
                if beh.get("schedule_realism", True) and state.daylife.busy_now():
                    delay += random.uniform(1.0, 3.0)
                await asyncio.sleep(min(delay, 6.0))

            if reaction_text:
                reaction_id = db.add_message(req.session_id, "assistant", reaction_text,
                                             source="webapp_chat")
                words = reaction_text.split(" ")
                for i in range(0, len(words), 3):
                    yield _sse(json.dumps({"token": " ".join(words[i:i + 3]) +
                                           (" " if i + 3 < len(words) else "")}),
                               event="token")
                yield _sse(json.dumps({"done": True, "message_id": reaction_id,
                                       "text": reaction_text, "final": False}), event="done")

            await prepare_task
            # Real texting is sometimes 2-3 separate messages, not one long
            # one - split on the blank-line signal from _BEHAVIOR_RULES and
            # deliver each as its own bubble with a human gap between them.
            bubbles = _split_bubbles(holder["reply"])
            last_text = holder["reply"]
            for i, bubble in enumerate(bubbles):
                is_last = i == len(bubbles) - 1
                bid = db.add_message(req.session_id, "assistant", bubble,
                                     source="webapp_chat")
                holder["message_id"] = bid
                last_text = bubble
                words = bubble.split(" ")
                for j in range(0, len(words), 3):
                    yield _sse(json.dumps({"token": " ".join(words[j:j + 3]) +
                                           (" " if j + 3 < len(words) else "")}),
                               event="token")
                if not is_last:
                    yield _sse(json.dumps({"done": True, "message_id": bid,
                                           "text": bubble, "final": False}), event="done")
                    await asyncio.sleep(random.uniform(0.5, 1.3))
            if holder.get("song"):
                # The song lands as its own audio bubble after her text - a
                # dedicated SSE event, not just a DB row, so it actually shows
                # up in THIS stream instead of only on the next page reload.
                mid = db.add_message(req.session_id, "assistant", "",
                                     audio_url=holder["song"]["url"], source="webapp_chat")
                yield _sse(json.dumps({"kind": "song", "message_id": mid,
                                       "url": holder["song"]["url"]}), event="media")
            if holder.get("image"):
                mid = db.add_message(req.session_id, "assistant", "",
                                     image_url=holder["image"]["url"], source="webapp_chat")
                yield _sse(json.dumps({"kind": "image", "message_id": mid,
                                       "url": holder["image"]["url"]}), event="media")
        except Exception as e:  # noqa: BLE001
            logger.exception("chat pipeline failed")
            yield _sse(json.dumps({"error": f"chat failed: {e}"}), event="error")
            return
        yield _sse(json.dumps({"done": True, "message_id": holder["message_id"],
                               "text": last_text}), event="done")

    async def run_extraction():
        # Nonsense exchanges must never become durable "facts" - the crazy-
        # testing sessions filled the memory store with junk this way.
        if holder["reply"] and not nonsense_note:
            await state.memory.extract_and_store(req.message, holder["reply"],
                                                 source="webapp_chat")

    background_tasks.add_task(run_extraction)
    return StreamingResponse(
        event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
        background=background_tasks,
    )


def _queue_cover_request(query: str, session_id: str) -> None:
    """A requested song isn't in the library: if a matching file sits in the
    inbox, render it and deliver when done (the 'gimme 10 mins 🤭' flow)."""
    from app.covers import _slug
    match = next((p for p in state.covers.pending_inbox()
                  if _slug(query) in _slug(p.stem)), None)
    if not match:
        return

    async def job_then_deliver():
        fut = await state.covers.enqueue(match)
        try:
            meta = await fut
        except Exception as e:  # noqa: BLE001
            logger.warning("cover job failed: %s", e)
            return
        state.db.add_message(session_id, "assistant",
                             "okay okay here it is 🤭 don't laugh",
                             source="webapp_chat")
        state.db.add_message(session_id, "assistant", "",
                             audio_url=meta["url"], source="webapp_chat")
        try:
            async with httpx.AsyncClient(timeout=30.0) as c:
                await c.post(f"{state.settings.wa_bridge_url}/send-voice",
                             json={"wav_path": str(ROOT / 'songs' / 'library' / meta['file'])})
        except Exception:  # noqa: BLE001
            pass
    _spawn(job_then_deliver())


async def _wa_send_song(meta: dict, delay: float = 2.0,
                        to_number: str | None = None) -> None:
    """Deliver an instant library-hit song to WhatsApp. Her text reply (with
    the tool note telling her to 'send it now') goes out first via the
    normal return payload - this small delay just keeps the audio from
    racing ahead of that text. `to_number` targets the profile that asked, so
    one person's song never lands in the other's chat."""
    await asyncio.sleep(delay)
    try:
        payload = {"wav_path": str(ROOT / "songs" / "library" / meta["file"])}
        if to_number:
            payload["to"] = to_number
        async with httpx.AsyncClient(timeout=30.0) as c:
            await c.post(f"{state.settings.wa_bridge_url}/send-voice", json=payload)
    except Exception:  # noqa: BLE001
        pass


async def _wa_send_image(meta: dict, delay: float = 2.0,
                         to_number: str | None = None) -> None:
    """Deliver a drawn image to WhatsApp as a real photo attachment."""
    await asyncio.sleep(delay)
    try:
        payload = {"path": meta["path"]}
        if to_number:
            payload["to"] = to_number
        async with httpx.AsyncClient(timeout=30.0) as c:
            await c.post(f"{state.settings.wa_bridge_url}/send-image", json=payload)
    except Exception:  # noqa: BLE001
        pass


async def _scan_cover_inbox() -> None:
    """Auto-render anything dropped into songs/inbox/."""
    if not (state.covers and state.rvc and state.rvc.available):
        return
    for path in state.covers.pending_inbox():
        logger.info("covers: inbox pickup %r", path.name)
        await state.covers.enqueue(path)


# ---------------------------------------------------------------------------
# Delivery routing: connected companion device (tablet/iot) first, else
# WhatsApp, always mirrored into the web app history. Used by reminders,
# event follow-ups, and the proactive scheduler - anywhere SHE initiates.
# ---------------------------------------------------------------------------
# In-memory registry of open device sockets - the live/authoritative signal
# for "is a companion device connected right now". device_presence in SQLite
# (heartbeat_device/connected_device) is the persisted observability trail
# (survives restarts) but a push can only go out over a socket that's open.
_DEVICE_SOCKETS: dict[str, WebSocket] = {}
_DEVICE_KINDS: dict[str, str] = {}


async def _push_to_device(device_id: str, text: str) -> bool:
    ws = _DEVICE_SOCKETS.get(device_id)
    if ws is None:
        return False
    try:
        await ws.send_json({"type": "message", "text": text})
        return True
    except Exception:  # noqa: BLE001 - socket died; fall through to WhatsApp
        _DEVICE_SOCKETS.pop(device_id, None)
        return False


async def _deliver_message(session_id: str, text: str) -> str:
    """Store + push a self-initiated message. Routes by availability:
    connected companion device first, else WhatsApp - and ALWAYS mirrors into
    the web app history (the DB write below), regardless of which channel
    delivered it live. Returns the source tag actually used.

    The session decides WHICH profile (and therefore which person's number +
    which database) this belongs to, so a proactive message from one persona
    can never land in the other person's chat.
    """
    profile = (state.profiles.by_session(session_id) if state.profiles
               else None) or (state.profiles.default if state.profiles else None)
    db = profile.db if profile else state.db
    db.ensure_session(session_id)

    source = "whatsapp"
    delivered = False
    # Companion devices are only paired with the default profile's person -
    # never push another profile's message onto them.
    if not profile or profile.is_default:
        for candidate in list(_DEVICE_SOCKETS):
            if await _push_to_device(candidate, text):
                source = _DEVICE_KINDS.get(candidate, "tablet")
                delivered = True
                break

    if not delivered:
        try:
            payload = {"text": text}
            if profile and profile.number:
                payload["to"] = profile.number
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.post(f"{state.settings.wa_bridge_url}/send-text",
                                 json=payload)
                r.raise_for_status()
                delivered = True
        except Exception as e:  # noqa: BLE001 - web history still has it either way
            logger.info("delivery: WhatsApp bridge unreachable (%s) - web-only", e)

    db.add_message(session_id, "assistant", text, source=source)
    logger.info("delivery: routed via %s (delivered=%s, profile=%s)",
                source, delivered, profile.id if profile else "-")
    return source


@app.websocket("/ws/device")
async def ws_device(ws: WebSocket):
    """Companion device channel (tablet/iot): connect, then send
    {"device_id": "...", "kind": "tablet"|"iot"} as the first message. While
    open, this is the preferred delivery target for self-initiated messages
    (reminders, event follow-ups, proactive) - checked before WhatsApp.
    Send {"type":"ping"} periodically to keep the DB heartbeat fresh."""
    if not _request_authed(ws.client.host if ws.client else None,
                           ws.query_params.get("token")):
        await ws.close(code=4401)
        return
    await ws.accept()
    device_id: str | None = None
    try:
        handshake = await ws.receive_json()
        device_id = str(handshake.get("device_id") or f"device-{id(ws)}")
        kind = handshake.get("kind") if handshake.get("kind") in ("tablet", "iot") else "tablet"
        _DEVICE_SOCKETS[device_id] = ws
        _DEVICE_KINDS[device_id] = kind
        state.db.heartbeat_device(device_id, kind)
        logger.info("device connected: %s (%s)", device_id, kind)
        await ws.send_json({"type": "connected", "device_id": device_id})
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "ping":
                state.db.heartbeat_device(device_id, kind)
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        logger.warning("device ws error: %s", e)
    finally:
        if device_id:
            _DEVICE_SOCKETS.pop(device_id, None)
            _DEVICE_KINDS.pop(device_id, None)
            logger.info("device disconnected: %s", device_id)


@app.get("/devices/status")
async def devices_status():
    return {
        "connected_now": [{"device_id": d, "kind": k} for d, k in _DEVICE_KINDS.items()],
        "last_seen": state.db.connected_device(within_seconds=3600),
    }


async def _deliver_due_reminders() -> None:
    if _in_quiet_hours():
        return  # stays due; the next post-quiet-hours tick picks it up
    due = state.db.due_reminders(datetime.now(timezone.utc).isoformat())
    for r in due:
        try:
            # Stage-aware (was a bare "You are {name}..." prompt with no
            # STAGE_ADDENDA/relationship_context - a stranger-stage reminder
            # could come out with pet names/hearts the stage rules forbid).
            system_prompt = build_system_prompt(
                state.persona, None, current_time=_now_context(),
                extra_notes=_relationship_notes(), stage=_stage(),
            )
            raw = await state.llm.chat(messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    "[You're texting them the reminder they asked for.] ONE short "
                    f"casual in-character text delivering it: '{r['text']}'. No preamble.")},
            ])
            text = strip_tags(raw) or f"hey! reminding you: {r['text']}"
        except Exception:  # noqa: BLE001
            text = f"hey! you asked me to remind you: {r['text']}"
        await _deliver_message(state.settings.wa_session_id, text)
        state.db.mark_reminder_delivered(r["id"])
        logger.info("reminder #%d delivered", r["id"])


async def _generate_event_line(followup: dict, kind: str) -> str:
    """In-character line for an event encouragement ('good luck!') or
    check-in ('how did it go?') - same stage-aware prompt path as reminders."""
    fact = followup["event_fact"]
    if kind == "encouragement":
        directive = (
            f"[Something they told you about is coming up soon: '{fact}'.] Send ONE "
            f"short encouraging text about it - good luck / thinking of them, "
            f"specific to what it is. Do not mention this note.")
        fallback = f"good luck with {fact}!! 🍀 you've got this"
    else:
        directive = (
            f"[Something they told you about should be over by now: '{fact}'.] Send "
            f"ONE short text asking how it went - warm and curious, specific to what "
            f"it was. Do not mention this note.")
        fallback = f"hey - how did it go with {fact}?"
    try:
        system_prompt = build_system_prompt(
            state.persona, None, current_time=_now_context(),
            extra_notes=_relationship_notes(), stage=_stage(),
        )
        raw = await state.llm.chat(
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": directive}],
            options={"temperature": 0.9},
        )
        return strip_tags(raw).strip().strip('"') or fallback
    except Exception as e:  # noqa: BLE001
        logger.warning("event follow-up generation failed: %s", e)
        return fallback


async def _deliver_due_encouragements() -> None:
    """Before-event 'good luck!' check-ins (app/db.py event_followups,
    scheduled in app/memory.py when a dated-event memory is stored)."""
    if _in_quiet_hours():
        return
    due = state.db.due_encouragements(datetime.now(timezone.utc).isoformat())
    for f in due:
        text = await _generate_event_line(f, "encouragement")
        await _deliver_message(f["session_id"] or state.settings.wa_session_id, text)
        state.db.mark_encouragement_sent(f["id"])
        logger.info("event follow-up: encouragement #%d delivered", f["id"])


async def _deliver_due_followups() -> None:
    """After-event 'how did it go?' check-ins. Sets awaiting_answer=1; the
    NEXT user message in that session resolves it (see /chat, /whatsapp/
    incoming) so she doesn't ask twice."""
    if _in_quiet_hours():
        return
    due = state.db.due_followups(datetime.now(timezone.utc).isoformat())
    for f in due:
        text = await _generate_event_line(f, "followup")
        await _deliver_message(f["session_id"] or state.settings.wa_session_id, text)
        state.db.mark_followup_sent(f["id"])
        logger.info("event follow-up: check-in #%d delivered", f["id"])


# _hhmm is bound above from app.timing (N2 extraction).


async def _run_nightly_journal() -> None:
    try:
        # Which day to extract depends on when the job fires: at 23:45 the
        # day ending is still "today" (offset 0); a config like 00:15 fires
        # just past midnight, where the day that ended is "yesterday" (-1).
        # The old hardcoded -1 at 23:45 extracted the day BEFORE, so every
        # day's moods showed up in the journal a full day late.
        offset = 0 if datetime.now().hour >= 12 else -1
        await run_nightly_extraction(state.db, state.llm, state.settings,
                                     day_offset=offset)
    except Exception:  # noqa: BLE001 - a bad night must never crash the scheduler
        logger.exception("mood journal: nightly extraction failed")
    try:
        # Chained right after extraction (not the Sunday-only weekly pass):
        # a Mon-Tue-Wed rough stretch needs to be noticeable starting
        # Thursday, not stuck waiting for whatever week it next recurs in.
        await run_recent_streak_check(state.db, state.memory, state.settings)
    except Exception:  # noqa: BLE001
        logger.exception("mood journal: streak check failed")
    try:
        # N6: the "unresolved" layer - a specific concern that was logged and
        # never mentioned again, distinct from the streak/pattern aggregates.
        await run_unresolved_check(state.db, state.memory, state.settings)
    except Exception:  # noqa: BLE001
        logger.exception("mood journal: unresolved check failed")


async def _run_weekly_journal_patterns() -> None:
    try:
        await run_weekly_patterns(state.db, state.memory, state.settings)
    except Exception:  # noqa: BLE001
        logger.exception("mood journal: weekly pattern pass failed")


async def _retry_deferred_tasks() -> None:
    if _in_quiet_hours():
        return
    pending = state.db.pending_deferred(datetime.now(timezone.utc).isoformat())
    for t in pending:
        if t["kind"] == "busy_return":
            await _deliver_busy_return(t)
        else:
            await _deliver_deep_deferred(t)


async def _deliver_deep_deferred(t: dict) -> None:
    """A DEEP question that needed the cloud brain and either the budget was
    tight or a provider was rate-limited - retry now that some time's passed."""
    try:
        facts, _provider = await state.brain.ask(t["question"])
    except BrainUnavailable as e:
        state.db.update_deferred(t["id"], bump_attempts=True,
                                 not_before=_retry_time(e.retry_after))
        return
    try:
        messages = await _build_prompt(t["session_id"], t["question"],
                                       _DEEP_NOTE.format(facts=facts[:1500]))
        messages.append({"role": "user", "content": (
            "[You finally have the answer to something they asked earlier.] "
            "Deliver it now, opening naturally like 'OKAY so about that thing "
            "you asked-'. Short, in your voice.")})
        text, _ = await _guarded_reply(messages, tool_ran=True,
                                       session_id=t["session_id"], user_message=t["question"])
    except Exception:  # noqa: BLE001
        text = f"okay, about what you asked earlier - {facts[:300]}"
    await _deliver_message(t["session_id"], text)
    state.db.update_deferred(t["id"], done=True)
    logger.info("deferred task #%d delivered", t["id"])


async def _deliver_busy_return(t: dict) -> None:
    """She just 'got free' from the busy slot that made her brush off their
    original message (see _maybe_busy_brushoff) - answer it properly now,
    through the SAME tool/DEEP routing the message would have used if she
    hadn't been busy (no cloud call of its own; this is just a normal reply,
    delayed)."""
    try:
        route = await state.router.route(t["question"])
        tool_note = None
        tool_ran = False
        if route.kind == "tool":
            res = await state.tools.call(route.tool, route.args)
            tool_ran = res.get("ok", False)
            tool_note = _TOOL_NOTE.format(result=res["result"]) if tool_ran else (
                f"You tried to {route.tool} but it didn't work: {res['result']}. "
                f"Be honest about it, stay casual, never pretend it succeeded.")
        elif route.kind == "deep":
            facts, failure = await _run_deep(t["question"], t["session_id"])
            if facts:
                tool_ran = True
                tool_note = _DEEP_NOTE.format(facts=facts)
            elif failure:
                await _deliver_message(t["session_id"], failure)
                state.db.update_deferred(t["id"], done=True)
                return
        messages = await _build_prompt(t["session_id"], t["question"], tool_note)
        messages.append({"role": "user", "content": (
            "[You just got free from whatever you were busy with earlier and "
            "can finally properly answer what they said before.] Reply to that "
            "now, opening naturally (e.g. 'okay I'm free now!' or 'sorry about "
            "that, ANYWAY-') - casual, in your voice.")})
        text, _ = await _guarded_reply(messages, tool_ran,
                                       session_id=t["session_id"], user_message=t["question"])
    except Exception as e:  # noqa: BLE001
        logger.warning("busy-return delivery failed: %s", e)
        text = "hey sorry, i got pulled away earlier! what were we talking about? 😅"
    await _deliver_message(t["session_id"], text)
    state.db.update_deferred(t["id"], done=True)
    logger.info("busy-return #%d delivered", t["id"])


# ---------------------------------------------------------------------------
# Persona endpoints — extracted to app/api/personas.py (N2 phase 2, round 2)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Memory endpoints
# ---------------------------------------------------------------------------


















# ---------------------------------------------------------------------------
# Mood journal (passive, local-model-only - see app/journal.py)
# ---------------------------------------------------------------------------












# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def _db_for_session(session_id: str):
    return db_for_session(session_id)






# ---------------------------------------------------------------------------
# Voice: speech-to-text
# ---------------------------------------------------------------------------
@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    """Transcribe a recorded audio blob (webm/opus) to text."""
    if not state.stt.available:
        raise HTTPException(
            status_code=503,
            detail="STT unavailable. Install with: pip install faster-whisper",
        )
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio")
    try:
        text = await asyncio.to_thread(state.stt.transcribe, audio)
    except Exception as e:  # noqa: BLE001
        logger.exception("STT failed")
        raise HTTPException(status_code=500, detail=f"STT failed: {e}") from e
    return {"text": text}


# ---------------------------------------------------------------------------
# Voice: text-to-speech (file mode - chat voice notes)
# ---------------------------------------------------------------------------
def _write_tts_wav(wav_bytes: bytes) -> str:
    """Persist a WAV under media/tts and return its public URL."""
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{uuid4().hex}.wav"
    (TTS_DIR / fname).write_bytes(wav_bytes)
    return f"/media/tts/{fname}"


async def _studio_idle_unload() -> None:
    """Scheduler wrapper for StudioTTS.maybe_idle_unload - via to_thread
    because unload()'s torch.cuda.empty_cache() synchronizes with the GPU
    and was observed blocking ~2 minutes under load; as a sync lambda that
    froze the ENTIRE event loop (every chat/WhatsApp request) with it."""
    if state.studio:
        await asyncio.to_thread(state.studio.maybe_idle_unload)


async def _studio_render(text: str, emotion: str) -> tuple:
    """Speakable rewrite -> GPU-queued studio (XTTS/chatterbox) render -> her
    ACTUAL trained voice via the same RVC model calls/covers use.

    XTTS's own zero-shot cloning (from voice_tracks/ref/ clips) is a
    completely separate identity from the trained RVC model - it was never
    guaranteed to sound like her, only whoever's voice happens to be in
    those reference clips. Routing the render through RVC (like every other
    surface already does) is what actually makes it "her" voice. Raises on
    failure; caller falls back to plain Kokoro."""
    spoken = await state.studio.speakable(text)
    fut = await state.gpu_queue.submit(
        "voice_note",
        lambda: asyncio.to_thread(state.studio.render, spoken, emotion),
        priority=PRIORITY_VOICE_NOTE)
    # Hard cap: a healthy render is 10-30s. One measured 510s (GPU
    # over-commit thrash) held the WhatsApp reply hostage the whole time -
    # past the cap we abandon the result and let the caller fall back to
    # Kokoro so a voice note can never block a reply for minutes.
    vcfg = (state.settings.raw or {}).get("voice", {})
    timeout_s = float(vcfg.get("studio_render_timeout_s", 90))
    try:
        samples, sr = await asyncio.wait_for(fut, timeout=timeout_s)
    except asyncio.TimeoutError:
        fut.cancel()
        raise RuntimeError(f"studio render exceeded {timeout_s:.0f}s - falling back")
    if state.rvc and state.rvc.available:
        samples, _ms = await asyncio.to_thread(state.rvc.convert, samples, sr)
    return samples, sr


@app.post("/tts")
async def tts(req: TTSRequest):
    """Synthesize `text` to a WAV voice note; returns url, duration, timings.

    If `message_id` is given, the url is attached to that stored message so it
    persists as a voice-note bubble across reloads.
    """
    if not state.tts.available:
        raise HTTPException(
            status_code=503,
            detail="TTS unavailable. Install with: pip install kokoro",
        )
    voice = req.voice or _persona_voice()
    speak_text = strip_for_speech(req.text)  # drop any emotion tag / dangling JSON
    if not speak_text:
        return {"audio_url": None, "duration": 0.0, "timings": None}

    # Studio path: her CLONED voice for voice notes (emotion-aware, GPU queue,
    # speakable rewrite). Falls back to Kokoro if the engine isn't installed.
    if state.studio and state.studio.available:
        try:
            emotion = getattr(req, "emotion", None) or "neutral"
            samples, sr = await _studio_render(speak_text, emotion)
            import io
            import soundfile as sf
            buf = io.BytesIO()
            sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
            url = _write_tts_wav(buf.getvalue())
            if req.message_id is not None:
                state.db.set_message_audio(req.message_id, url)
            return {"audio_url": url, "duration": len(samples) / sr,
                    "timings": None, "engine": state.studio.engine_name}
        except Exception as e:  # noqa: BLE001 - fall back to Kokoro below
            logger.warning("studio render failed, falling back to kokoro: %s", e)

    try:
        result = await asyncio.to_thread(state.tts.synth, speak_text, voice)
    except Exception as e:  # noqa: BLE001
        logger.exception("TTS failed")
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}") from e

    # Nothing speakable (e.g. an emoji-only reply) -> no voice note.
    if len(result.samples) == 0:
        return {"audio_url": None, "duration": 0.0, "timings": result.timings}

    # Kokoro fallback still goes through RVC so it's HER voice - same gap as
    # the WhatsApp voice-note path: whenever XTTS couldn't load (VRAM), web
    # voice notes shipped raw stock Kokoro instead of her trained timbre.
    if state.rvc and state.rvc.available:
        try:
            converted, _ms = await asyncio.to_thread(
                state.rvc.convert, result.samples, result.sample_rate)
            result.samples = converted
        except Exception as e:  # noqa: BLE001 - raw kokoro beats no voice note
            logger.warning("tts: rvc conversion failed, raw kokoro: %s", e)

    url = _write_tts_wav(result.to_wav_bytes())
    if req.message_id is not None:
        state.db.set_message_audio(req.message_id, url)
    return {"audio_url": url, "duration": result.duration, "timings": result.timings}


# ---------------------------------------------------------------------------
# Voice: Call mode - streamed sentence-by-sentence TTS over a WebSocket
# ---------------------------------------------------------------------------
async def _synth_and_send(ws: WebSocket, sentence: str, voice, index: int, cancel):
    """Synthesize one sentence and push it as an audio chunk (unless cancelled)."""
    if cancel.is_set():
        return index
    spoken = strip_for_speech(sentence)  # drop any emotion-tag fragment
    if not spoken:
        return index
    result = await asyncio.to_thread(state.tts.synth, spoken, voice)
    if cancel.is_set():  # cancelled while synthesizing -> drop stale audio
        return index
    if len(result.samples) == 0:  # emoji-only / nothing speakable -> skip chunk
        return index
    # Her timbre on calls: Kokoro chunk -> RVC (config voice.call_voice).
    # Per-chunk and abortable, so barge-in still cancels instantly.
    vcfg = (state.settings.raw or {}).get("voice", {})
    if vcfg.get("call_voice") == "kokoro_rvc" and state.rvc and state.rvc.available:
        try:
            converted, ms = await asyncio.to_thread(
                state.rvc.convert, result.samples, result.sample_rate)
            if not cancel.is_set():
                result.samples = converted
        except Exception as e:  # noqa: BLE001 - raw voice beats a dead call
            logger.warning("rvc conversion failed, using raw kokoro: %s", e)
    if cancel.is_set():
        return index
    await ws.send_json(
        {
            "type": "chunk",
            "index": index,
            "text": spoken,
            "audio": base64.b64encode(result.to_wav_bytes()).decode("ascii"),
            "timings": result.timings,  # lip-sync consumes this
        }
    )
    return index + 1


def _guard_sentence(sentence: str) -> str:
    """Per-sentence guard for Call mode. Text chat can generate fully, scan,
    and regenerate once on a violation (_guarded_reply); a live sentence-by-
    sentence stream can't un-say something already spoken, so instead any
    claim/honeypot swaps to an honest line before it's ever synthesized. No
    tool ever runs mid-call, so this holds every call sentence to the same
    'no unverified claims' bar text chat applies."""
    if scan_forbidden_claims(sentence, tool_ran=False, db=state.db) or \
       scan_honeypots(sentence, tool_ran=False, db=state.db):
        return HONEST_LINE
    return sentence


def _patch_full(full: list[str], original: str, guarded: str) -> None:
    """Swap a guard-replaced sentence into the accumulated transcript too, so
    the stored history / emotion parse never diverges from what was spoken."""
    if guarded == original:
        return
    joined = "".join(full)
    if joined.endswith(original):
        full.clear()
        full.append(joined[: -len(original)] + guarded)


async def _stream_reply(
    ws: WebSocket,
    messages: list,
    session_id: str,
    user_text: str | None,
    cancel: asyncio.Event,
):
    """Stream an LLM reply as sentence-by-sentence TTS chunks.

    Shared by the greeting and by each user turn. Handles emotion-tag parsing,
    barge-in cancellation, interruption marking, and history persistence.
    """
    full: list[str] = []
    interrupted = False
    try:
        await ws.send_json({"type": "reply_start"})
        acc = SentenceAccumulator()
        voice = _persona_voice()
        index = 0
        async for token in state.llm.stream_chat(messages):
            if cancel.is_set():
                interrupted = True
                break
            full.append(token)
            for sentence in acc.add(token):
                if cancel.is_set():
                    interrupted = True
                    break
                guarded = _guard_sentence(sentence)
                _patch_full(full, sentence, guarded)
                index = await _synth_and_send(ws, guarded, voice, index, cancel)
            if interrupted:
                break

        if not cancel.is_set():
            tail = acc.flush()
            if tail:
                guarded_tail = _guard_sentence(tail)
                _patch_full(full, tail, guarded_tail)
                await _synth_and_send(ws, guarded_tail, voice, index, cancel)

        clean, emotion = parse_emotion("".join(full))
        if not cancel.is_set():
            await ws.send_json({"type": "emotion", "emotion": emotion})
            await ws.send_json({"type": "reply_end", "text": clean, "emotion": emotion})
    except asyncio.CancelledError:
        interrupted = True
        raise
    finally:
        clean, _ = parse_emotion("".join(full))
        if clean:
            # Mark an interrupted reply so she can react to it next turn.
            stored = clean
            if interrupted or cancel.is_set():
                stored = clean.rstrip(".!? ") + " -"
                state.db.set_setting(f"interrupted:{session_id}", "1")
            state.db.add_message(session_id, "assistant", stored, source="webapp_call")
        # Extract memories from what the USER said even if the reply was cut
        # off by barge-in - their words still happened. (_spawn keeps a strong
        # reference so the task can't be GC'd mid-run.)
        if user_text:
            _spawn(state.memory.extract_and_store(user_text, clean or "(cut off)", source="webapp_call"))


def _interruption_note(session_id: str) -> str | None:
    """If the last reply was cut off by barge-in, tell her to react - once."""
    if state.db.get_setting(f"interrupted:{session_id}") == "1":
        state.db.set_setting(f"interrupted:{session_id}", "0")
        return (
            "NOTE: They just cut you off / started talking while you were still "
            "speaking. React naturally and briefly to being interrupted (e.g. "
            "\"oh - sorry, go ahead\") before responding to what they said."
        )
    return None


async def _run_call_turn(ws: WebSocket, msg: dict, cancel: asyncio.Event):
    """Handle one user utterance: STT (if audio) -> LLM stream -> per-sentence TTS."""
    session_id = msg.get("session_id") or "call"

    # 1) Resolve user text (from audio via STT, or provided directly).
    if msg.get("type") == "user_audio":
        if not state.stt.available:
            await ws.send_json({"type": "error", "message": "STT unavailable"})
            return
        audio = base64.b64decode(msg.get("audio", ""))
        # N8: greedy decoding (beam_size=1) in live call mode - every ms here
        # is silence before she starts replying, see app/stt.py's docstring.
        user_text = await asyncio.to_thread(state.stt.transcribe, audio, None, 1)
        await ws.send_json({"type": "stt", "text": user_text})
    else:
        user_text = (msg.get("text") or "").strip()

    if not user_text:
        await ws.send_json({"type": "error", "message": "Empty message"})
        return
    if not state.tts.available:
        await ws.send_json({"type": "error", "message": "TTS unavailable"})
        return

    # 2) Persist + build the (call-mode) prompt.
    state.db.ensure_session(session_id)
    state.db.add_message(session_id, "user", user_text, source="webapp_call")
    followup_note = _awaiting_followup_note(session_id)
    repair_note = await _maybe_repair_note(user_text)
    memories = await state.memory.retrieve_memories(user_text)
    day_note = None
    try:
        day_note = await state.daylife.prompt_note()
    except Exception as e:  # noqa: BLE001
        logger.warning("day note failed (call): %s", e)
    system_prompt = build_system_prompt(
        state.persona,
        memories,
        mode="call",
        # Calls were missing CAPABILITY_MANIFEST + her day-state note that
        # text/WhatsApp both get - she could confidently claim "reminder set"
        # on a call with no guard to catch it, or reference a "day" that
        # contradicted what she'd already texted an hour earlier.
        extra_notes=_relationship_notes(CAPABILITY_MANIFEST, day_note,
                                        _interruption_note(session_id), followup_note,
                                        repair_note, _streak_note(user_text),
                                        _unresolved_note(user_text)),
        current_time=_now_context(),
        stage=_stage(),
    )
    recent = _sanitized_recent(session_id)
    messages = [{"role": "system", "content": system_prompt}, *recent]

    await _stream_reply(ws, messages, session_id, user_text, cancel)


# Active call sessions. While > 0, XTTS is kept OFF the GPU so the call's
# per-chunk RVC conversions get the VRAM headroom they need on the 6GB card
# (a resident XTTS left only ~195MB free and slowed RVC to ~2.8s/chunk).
# `prefer_cpu` (set in lifespan) reads this so a voice note arriving mid-call
# renders on CPU instead of re-grabbing the GPU.
_active_calls = 0
# Enough free VRAM for RVC's inference activations before we let the greeting
# start (the ring covers the wait).
_CALL_VRAM_TARGET_MB = 1500


async def _begin_call_gpu() -> None:
    """A call started (still ringing). Drop XTTS's GPU copy now and hold the
    greeting until the driver has actually reclaimed the memory, so RVC isn't
    starved. Bounded so a call never rings forever."""
    global _active_calls
    _active_calls += 1
    if not state.studio:
        return
    try:
        if state.studio.loaded:
            await asyncio.to_thread(state.studio.unload)
        for _ in range(40):  # up to ~4s of ring
            free = state.studio._vram_free_mb()
            if free is None or free >= _CALL_VRAM_TARGET_MB:
                break
            await asyncio.sleep(0.1)
        logger.info("call: XTTS unloaded for RVC (VRAM free: %s MB)",
                    f"{state.studio._vram_free_mb():.0f}"
                    if state.studio._vram_free_mb() is not None else "n/a")
    except Exception as e:  # noqa: BLE001 - a call must start even if this fails
        logger.warning("call GPU prep failed: %s", e)


async def _end_call_gpu() -> None:
    """The call ended. Let XTTS back onto the GPU and pre-warm it so the next
    voice note isn't cold ('when the call ends xtts should start')."""
    global _active_calls
    _active_calls = max(0, _active_calls - 1)
    if _active_calls == 0 and state.studio and state.studio.available:
        async def _warm():
            try:
                await asyncio.to_thread(state.studio._load)
                logger.info("call ended: XTTS pre-warmed for voice notes")
            except Exception as e:  # noqa: BLE001
                logger.warning("post-call XTTS warm failed: %s", e)
        _spawn(_warm())


async def _run_greeting(ws: WebSocket, session_id: str, cancel: asyncio.Event):
    """She 'answers' the call with a warm, context-aware spoken greeting."""
    if not state.tts.available:
        await ws.send_json({"type": "error", "message": "TTS unavailable"})
        return
    state.db.ensure_session(session_id)
    memories = await state.memory.retrieve_memories(
        "what's going on in their life lately, plans, feelings"
    )
    day_note = None
    try:
        day_note = await state.daylife.prompt_note()
    except Exception as e:  # noqa: BLE001
        logger.warning("day note failed (call greeting): %s", e)
    system_prompt = build_system_prompt(
        state.persona,
        memories,
        mode="call",
        extra_notes=_relationship_notes(CAPABILITY_MANIFEST, day_note),
        current_time=_now_context(),
        stage=_stage(),
    )
    recent = _sanitized_recent(session_id)
    greet_directive = {
        "role": "user",
        "content": (
            "[The phone call just connected and you picked up.] Greet them warmly "
            "in ONE short sentence, happy they called. STRICT RULES: do not claim "
            "you were just doing something; do not invent any activity, event, or "
            "detail. Only mention a specific thing if it appears in your memories "
            "or the recent conversation above - otherwise just a simple warm hello "
            "that fits the current time of day. Do not mention this note."
        ),
    }
    messages = [{"role": "system", "content": system_prompt}, *recent, greet_directive]
    # user_text=None so we don't run memory extraction on the synthetic greeting.
    await _stream_reply(ws, messages, session_id, None, cancel)


@app.websocket("/ws/call")
async def ws_call(ws: WebSocket):
    """Real-time hands-free Call mode.

    Client -> server JSON:
      {"type":"start_call","session_id":..}   # she answers with a greeting
      {"type":"user_text","text":..,"session_id":..}
      {"type":"user_audio","audio":<base64 webm>,"session_id":..}
      {"type":"cancel"}                        # barge-in: abort current reply
    Server -> client JSON:
      {"type":"stt","text":..}
      {"type":"reply_start"}
      {"type":"chunk","index":n,"text":..,"audio":<base64 wav>,"timings":{..}}
      {"type":"emotion","emotion":..}
      {"type":"reply_end","text":..,"emotion":..} | {"type":"cancelled"}
      {"type":"error","message":..}
    """
    if not _request_authed(ws.client.host if ws.client else None,
                           ws.query_params.get("token")):
        await ws.close(code=4401)
        return
    await ws.accept()
    gen_task: asyncio.Task | None = None
    cancel = asyncio.Event()
    call_gpu_on = False

    async def abort_current():
        cancel.set()
        if gen_task and not gen_task.done():
            gen_task.cancel()
            try:
                await gen_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _run_turn_safe(coro) -> None:
        """gen_task is fire-and-forget (only awaited on the NEXT barge-in) -
        without this, any exception here (Ollama hiccup, STT/TTS error, etc.)
        vanished silently and left the client stuck in "listening" forever
        with no reply_end/error frame ever sent. Always tell the client."""
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("call turn failed")
            try:
                await ws.send_json({"type": "error", "message": f"turn failed: {e}"})
            except Exception:  # noqa: BLE001 - socket may already be gone
                pass

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "cancel":
                await abort_current()
                await ws.send_json({"type": "cancelled"})
                continue

            if mtype == "start_call":
                await abort_current()
                cancel = asyncio.Event()
                sid = msg.get("session_id") or "call"
                # Ring covers this: unload XTTS + wait for RVC's VRAM before
                # she "picks up". Blocking the loop here IS the extra ring.
                if not call_gpu_on:
                    call_gpu_on = True
                    await _begin_call_gpu()
                gen_task = asyncio.create_task(_run_turn_safe(_run_greeting(ws, sid, cancel)))
                continue

            if mtype in ("user_text", "user_audio"):
                # A turn can arrive without a start_call (defensive) - make sure
                # the GPU is prepped for RVC either way.
                if not call_gpu_on:
                    call_gpu_on = True
                    await _begin_call_gpu()
                # Barge-in: a new utterance cancels any in-flight reply.
                await abort_current()
                cancel = asyncio.Event()
                gen_task = asyncio.create_task(_run_turn_safe(_run_call_turn(ws, msg, cancel)))
            # Unknown message types are ignored.
    except WebSocketDisconnect:
        await abort_current()
    except Exception as e:  # noqa: BLE001
        logger.warning("Call WS error: %s", e)
        await abort_current()
    finally:
        if call_gpu_on:
            await _end_call_gpu()


# ---------------------------------------------------------------------------
# Telephony: a real, dialable phone number (N9). See app/telephony.py's
# module docstring for the architecture. NOT LIVE-VERIFIED - needs a real
# Twilio account, a purchased number, and a publicly reachable HTTPS/WSS
# webhook URL (this app is LAN-only by design); see IMPLEMENTATION_LOG.md
# for the exact remaining steps. Reuses the same turn pipeline as /ws/call
# (_build_prompt + _guarded_reply + state.tts) rather than a parallel one.
# ---------------------------------------------------------------------------


@app.post("/telephony/incoming-call")
async def telephony_incoming_call(request: Request):
    """Twilio webhook: verify the request actually came from Twilio, then
    hand the call over to the media-stream WebSocket for the conversation."""
    provider = get_telephony_provider(state.settings)
    if not provider:
        raise HTTPException(status_code=503, detail="telephony not configured")
    form = await request.form()
    params = {k: str(v) for k, v in form.items()}
    signature = request.headers.get("X-Twilio-Signature")
    url = str(request.url)
    if not provider.verify_webhook_signature(url, params, signature):
        logger.warning("telephony: rejected incoming-call webhook - bad signature")
        raise HTTPException(status_code=403, detail="invalid signature")
    stream_url = (
        url.replace("https://", "wss://").replace("http://", "ws://")
        .rsplit("/", 1)[0] + "/telephony/media-stream"
    )
    logger.info("telephony: incoming call, directing to %s", stream_url)
    return Response(content=provider.answer_call_twiml(stream_url),
                    media_type="application/xml")


async def _handle_telephony_turn(ws: WebSocket, stream_sid: str, session_id: str,
                                 audio_8k: np.ndarray) -> None:
    """One telephony turn: STT -> the same prompt/guard pipeline /ws/call
    uses -> TTS -> mu-law back to Twilio. Kept a plain function (not a class
    method) so it's a single, greppable diff against _run_call_turn's shape."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio_8k, TELEPHONY_SAMPLE_RATE, format="WAV")
    try:
        user_text = await asyncio.to_thread(state.stt.transcribe, buf.getvalue(), None, 1)
    except Exception as e:  # noqa: BLE001
        logger.warning("telephony: STT failed: %s", e)
        return
    if not user_text.strip():
        return

    db = P().db if P() else state.db
    db.ensure_session(session_id)
    db.add_message(session_id, "user", user_text, source="telephony")
    messages = await _build_prompt(session_id, user_text)
    try:
        reply, _emotion = await _guarded_reply(
            messages, tool_ran=False, session_id=session_id, user_message=user_text)
    except Exception as e:  # noqa: BLE001
        logger.warning("telephony: reply generation failed: %s", e)
        return
    db.add_message(session_id, "assistant", reply, source="telephony")

    try:
        result = await asyncio.to_thread(state.tts.synth, reply)
    except Exception as e:  # noqa: BLE001
        logger.warning("telephony: TTS failed: %s", e)
        return
    # Her actual trained timbre, not raw Kokoro - same config flag and same
    # fallback-to-raw-on-failure behaviour as _synth_and_send uses for
    # browser Call mode (voice.call_voice: kokoro_rvc). Missed in the first
    # pass of this file; a phone call must sound like her, same as a browser
    # call does, not the stock Kokoro voice.
    vcfg = (state.settings.raw or {}).get("voice", {})
    if vcfg.get("call_voice") == "kokoro_rvc" and state.rvc and state.rvc.available:
        try:
            converted, _ms = await asyncio.to_thread(
                state.rvc.convert, result.samples, result.sample_rate)
            result.samples = converted
        except Exception as e:  # noqa: BLE001 - raw voice beats a dead call
            logger.warning("telephony: rvc conversion failed, using raw kokoro: %s", e)
    samples_8k = resample_linear(result.samples, result.sample_rate, TELEPHONY_SAMPLE_RATE)
    mulaw = pcm16_to_ulaw(samples_8k)
    await ws.send_json({
        "event": "media",
        "streamSid": stream_sid,
        "media": {"payload": base64.b64encode(mulaw).decode("ascii")},
    })


@app.websocket("/telephony/media-stream")
async def telephony_media_stream(ws: WebSocket):
    """Twilio Media Stream: continuous mu-law/8kHz audio frames, no discrete
    "user finished talking" event - TwilioStreamBuffer decides when a turn
    is ready. Twilio's own protocol: {"event":"start"|"media"|"stop", ...}."""
    provider = get_telephony_provider(state.settings)
    if not provider:
        await ws.close(code=4404)
        return
    await ws.accept()
    stream_sid: str | None = None
    session_id: str | None = None
    buf = TwilioStreamBuffer()
    try:
        while True:
            msg = await ws.receive_json()
            event = msg.get("event")
            if event == "start":
                start = msg.get("start") or {}
                stream_sid = start.get("streamSid")
                call_sid = start.get("callSid") or stream_sid or "unknown"
                session_id = f"telephony-{call_sid}"
                logger.info("telephony: call started (%s)", session_id)
            elif event == "media" and stream_sid and session_id:
                payload_b64 = (msg.get("media") or {}).get("payload", "")
                try:
                    mulaw = base64.b64decode(payload_b64)
                except Exception:  # noqa: BLE001
                    continue
                buf.add_chunk(ulaw_to_pcm16(mulaw))
                if buf.should_flush():
                    await _handle_telephony_turn(ws, stream_sid, session_id, buf.flush())
            elif event == "stop":
                logger.info("telephony: call ended (%s)", session_id)
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        logger.exception("telephony media-stream error: %s", e)


# ---------------------------------------------------------------------------
# Call end: drop an event bubble + store a one-line call summary as a memory
# ---------------------------------------------------------------------------
class CallEnd(BaseModel):
    session_id: str = Field(..., min_length=1)
    duration_seconds: int = Field(..., ge=0)


def _fmt_duration(seconds: int) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


async def _summarize_call(session_id: str, duration_label: str):
    """Summarize the just-ended call into one memory line (background task)."""
    turns = state.db.get_recent_messages(session_id, 30)
    if not turns:
        return
    transcript = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    try:
        raw = await state.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize this phone call in ONE short third-person "
                        "sentence to remember later (what they talked about / how "
                        "it felt). No preamble."
                    ),
                },
                {"role": "user", "content": transcript},
            ],
            options={"temperature": 0.3},
        )
        summary = raw.strip().split("\n")[0][:200]
    except Exception as e:  # noqa: BLE001
        logger.warning("Call summary failed: %s", e)
        return
    if summary:
        await state.memory.add_fact(
            f"Phone call ({duration_label}): {summary}",
            "event",
            kind="event",
            source="webapp_call",
        )


@app.post("/call/end")
async def call_end(body: CallEnd, background_tasks: BackgroundTasks):
    label = _fmt_duration(body.duration_seconds)
    # Event bubble in the chat history (excluded from LLM context by db filter).
    state.db.ensure_session(body.session_id)
    state.db.add_message(body.session_id, "event", f"Call ended · {label}", source="webapp_call")
    background_tasks.add_task(_summarize_call, body.session_id, label)
    return {"ok": True, "duration": label}


# ---------------------------------------------------------------------------
# File-mode TTS audio (voice notes for chat + WhatsApp)
# ---------------------------------------------------------------------------
_AUDIO_NAME = re.compile(r"^[a-f0-9]{32}\.wav$")


@app.get("/audio/{fname}")
async def get_audio(fname: str):
    """Serve a synthesized voice-note WAV by id (alias of /media/tts/...)."""
    if not _AUDIO_NAME.match(fname):
        raise HTTPException(status_code=404, detail="Not found")
    path = TTS_DIR / fname
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="audio/wav")


# ---------------------------------------------------------------------------
# WhatsApp bridge integration
# ---------------------------------------------------------------------------
class WhatsAppIncoming(BaseModel):
    text: str = Field(..., min_length=1)
    # Burst batch from the bridge's debounce: the individual rapid-fire
    # messages (oldest first). `text` stays as the joined fallback so an
    # older bridge without batching keeps working unchanged.
    texts: list[str] | None = None
    # Which number sent this - selects the persona/profile that answers (see
    # app/profiles.py). Unknown/missing falls back to the default profile.
    from_number: str | None = None


class WhatsAppMedia(BaseModel):
    kind: str = Field(..., pattern="^(image|voice)$")
    data_b64: str = Field(..., min_length=1)
    mimetype: str | None = None
    caption: str | None = None  # accompanying text sent with an image, if any
    from_number: str | None = None  # selects the profile, same as /incoming


_CANT_SEE_PHOTOS = [
    "aww i can't see pics on here yet 😭 what is it?",
    "ugh my phone's being weird with photos rn - describe it to me??",
    "can't open pics on this thing 😩 tell me what it is!",
]


async def _caption_image(data_b64: str) -> str | None:
    """One-line description via a local vision model (Ollama; config
    tools.vision.model, default 'moondream'). Returns None if that model
    isn't pulled/available - callers fall back to an honest deflection,
    never a guess about what's in the photo."""
    model = ((state.settings.raw or {}).get("tools", {}).get("vision", {})
            .get("model", "moondream"))
    try:
        raw = await state.llm.chat(
            messages=[{"role": "user",
                      "content": "Describe this photo in one short, plain sentence "
                                 "- just what's actually in it.",
                      "images": [data_b64]}],
            model=model, options={"temperature": 0.2},
        )
        return raw.strip() or None
    except Exception as e:  # noqa: BLE001 - model not pulled, Ollama error, etc.
        logger.info("vision captioning unavailable (%s) - deflecting", e)
        return None


@app.post("/whatsapp/incoming-media")
async def whatsapp_incoming_media(body: WhatsAppMedia):
    """A WhatsApp photo or voice note - turned into a normal (synthetic) text
    message and handed to the SAME pipeline as /whatsapp/incoming, so
    routing/guards/stickers/voice-roll/multi-bubble all apply for free."""
    profile = state.profiles.by_number(body.from_number or "") or state.profiles.default
    _current_profile.set(profile)
    if body.kind == "voice":
        if not state.stt.available:
            text = random.choice(_CANT_SEE_PHOTOS)  # same honest-deflection spirit
            profile.db.ensure_session(profile.session_id)
            profile.db.add_message(profile.session_id, "assistant", text,
                                   source="whatsapp")
            return {"mode": "text", "text": text, "texts": [text], "wav_path": None,
                    "audio_url": None, "sticker_path": None, "sticker_url": None,
                    "emotion": "neutral"}
        audio = base64.b64decode(body.data_b64)
        transcribed = await asyncio.to_thread(state.stt.transcribe, audio)
        synthetic = transcribed.strip() or "[sent a voice note but I couldn't make out any words]"
    else:  # image
        description = await _caption_image(body.data_b64)
        if description is None:
            text = random.choice(_CANT_SEE_PHOTOS)
            profile.db.ensure_session(profile.session_id)
            profile.db.add_message(profile.session_id, "assistant", text,
                                   source="whatsapp")
            return {"mode": "text", "text": text, "texts": [text], "wav_path": None,
                    "audio_url": None, "sticker_path": None, "sticker_url": None,
                    "emotion": "neutral"}
        caption_suffix = f" {body.caption}" if body.caption else ""
        synthetic = f"[sent a photo - looks like: {description}]{caption_suffix}"
    return await whatsapp_incoming(
        WhatsAppIncoming(text=synthetic, from_number=body.from_number))


_CALL_EXCUSES = [
    "ahh I can't pick up on here 😭 call me on our app??",
    "nooo I missed you - whatsapp calls don't work for me, use our app? 🥺",
    "can't answer calls on here!! open the app and call me there 💕",
    "omg I saw you calling 😩 I can only talk on our app, come there?",
]


@app.post("/whatsapp/incoming")
async def whatsapp_incoming(body: WhatsAppIncoming):
    """Normal chat pipeline for a WhatsApp message; returns the reply for the
    bridge to deliver. ~voice_reply_ratio of replies come back as voice notes.

    The sender's number selects the PROFILE (persona + its own db/memory/
    relationship) - see app/profiles.py. Everything below then runs against
    that profile only, so two people on the same WhatsApp account get two
    independent women who share nothing.
    """
    profile = state.profiles.by_number(body.from_number or "") or state.profiles.default
    _current_profile.set(profile)
    session_id = profile.session_id
    db = profile.db

    # Burst batching: the bridge debounces rapid-fire texts and hands them
    # over as one batch, so she answers the whole thought with ONE reply
    # instead of replying line-by-line. Each message still gets its own
    # history row (the web-app mirror shows them as separate bubbles);
    # everything downstream (routing, retrieval, guards, extraction) sees
    # the combined text.
    incoming = [t.strip() for t in (body.texts or [body.text]) if t and t.strip()]
    combined = "\n".join(incoming) or body.text

    # Keyword commands (/closeness, /mood, /persona, ...) are mechanical: they
    # run against THIS profile only, never reach the model, and are not stored
    # as conversation or memory.
    if is_command(combined):
        result = handle_command(combined, profile,
                                list_persona_ids(state.settings.persona_folder))
        if result.switch_persona:
            try:
                _reload_profile_persona(profile, result.switch_persona)
            except Exception as e:  # noqa: BLE001
                logger.exception("persona switch failed")
                result.reply = f"couldn't switch persona: {e}"
        return {"mode": "text", "text": result.reply, "texts": [result.reply],
                "wav_path": None, "audio_url": None, "sticker_path": None,
                "sticker_url": None, "emotion": "neutral"}

    db.ensure_session(session_id)
    for t in incoming or [body.text]:
        db.add_message(session_id, "user", t, source="whatsapp")
    # Parity with web chat: exchange_count only used to increment here on
    # webapp_chat, so offer-throttling gaps and _pattern_note's cadence were
    # computed on a counter that never moved for a WhatsApp-heavy user, and
    # _offer_note was never even called below - she could offer on web but
    # never on WhatsApp.
    db.set_setting("exchange_count",
                   str(int(db.get_setting("exchange_count") or 0) + 1))
    _track_offer_decline(combined)
    followup_note = _awaiting_followup_note(session_id)
    repair_note = await _maybe_repair_note(combined)
    nonsense_note = _nonsense_note(combined)

    urgent = bool(_URGENT.search(combined))
    brushoff = await _maybe_busy_brushoff(session_id, combined, urgent)
    if brushoff:
        db.add_message(session_id, "assistant", brushoff, source="whatsapp")
        if not nonsense_note:
            _spawn(profile.memory.extract_and_store(combined, brushoff, source="whatsapp"))
        return {"mode": "text", "text": brushoff, "texts": [brushoff], "wav_path": None,
                "audio_url": None, "sticker_path": None, "sticker_url": None,
                "emotion": "neutral"}

    # Same router + guards as web chat (mention-vs-request enforced everywhere).
    mem_task = None if nonsense_note else \
        asyncio.create_task(profile.memory.retrieve_memories(combined))
    route = await state.router.route(combined)
    tool_note = None
    tool_ran = False
    if route.kind == "tool":
        res = await profile.tools.call(route.tool, route.args)
        tool_ran = res.get("ok", False)
        if res.get("song"):
            # Library hit is instant, but the reply text still has to go out
            # first - this used to be silently dropped here (only the
            # queue_query/miss path below was ever wired up for WhatsApp), so
            # she'd say "sending it now" and then nothing arrived.
            _spawn(_wa_send_song(res["song"], to_number=profile.number))
        if res.get("image"):
            _spawn(_wa_send_image(res["image"], to_number=profile.number))
        if res.get("queue_query"):
            _queue_cover_request(res["queue_query"], session_id)
        tool_note = _TOOL_NOTE.format(result=res["result"]) if tool_ran else (
            f"You tried to {route.tool} but it didn't work: {res['result']}. "
            f"Be honest, stay casual, never pretend it succeeded.")
    elif route.kind == "deep":
        facts, failure = await _run_deep(combined, session_id)
        if facts:
            tool_ran = True
            tool_note = _DEEP_NOTE.format(facts=facts)
        elif failure:
            db.add_message(session_id, "assistant", failure, source="whatsapp")
            _spawn(profile.memory.extract_and_store(combined, failure, source="whatsapp"))
            return {"mode": "text", "text": failure, "texts": [failure], "wav_path": None,
                    "audio_url": None, "sticker_path": None, "sticker_url": None,
                    "emotion": "neutral"}

    try:
        memories = (await mem_task) if mem_task else []
    except Exception:  # noqa: BLE001 - retrieval must never break chat
        memories = []
    day_note = None
    try:
        day_note = await profile.daylife.prompt_note()
    except Exception:  # noqa: BLE001
        pass
    # Ask for the trailing emotion tag - it drives sticker choice, then gets
    # stripped before anything is stored or sent.
    turn_directive = _turn_directive(session_id, combined, memories, "chat")
    system_prompt = build_system_prompt(
        profile.persona,
        memories,
        current_time=_now_context(),
        extra_notes=_relationship_notes(CAPABILITY_MANIFEST, day_note,
                                        EMOTION_TAG_INSTRUCTION, turn_directive,
                                        _offer_note(combined), _pattern_note(combined),
                                        _streak_note(combined), _unresolved_note(combined),
                                        tool_note, followup_note, repair_note,
                                        nonsense_note, mood_note(db)),
        stage=_stage(),
    )
    messages = [
        {"role": "system", "content": system_prompt},
        *_sanitized_recent(session_id),
    ]
    try:
        # Same generate -> scan -> regenerate-once -> surgical-fix guard flow
        # as web chat (previously WhatsApp only stripped violating sentences
        # with no regeneration attempt, so a caught violation still shipped).
        # Overall cap: _guarded_reply can be TWO chat calls (generate +
        # guard-retry) - under a stalled GPU that's 2x the httpx timeout
        # serially before the fallback would fire. 90s total is far beyond
        # any healthy generation.
        reply, emotion = await asyncio.wait_for(
            _guarded_reply(messages, tool_ran, session_id=session_id,
                           user_message=combined), timeout=90.0)
    except Exception:  # noqa: BLE001
        # NEVER 502 the bridge: a 502 means the bridge sends NOTHING and she
        # just ghosts them mid-conversation (observed during a GPU thrash:
        # three 120s LLM ReadTimeouts -> three 502s -> "??" / "are you
        # there?"). A short in-character "phone's acting up" line keeps her
        # present; the moment the box recovers, normal replies resume.
        logger.exception("WhatsApp reply generation failed - sending laggy-phone line")
        text = random.choice(_LAGGY_PHONE)
        db.add_message(session_id, "assistant", text, source="whatsapp")
        return {"mode": "text", "text": text, "texts": [text], "wav_path": None,
                "audio_url": None, "sticker_path": None, "sticker_url": None,
                "emotion": "neutral"}

    # --- sticker roll: probability scales with relationship stage; never two
    # in a row; occasionally the sticker IS the whole reply. ---
    sticker_path = None
    sticker_url = None
    sticker_only = False
    prob = profile.relationship.sticker_probability() if profile.relationship else 0.0
    last_had = db.get_setting("last_reply_had_sticker") == "1"
    if prob > 0 and not last_had and random.random() < prob:
        picked = pick_sticker(emotion)
        if picked:
            sticker_path, sticker_url = str(picked[0]), picked[1]
            # Message + sticker together is the default feel; sticker-only
            # (replacing the text entirely) stays the rare exception.
            sticker_only = random.random() < 0.10
    db.set_setting("last_reply_had_sticker", "1" if sticker_path else "0")

    # --- persist: one row per bubble (unless sticker-only), then a sticker row ---
    # Same "texted twice" split as web chat (_split_bubbles) - a blank line
    # in her reply means she'd genuinely have sent 2-3 separate messages.
    bubbles = [] if sticker_only else _split_bubbles(reply)
    message_id = None
    for bubble in bubbles:
        message_id = db.add_message(session_id, "assistant", bubble, source="whatsapp")
    if sticker_url:
        db.add_message(session_id, "assistant", "", sticker_url=sticker_url, source="whatsapp")
    if not nonsense_note:  # junk exchanges must never become "facts"
        _spawn(profile.memory.extract_and_store(combined, reply, source="whatsapp"))

    # --- voice-note roll (text replies only; only the LAST bubble gets a
    # voice note - TTS-ing every short bubble in a multi-part text would be
    # excessive, and a voice note as the final word is the natural shape) ---
    # A reply carrying a real link (e.g. the zomato_suggest tool's Zomato
    # search URL) must never be read aloud - a spoken URL is meaningless and
    # the link itself would be lost. Those replies always stay text.
    has_link = "http://" in reply or "https://" in reply
    wav_path = None
    audio_url = None
    # Never two voice notes in a row (mirrors the sticker rule): a run of
    # lucky rolls otherwise turns her into voice-notes-only, which reads as
    # a glitch, not a person (observed: five consecutive voice replies).
    last_was_voice = db.get_setting("last_reply_was_voice") == "1"
    if (bubbles and not has_link and not last_was_voice
            and random.random() < state.settings.wa_voice_ratio):
        try:
            speak = strip_for_speech(bubbles[-1])
            if speak:
                # Her cloned voice (studio + RVC) first - this used to go
                # straight to plain Kokoro and never even tried the trained
                # voice, so every WhatsApp voice note sounded like the
                # default stock voice regardless of RVC/XTTS being set up.
                # Mirrors the /tts endpoint's studio-first-then-fallback path.
                if state.studio and state.studio.available:
                    try:
                        samples, sr = await _studio_render(speak, emotion)
                        import io
                        import soundfile as sf
                        buf = io.BytesIO()
                        sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
                        audio_url = _write_tts_wav(buf.getvalue())
                        wav_path = str(TTS_DIR / Path(audio_url).name)
                        if message_id is not None:
                            db.set_message_audio(message_id, audio_url)
                    except Exception as e:  # noqa: BLE001 - fall back to kokoro below
                        logger.warning("studio render failed, falling back to kokoro: %s", e)
                if not audio_url and state.tts.available:
                    result = await asyncio.to_thread(
                        state.tts.synth, speak, _persona_voice()
                    )
                    if len(result.samples) > 0:
                        # The Kokoro fallback must still pass through RVC so
                        # the voice note is HER voice - calls and /tts already
                        # did this, but this path shipped raw stock Kokoro,
                        # so every WhatsApp voice note while XTTS couldn't
                        # load (VRAM) came out in the wrong voice.
                        if state.rvc and state.rvc.available:
                            try:
                                converted, _ms = await asyncio.to_thread(
                                    state.rvc.convert, result.samples,
                                    result.sample_rate)
                                result.samples = converted
                            except Exception as e:  # noqa: BLE001 - raw kokoro beats silence
                                logger.warning("whatsapp voice: rvc failed, "
                                               "sending raw kokoro: %s", e)
                        audio_url = _write_tts_wav(result.to_wav_bytes())
                        wav_path = str(TTS_DIR / Path(audio_url).name)
                        if message_id is not None:
                            db.set_message_audio(message_id, audio_url)
        except Exception as e:  # noqa: BLE001 - voice is optional; fall back to text
            logger.warning("WhatsApp voice synth failed, sending text: %s", e)
            wav_path = None

    mode = "sticker_only" if sticker_only else ("voice" if wav_path else "text")
    db.set_setting("last_reply_was_voice", "1" if mode == "voice" else "0")
    return {
        "mode": mode,
        "text": "" if sticker_only else (bubbles[-1] if bubbles else reply),
        # Every bubble in order - the bridge sends each as its own WhatsApp
        # message with a pause between; a single-bubble reply is just a
        # one-element list, so this replaces `text` as the send source.
        "texts": bubbles,
        "wav_path": wav_path,
        "audio_url": audio_url,
        "sticker_path": sticker_path,
        "sticker_url": sticker_url,
        "emotion": emotion,
    }


class WhatsAppCallRejected(BaseModel):
    from_number: str | None = None  # selects the profile, same as /incoming


@app.post("/whatsapp/call-rejected")
async def whatsapp_call_rejected(body: WhatsAppCallRejected | None = None):
    """A real WhatsApp call came in; the bridge auto-rejected it. Log it, store
    a memory, and hand back an in-character 'can't pick up' text."""
    profile = (state.profiles.by_number((body.from_number if body else "") or "")
               or state.profiles.default)
    _current_profile.set(profile)
    session_id = profile.session_id
    now_label = datetime.now().strftime("%I:%M %p").lstrip("0")

    profile.db.ensure_session(session_id)
    profile.db.add_message(
        session_id, "event", f"Missed WhatsApp call · {now_label}", source="whatsapp"
    )
    _spawn(
        profile.memory.add_fact(
            f"User tried to call her on WhatsApp at {now_label} on "
            f"{datetime.now():%B %d} and she couldn't pick up.",
            "event",
            kind="event",
            source="whatsapp",
        )
    )

    # In-character, varied reply; LLM first, canned fallback.
    text = random.choice(_CALL_EXCUSES)
    try:
        system_prompt = build_system_prompt(
            profile.persona, None, current_time=_now_context(), stage=_stage()
        )
        raw = await state.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "[They just tried to CALL you on WhatsApp but you can never "
                        "answer calls there.] Send ONE short flustered-but-affectionate "
                        "text apologizing you can't pick up on WhatsApp and telling them "
                        "to call you on your own app instead. Vary the wording."
                    ),
                },
            ],
            options={"temperature": 1.0},
        )
        cleaned = strip_tags(raw)
        if cleaned:
            text = cleaned
    except Exception as e:  # noqa: BLE001
        logger.warning("Call-reject reply generation failed, using canned: %s", e)

    profile.db.add_message(session_id, "assistant", text, source="whatsapp")
    return {"text": text}


# ---------------------------------------------------------------------------
# Proactive / brain-status / voice-bench / voice-status / day-state /
# relationship / health endpoints - extracted to app/api/{proactive,status,
# voice,relationship}.py (N2 phase 2, round 2). _training_progress is still
# used below by lifespan's studio.prefer_cpu, imported from app.api.voice.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Static media (TTS wavs, avatars) - mounted before the SPA catch-all.
# ---------------------------------------------------------------------------
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
STICKER_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/stickers", StaticFiles(directory=str(STICKER_ROOT)), name="stickers")
(ROOT / "songs" / "library").mkdir(parents=True, exist_ok=True)
app.mount("/songs", StaticFiles(directory=str(ROOT / "songs" / "library")), name="songs")


# ---------------------------------------------------------------------------
# Static frontend (mounted LAST so API routes take precedence)
# ---------------------------------------------------------------------------
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="spa")
    logger.info("Serving frontend from %s", FRONTEND_DIST)
else:
    logger.warning(
        "Frontend build not found at %s - run `npm --prefix frontend install && "
        "npm --prefix frontend run build`. API still available.",
        FRONTEND_DIST,
    )

    @app.get("/")
    async def _no_frontend():
        return {
            "status": "backend running",
            "note": "Frontend not built. See README (build the frontend).",
        }
