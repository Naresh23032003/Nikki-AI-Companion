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
import json
import logging
import mimetypes
import random
import re
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ROOT, load_settings
from app.dayseed import DayLife
from app.db import Database
from app.guards import (
    CAPABILITY_MANIFEST,
    CLAIM_PATTERNS,
    HONEST_LINE,
    REFUSAL_DEFLECT,
    REFUSAL_PATTERNS,
    STATUS_PATTERNS,
    guard_stats,
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
from app.memory import VALID_CATEGORIES, MemoryStore
from app.persona import build_system_prompt, list_persona_ids, load_persona
from app.profiles import Profile, ProfileRegistry, load_profiles
from app.relationship import STAGES, RelationshipTracker
from app.stickers import STICKER_ROOT, ensure_dirs as ensure_sticker_dirs, pick_sticker
from app.stt import STTEngine
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


class MemoryCreate(BaseModel):
    fact: str = Field(..., min_length=1)
    category: str = Field(default="personal_info")


class ActivePersona(BaseModel):
    id: str = Field(..., min_length=1)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    message_id: int | None = None  # attach the WAV url to this stored message
    voice: str | None = None       # override the persona's voice
    emotion: str | None = None     # picks ref/<emotion> clip / studio preset


# ---------------------------------------------------------------------------
# App state / lifespan
# ---------------------------------------------------------------------------
class AppState:
    settings = None
    persona = None
    db: Database | None = None
    llm: OllamaClient | None = None
    memory: MemoryStore | None = None
    stt: STTEngine | None = None
    tts: TTSEngine | None = None
    proactive = None  # ProactiveEngine
    relationship: RelationshipTracker | None = None
    brain: CloudBrain | None = None
    router: Router | None = None
    tools: ToolRunner | None = None
    tool_ctx: ToolContext | None = None
    daylife: DayLife | None = None
    rvc: RVCConverter | None = None
    studio: StudioTTS | None = None
    covers: CoverPipeline | None = None
    gpu_queue: GPUJobQueue | None = None
    profiles: ProfileRegistry | None = None


state = AppState()

# The profile (persona + its isolated db/memory/relationship) this request
# belongs to. WhatsApp sets it per incoming message from the sender's number;
# everything else (web app, calls) leaves it unset and gets the default
# profile, which IS state.db/state.persona - i.e. unchanged behavior.
_current_profile: ContextVar[Profile | None] = ContextVar("current_profile", default=None)


def P() -> Profile:
    """The profile serving this request. Never None once startup has run."""
    p = _current_profile.get()
    if p is not None:
        return p
    return state.profiles.default if state.profiles else None


def _persona_voice() -> str | None:
    """The active persona's voice, or None to fall back to the default."""
    p = P()
    persona = p.persona if p else state.persona
    return (persona.voice or None) if persona else None


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
def _persona_public(persona) -> dict:
    return {
        "id": persona.id,
        "name": persona.name,
        "age": persona.age,
        "avatar_id": persona.avatar_id,
        "relationship_context": persona.relationship_context,
        # Cache-busted so a freshly uploaded photo shows immediately.
        "photo_url": f"/personas/{persona.id}/photo",
    }


def _resolve_photo_path(persona_id: str) -> Path:
    """Find the profile photo file for a persona.

    Precedence: uploaded override (DB) -> YAML profile_pic -> default avatar.
    """
    override = state.db.get_setting(f"profile_pic:{persona_id}")
    candidates = []
    if override:
        candidates.append(Path(override))
    try:
        p = load_persona(state.settings.persona_folder, persona_id)
        if p.profile_pic:
            candidates.append(Path(p.profile_pic))
    except FileNotFoundError:
        pass

    for c in candidates:
        path = c if c.is_absolute() else (ROOT / c)
        if path.exists():
            return path
    return DEFAULT_AVATAR


def _sse(data: str, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"


def _now_context() -> str:
    """Human-readable local date+time for the system prompt, e.g.
    'Saturday, July 11, 12:14 PM (afternoon)'. Weekday, date and time all
    come from this ONE line - when the date lived in a different note the
    model couldn't bind them and would get the weekday wrong."""
    from datetime import datetime

    now = datetime.now()
    h = now.hour
    tod = (
        "the middle of the night" if h < 4 else
        "early morning" if h < 7 else
        "morning" if h < 12 else
        "afternoon" if h < 17 else
        "evening" if h < 21 else
        "night"
    )
    return (f"{now.strftime('%A')}, {now.strftime('%B')} {now.day}, "
            f"{now.strftime('%I:%M %p').lstrip('0')} ({tod})")


def _in_quiet_hours(now: datetime | None = None) -> bool:
    """behavior.quiet_hours ('01:00-07:30' style) - she's 'asleep'. Used to
    hold self-initiated deliveries (reminders, deferred answers, event
    follow-ups) until it's over instead of firing at 3am; the item stays
    queued and fires on the next scheduler tick after the window ends."""
    beh = state.settings.behavior or {}
    raw = (beh.get("quiet_hours") or "").strip()
    if not raw:
        return False
    try:
        s, e = raw.split("-", 1)
        sh, sm = (int(x) for x in s.strip().split(":"))
        eh, em = (int(x) for x in e.strip().split(":"))
    except (ValueError, AttributeError):
        return False
    start, end = dtime(sh, sm), dtime(eh, em)
    t = (now or datetime.now()).time()
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end  # window crossing midnight


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
    system_prompt = build_system_prompt(
        state.persona, memories, mode=mode,
        current_time=_now_context(),
        extra_notes=_relationship_notes(CAPABILITY_MANIFEST, day_note,
                                        _offer_note(query), _pattern_note(query),
                                        _streak_note(query), tool_note, extra_note),
        stage=_stage(),
    )
    return [{"role": "system", "content": system_prompt},
            *_sanitized_recent(session_id)]


def _awaiting_followup_note(session_id: str) -> str | None:
    """If she's waiting on an answer to an event check-in ('how did it go?'),
    resolve it now - the NEXT user message is treated as the answer (see
    app/db.py get_awaiting_followup's docstring for the intended design) -
    and tell her this reply is likely that answer so she doesn't re-ask."""
    awaiting = P().db.get_awaiting_followup(session_id)
    if not awaiting:
        return None
    P().db.resolve_event_followup(awaiting["id"])
    return (f"NOTE: You recently asked them about '{awaiting['event_fact']}' - this "
            f"message is likely their answer. React naturally to what they say now; "
            f"don't ask again.")


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
    """Offer throttling: after a relevant MENTION she may offer an action -
    rate-limited, stage-gated, and permanently dropped once declined."""
    beh = state.settings.behavior or {}
    eagerness = float(beh.get("eagerness", 0.2))
    gap = int(beh.get("offer_min_gap", 6))
    stage = _stage() or "stranger"
    if stage in ("stranger", "acquaintance"):
        return None
    topics = {
        "food": r"\b(hungry|starving|haven'?t eaten|no food|skip(ping)? (lunch|dinner))\b",
        "weather": r"\b(so (hot|cold)|freezing|melting|raining|weather)\b",
        "money": r"\b(broke|money'?s tight|expensive|overspent)\b",
    }
    topic = next((t for t, p in topics.items() if re.search(p, message, re.I)), None)
    if not topic:
        return None
    try:
        declined = set(json.loads(P().db.get_setting("declined_offers") or "[]"))
    except json.JSONDecodeError:
        declined = set()
    if topic in declined:
        return None
    n = int(P().db.get_setting("exchange_count") or 0)
    last = int(P().db.get_setting("last_offer_at") or -999)
    if n - last < gap or random.random() > eagerness:
        return None
    P().db.set_setting("last_offer_at", str(n))
    # _track_offer_decline() reads this back if the next message declines -
    # without it, a decline always recorded the literal string "last"
    # instead of the actual topic, so "permanently dropped once declined"
    # silently never worked.
    P().db.set_setting("last_offer_topic", topic)
    return (f"They just mentioned something about {topic}. Respond like a person "
            f"first (empathy/teasing/curiosity). You MAY casually offer to help "
            f"with it mid-conversation if it feels natural - one soft offer, "
            f"never as your opening line, and drop it instantly if declined.")


def _pattern_note(message: str) -> str | None:
    """Occasional, throttled reference to a weekly-detected mood-journal
    pattern (category="relationship" memories prefixed "Pattern: ") - the
    girlfriend part of the journal: gentle noticing, never a report. Gated by
    the same eagerness dial as _offer_note, but content-independent and much
    rarer (patterns aren't tied to any one message)."""
    stage = _stage() or "stranger"
    if stage in ("stranger", "acquaintance"):
        return None
    beh = state.settings.behavior or {}
    eagerness = float(beh.get("eagerness", 0.2))
    gap = int(beh.get("offer_min_gap", 6)) * 4
    # Cheap gate check BEFORE the DB query - this runs on every message but
    # the roll usually says no, so querying memories first was pure waste.
    n = int(P().db.get_setting("exchange_count") or 0)
    last_n = int(P().db.get_setting("last_pattern_ref_at") or -999)
    if n - last_n < gap or random.random() > eagerness * 0.4:
        return None
    patterns = [m for m in P().db.list_memories_by_category("relationship")
               if (m.get("fact") or "").startswith("Pattern:")]
    if not patterns:
        return None
    last_id = int(P().db.get_setting("last_pattern_ref_id") or 0)
    candidate = next((p for p in patterns if p["id"] != last_id), patterns[0])
    P().db.set_setting("last_pattern_ref_at", str(n))
    P().db.set_setting("last_pattern_ref_id", str(candidate["id"]))
    fact = candidate["fact"][len("Pattern:"):].strip()
    return (f"NOTE: from quietly paying attention over time you've noticed this about "
            f"them: {fact}. You MAY bring it up naturally if the moment fits - as one "
            f"gentle, caring observation, never as a report/stats/list, never mentioning "
            f"a 'journal' or that you track anything. Skip it entirely if it doesn't fit.")


def _streak_note(message: str) -> str | None:
    """A SHORT-TERM rough-streak flag (run_recent_streak_check, chained onto
    the NIGHTLY job - distinct from the long-term weekly Pattern: system
    above). Surfaces promptly: a short gap (not _pattern_note's *4 throttle),
    since 'you've seemed off the last few days' is time-sensitive - bringing
    it up two weeks late would feel odd. One-shot: consumed (deleted) the
    moment it's used, unlike a genuine Pattern: which stays referenceable."""
    stage = _stage() or "stranger"
    if stage in ("stranger", "acquaintance"):
        return None
    beh = state.settings.behavior or {}
    eagerness = float(beh.get("eagerness", 0.2))
    gap = int(beh.get("offer_min_gap", 6))
    n = int(P().db.get_setting("exchange_count") or 0)
    last_n = int(P().db.get_setting("last_streak_ref_at") or -999)
    if n - last_n < gap or random.random() > eagerness:
        return None
    streaks = [m for m in P().db.list_memories_by_category("relationship")
              if (m.get("fact") or "").startswith("Streak:")]
    if not streaks:
        return None
    candidate = streaks[0]
    P().db.set_setting("last_streak_ref_at", str(n))
    P().db.delete_memory(candidate["id"])
    P().memory.remove(candidate["id"])
    fact = candidate["fact"][len("Streak:"):].strip()
    return (f"NOTE: you've quietly noticed this about how they've been the last few "
            f"days: {fact}. Bring it up naturally as one gentle, caring check-in if "
            f"the moment fits - never as a report, never mentioning a 'journal' or "
            f"that you track anything. Skip it entirely if it doesn't fit right now.")


# ---------------------------------------------------------------------------
# Nonsense/spam realism: 30x "rrrrrr" used to make her confabulate an entire
# fake evening (invented plans, times, random memory fragments) because the
# model had no signal the input was noise. A real person notices immediately.
# ---------------------------------------------------------------------------
_LETTERS = re.compile(r"[a-zA-Z]+")
# Real texting tokens that survive collapsing but have no vowel (or are
# single letters) - must never count as keyboard-mash.
_SHORT_REAL = {"i", "u", "y", "k", "hm", "mhm", "ty", "np", "gm", "gn",
               "idk", "tbh", "btw", "rn", "pls", "plz", "thx", "xd",
               "shh", "psst", "tsk", "brb", "wtf", "smh", "fr", "ngl"}


def _is_gibberish(text: str) -> bool:
    """Keyboard-mash detector: no plausible word in the message. Pure
    emoji/punctuation ('??', '😂😂') is a real signal, NOT gibberish.
    Elongations ('noooo', 'hmmmm') collapse first so they read as words."""
    t = text.strip().lower()
    if not t or not re.search(r"[a-zA-Z]", t):
        return False
    for w in _LETTERS.findall(t):
        w = re.sub(r"(.)\1{2,}", r"\1", w)  # nooo -> no, hmmm -> hm
        if w in _SHORT_REAL or (len(w) >= 2 and set(w) & set("aeiou")):
            return False
    return True


def _nonsense_note(message: str) -> str | None:
    """Track consecutive gibberish/identical messages and hand the model a
    human way out. Returns a prompt note while a streak is active."""
    norm = re.sub(r"\s+", " ", message.strip().lower())
    last = P().db.get_setting("last_user_msg") or ""
    P().db.set_setting("last_user_msg", norm)
    gib = _is_gibberish(message)
    repeat = bool(norm) and norm == last
    streak = int(P().db.get_setting("nonsense_streak") or 0)
    streak = streak + 1 if (gib or repeat) else 0
    P().db.set_setting("nonsense_streak", str(streak))
    if streak == 0:
        return None
    what = "keyboard-mash gibberish" if gib else "the exact same message again"
    if streak >= 4:
        return (
            f"NOTE: they've now sent {what} {streak} times in a row. Stop playing "
            "along like it means something: reply with ONE very short dry line "
            "('ok you're clearly just mashing your keyboard 😂' / '...say something "
            "real and i'll answer'). Do NOT invent any events, people, plans or "
            "times. No questions.")
    return (
        f"NOTE: their message is just {what}. React like a real person would - "
        "confused or teasing ('did your cat walk on your keyboard?'), one short "
        "line. Do NOT treat it as meaningful, and do NOT invent plans, people, "
        "times or topics to fill the silence.")


def _track_offer_decline(message: str) -> None:
    """If she offered last turn and this reply is a decline, drop that topic."""
    if int(P().db.get_setting("last_offer_at") or -1) != \
       int(P().db.get_setting("exchange_count") or 0) - 1:
        return
    if re.match(r"^\s*(no+|nah|nope|don'?t|it'?s ok(ay)?|i'?m (fine|good))\b",
                message.strip(), re.I):
        try:
            declined = set(json.loads(P().db.get_setting("declined_offers") or "[]"))
        except json.JSONDecodeError:
            declined = set()
        declined.add(P().db.get_setting("last_offer_topic") or "last")
        P().db.set_setting("declined_offers", json.dumps(sorted(declined)))


def _persona_other_names(persona) -> list[str]:
    """Names of people from HER OWN backstory (life.friends) - never valid
    names for the person she's actually texting. See scan_identity_confusion."""
    friends = getattr(persona, "life", None) or {}
    return [f.get("name", "") for f in (friends.get("friends") or [])
            if isinstance(f, dict) and f.get("name")]


async def _guarded_reply(messages: list[dict], tool_ran: bool) -> tuple[str, str]:
    """Generate fully, run the guards, regenerate once on violation, then
    surgically fix anything left. Nothing reaches the user unvetted.

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
    if claims or speak or pots or refusal or identity:
        correction = (
            "CORRECTION NOTE: your draft violated the rules ("
            + ", ".join([*claims, *speak, *pots, *refusal, *identity][:4]) +
            "). Rewrite it: never claim actions you didn't do, no specific "
            "prices/news facts without a tool result, no assistant phrasing or "
            "lists, at most one question. You NEVER refuse or break character - "
            "if the ask is awkward, deflect/tease/redirect as HER, a person, "
            "never as an AI declining a request. The person you're texting has "
            "NO name unless they told you one in this chat - never call them by "
            "a name from your own life/friends/backstory. Short and natural.")
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

_MAX_BUBBLES = 4

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
_AUTO_SPLIT_MIN_CHARS = 80
_MIN_BUBBLE_CHARS = 12  # "right?" / "haha" rides along with its neighbor
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def _sentence_bubbles(text: str) -> list[str]:
    """One bubble per sentence; tiny fragments fold into the previous one.
    Replies with more sentences than _MAX_BUBBLES get the sentences grouped
    into _MAX_BUBBLES roughly length-balanced bubbles - folding all overflow
    into the LAST bubble re-created the exact wall-of-text this exists to
    prevent (observed: a 333-char final bubble)."""
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    if len(sentences) < 2:
        return [text]
    bubbles: list[str] = []
    for s in sentences:
        if bubbles and len(s) < _MIN_BUBBLE_CHARS:
            bubbles[-1] = f"{bubbles[-1]} {s}"
        else:
            bubbles.append(s)
    if len(bubbles) <= _MAX_BUBBLES:
        return bubbles
    per_bubble = sum(len(b) for b in bubbles) / _MAX_BUBBLES
    grouped, current = [], ""
    for b in bubbles:
        if (current and len(grouped) < _MAX_BUBBLES - 1
                and len(current) + len(b) > per_bubble * 1.15):
            grouped.append(current)
            current = b
        else:
            current = f"{current} {b}".strip()
    if current:
        grouped.append(current)
    return grouped


def _split_bubbles(text: str) -> list[str]:
    """Split a reply into separate text bubbles.

    Primary signal: blank lines - what _BEHAVIOR_RULES tells her to use for
    'texted twice' (a reaction, then the real thought). Small local models
    rarely emit that signal though, so multi-sentence replies over ~80 chars
    additionally get split one-sentence-per-bubble - a thought per send,
    like real texting. Short replies stay one bubble."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(parts) <= 1:
        cleaned = text.strip()
        if not cleaned:
            return []
        if len(cleaned) >= _AUTO_SPLIT_MIN_CHARS:
            return _sentence_bubbles(cleaned)
        return [cleaned]
    if len(parts) > _MAX_BUBBLES:
        # Overflow folds into the last bubble rather than being dropped.
        parts = parts[:_MAX_BUBBLES - 1] + [" ".join(parts[_MAX_BUBBLES - 1:])]
    return parts


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
        holder["reply"], _ = await _guarded_reply(messages, holder["tool_ran"])

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


def _hhmm(s: str, default: str = "23:45") -> tuple[int, int]:
    try:
        h, m = (s or default).split(":")
        return int(h), int(m)
    except ValueError:
        h, m = default.split(":")
        return int(h), int(m)


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
        text, _ = await _guarded_reply(messages, tool_ran=True)
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
        text, _ = await _guarded_reply(messages, tool_ran)
    except Exception as e:  # noqa: BLE001
        logger.warning("busy-return delivery failed: %s", e)
        text = "hey sorry, i got pulled away earlier! what were we talking about? 😅"
    await _deliver_message(t["session_id"], text)
    state.db.update_deferred(t["id"], done=True)
    logger.info("busy-return #%d delivered", t["id"])


# ---------------------------------------------------------------------------
# Persona endpoints
# ---------------------------------------------------------------------------
@app.get("/persona")
async def get_persona():
    return _persona_public(state.persona)


@app.get("/personas")
async def list_personas():
    ids = list_persona_ids(state.settings.persona_folder)
    personas = []
    for pid in ids:
        try:
            p = load_persona(state.settings.persona_folder, pid)
            personas.append(_persona_public(p))
        except Exception as e:  # noqa: BLE001 - skip malformed persona files
            logger.warning("Skipping persona '%s': %s", pid, e)
    return {"active": state.persona.id, "personas": personas}


@app.post("/personas/active")
async def set_active_persona(body: ActivePersona):
    if body.id not in list_persona_ids(state.settings.persona_folder):
        raise HTTPException(status_code=404, detail="Persona not found")
    state.persona = load_persona(state.settings.persona_folder, body.id)
    state.db.set_setting("active_persona", body.id)
    if state.tool_ctx:
        state.tool_ctx.persona = state.persona  # keep the draw tool's selfie appearance current
    if state.memory:
        state.memory.blocked_names = {state.persona.name.lower()}
    logger.info("Switched active persona to '%s'", body.id)
    return _persona_public(state.persona)


@app.get("/personas/{persona_id}/photo")
async def get_persona_photo(persona_id: str):
    path = _resolve_photo_path(persona_id)
    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-cache"})


@app.post("/persona/photo")
async def upload_persona_photo(file: UploadFile = File(...)):
    """Set the active persona's profile photo (shared with the WhatsApp account)."""
    persona_id = state.persona.id
    ext = Path(file.filename or "").suffix.lower() or ".png"
    # .svg deliberately excluded: it can carry <script>, and this file gets
    # served same-origin - an uploaded SVG could read the auth token straight
    # out of localStorage. The shipped default avatar (media/avatars/luna.svg)
    # is a static asset, not user-uploaded, so it's unaffected.
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {ext}")

    AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    dest = AVATAR_DIR / f"{persona_id}_upload{ext}"
    data = await file.read()
    dest.write_bytes(data)

    # Record the override; YAML profile_pic stays as the default fallback.
    state.db.set_setting(f"profile_pic:{persona_id}", str(dest))
    logger.info("Updated profile photo for '%s' -> %s", persona_id, dest.name)
    return _persona_public(state.persona)


# ---------------------------------------------------------------------------
# Memory endpoints
# ---------------------------------------------------------------------------
@app.get("/memory-graph")
async def memory_graph_page():
    template = ROOT / "app" / "templates" / "memory_graph.html"
    if not template.exists():
        raise HTTPException(status_code=404, detail="Memory graph template not found")
    return FileResponse(template)


@app.get("/memory-graph/data")
async def memory_graph_data():
    entities = state.db.list_entities()
    relations = state.db.list_relations(active_only=True)
    memories = state.db.list_memories()

    linked_memories: dict[int, list[dict]] = {int(e["id"]): [] for e in entities}
    for memory in memories:
        fact = memory.get("fact", "") or ""
        for entity in entities:
            entity_name = entity.get("name", "") or ""
            if entity_name.lower() in fact.lower():
                linked_memories[int(entity["id"])].append({
                    "id": memory["id"],
                    "fact": memory["fact"],
                    "category": memory["category"],
                })

    return {
        "entities": [
            {
                "id": int(entity["id"]),
                "label": entity["name"],
                "title": entity["name"],
                "type": entity["type"],
                "notes": entity.get("notes") or "",
                "linked_memories": linked_memories[int(entity["id"])],
            }
            for entity in entities
        ],
        "relations": [
            {
                "id": int(relation["id"]),
                "from": int(relation["source_id"]),
                "to": int(relation["target_id"]),
                "label": relation["relation"],
                "confidence": relation.get("confidence", 0.5),
            }
            for relation in relations
        ],
    }


@app.put("/memory-graph/entities/{entity_id}")
async def update_memory_graph_entity(entity_id: int, body: dict):
    name = (body.get("name") or "").strip()
    entity_type = (body.get("type") or "thing").strip()
    notes = (body.get("notes") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    updated = state.db.update_entity(entity_id, name=name, entity_type=entity_type, notes=notes)
    if not updated:
        raise HTTPException(status_code=404, detail="entity not found")
    return state.db.get_entity(entity_id)


@app.delete("/memory-graph/entities/{entity_id}")
async def delete_memory_graph_entity(entity_id: int):
    deleted = state.db.delete_entity(entity_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="entity not found")
    return {"deleted": entity_id}


@app.delete("/memory-graph/relations/{relation_id}")
async def delete_memory_graph_relation(relation_id: int):
    deleted = state.db.delete_relation(relation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="relation not found")
    return {"deleted": relation_id}


@app.get("/memories")
async def list_memories():
    return {"memories": state.db.list_memories()}


@app.post("/memories", status_code=201)
async def create_memory(mem: MemoryCreate):
    category = mem.category if mem.category in VALID_CATEGORIES else "personal_info"
    memory_id = state.db.add_memory(mem.fact.strip(), category)
    state.memory.sync_from_row(memory_id)
    return state.db.get_memory(memory_id)


@app.post("/memories/{memory_id}/complete")
async def complete_memory(memory_id: int):
    """Mark an event/plan memory completed: it stops being injected into
    prompts (used by the ✓ button in Settings and future follow-up tools)."""
    if not state.db.get_memory(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    state.db.complete_memory(memory_id)
    state.db.mark_event_resolved_by_memory(memory_id)
    return state.db.get_memory(memory_id)


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: int):
    if not state.db.delete_memory(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    state.memory.remove(memory_id)
    return {"deleted": memory_id}


# ---------------------------------------------------------------------------
# Mood journal (passive, local-model-only - see app/journal.py)
# ---------------------------------------------------------------------------
class MoodEntryEdit(BaseModel):
    mood_label: str | None = None
    intensity: int | None = Field(default=None, ge=1, le=5)
    why: str | None = None


@app.get("/journal")
async def list_journal(since: str | None = None, mood: str | None = None):
    return {"entries": state.db.list_mood_entries(since_date=since, mood_filter=mood)}


@app.put("/journal/{entry_id}")
async def edit_journal_entry(entry_id: int, body: MoodEntryEdit):
    """User corrections are final and feed back as a memory - a corrected
    mood is a stronger, more durable signal than an inferred one."""
    if not state.db.get_mood_entry(entry_id):
        raise HTTPException(status_code=404, detail="Entry not found")
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    updated = state.db.update_mood_entry(entry_id, **fields)
    if fields:
        await state.memory.add_fact(
            f"On {updated['date']}, the user corrected their mood journal - it "
            f"was actually {updated['mood_label']} ({updated['why']}).",
            "emotion", source="mood_journal_edit",
        )
    return updated


@app.delete("/journal/{entry_id}")
async def delete_journal_entry(entry_id: int):
    if not state.db.delete_mood_entry(entry_id):
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"deleted": entry_id}


@app.post("/journal/run-now")
async def journal_run_now(day_offset: int = 0):
    """DEV: manually trigger nightly extraction (default: today so far, not
    yesterday - for testing without waiting for the scheduled time)."""
    count = await run_nightly_extraction(state.db, state.llm, state.settings, day_offset=day_offset)
    return {"stored": count}


@app.post("/journal/run-weekly-now")
async def journal_run_weekly_now():
    """DEV: manually trigger the weekly pattern-awareness pass."""
    count = await run_weekly_patterns(state.db, state.memory, state.settings)
    return {"stored": count}


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
@app.get("/history/{session_id}")
async def get_history(session_id: str):
    return {"session_id": session_id, "messages": state.db.get_all_messages(session_id)}


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    removed = state.db.clear_session(session_id)
    return {"session_id": session_id, "cleared": removed}


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
        user_text = await asyncio.to_thread(state.stt.transcribe, audio)
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
                                        repair_note, _streak_note(user_text)),
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
    system_prompt = build_system_prompt(
        profile.persona,
        memories,
        current_time=_now_context(),
        extra_notes=_relationship_notes(CAPABILITY_MANIFEST, day_note,
                                        EMOTION_TAG_INSTRUCTION,
                                        _offer_note(combined), _pattern_note(combined),
                                        _streak_note(combined),
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
            _guarded_reply(messages, tool_ran), timeout=90.0)
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
# Proactive messaging controls
# ---------------------------------------------------------------------------
class PauseBody(BaseModel):
    hours: float = Field(..., ge=0, le=168)


@app.post("/proactive/pause")
async def proactive_pause(body: PauseBody):
    if not state.proactive:
        raise HTTPException(status_code=503, detail="Proactive scheduler not running")
    until = state.proactive.pause_for(body.hours)
    return {"paused_until": until}


@app.get("/proactive/status")
async def proactive_status():
    if not state.proactive:
        return {"enabled": False, "running": False}
    return {"running": True, **state.proactive.status()}


# ---------------------------------------------------------------------------
# Brain status + her day (dev)
# ---------------------------------------------------------------------------
@app.get("/brain/status")
async def brain_status():
    return {
        **(state.brain.status() if state.brain else {}),
        "routing": state.router.stats() if state.router else {},
        "guards": guard_stats(state.db),
        "deferred": state.db.list_deferred(10),
        "reminders_pending": state.db.list_reminders(pending_only=True),
    }


class BenchRequest(BaseModel):
    emotions: list[str] = Field(default=["neutral", "happy", "sad"])


@app.post("/voice/bench")
async def voice_bench(body: BenchRequest):
    """Consistency bench: same 3 sentences via studio clone AND kokoro(+rvc),
    per emotion, so you can tune until it's one person everywhere."""
    sentences = [
        "hey, i was just thinking about you.",
        "no way, tell me everything right now!",
        "okay fine, you win this one... this time.",
    ]
    bench_dir = MEDIA_DIR / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    out: dict = {"studio_available": bool(state.studio and state.studio.available),
                 "rvc_available": bool(state.rvc and state.rvc.available),
                 "renders": []}
    vcfg = (state.settings.raw or {}).get("voice", {})
    for emo in body.emotions:
        for i, line in enumerate(sentences):
            entry = {"emotion": emo, "line": i}
            # Kokoro (+ optional RVC) - the call voice.
            res = await asyncio.to_thread(state.tts.synth, line)
            samples, sr = res.samples, res.sample_rate
            if vcfg.get("call_voice") == "kokoro_rvc" and state.rvc.available:
                samples, _ = await asyncio.to_thread(state.rvc.convert, samples, sr)
            p = bench_dir / f"call_{emo}_{i}.wav"
            sf.write(p, samples, sr)
            entry["call_url"] = f"/media/bench/{p.name}"
            # Studio clone - the voice-note voice.
            if out["studio_available"]:
                try:
                    from app.gpu_queue import PRIORITY_BENCH
                    fut = await state.gpu_queue.submit(
                        f"bench_{emo}_{i}",
                        lambda l=line, e=emo: asyncio.to_thread(
                            state.studio.render, l, e),
                        priority=PRIORITY_BENCH)
                    s2, sr2 = await fut
                    p2 = bench_dir / f"studio_{emo}_{i}.wav"
                    sf.write(p2, s2, sr2)
                    entry["studio_url"] = f"/media/bench/{p2.name}"
                except Exception as e:  # noqa: BLE001
                    entry["studio_error"] = str(e)
            out["renders"].append(entry)
    return out


def _training_progress() -> dict | None:
    """Tail rvc_training.log (UTF-16 from Tee-Object) for stage/epoch lines."""
    log = ROOT / "rvc_training.log"
    if not log.exists():
        return None
    try:
        text = log.read_text(encoding="utf-16", errors="ignore")
    except (UnicodeError, OSError):
        try:
            text = log.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    stages = [l for l in lines if l.startswith("STAGE") or "COMPLETE" in l
              or "EXPORTED" in l]
    epochs = [l for l in lines if re.search(r"epoch[ =:]+\d+", l, re.I)]
    return {
        "last_stage": stages[-1] if stages else None,
        "last_epoch_line": epochs[-1][:120] if epochs else None,
        "done": any("TRAINING COMPLETE" in l for l in lines),
        "failed": any("FAILED" in l for l in stages[-1:]) if stages else False,
    }


@app.get("/voice/status")
async def voice_status():
    vcfg = (state.settings.raw or {}).get("voice", {})
    return {
        "training": _training_progress(),
        "call_voice": vcfg.get("call_voice", "kokoro_raw"),
        "rvc_ready": bool(state.rvc and state.rvc.available),
        "rvc_status": state.rvc.status_label() if state.rvc else "not trained",
        "rvc_last_latency_ms": state.rvc.last_latency_ms if state.rvc else None,
        "studio_engine": state.studio.engine_name if state.studio else None,
        "studio_installed": bool(state.studio and state.studio.available),
        "gpu_queue": state.gpu_queue.status() if state.gpu_queue else {},
        "song_library": [s.get("title") for s in state.covers.library()]
        if state.covers else [],
        "inbox_pending": [p.name for p in state.covers.pending_inbox()]
        if state.covers else [],
    }


@app.get("/day-state")
async def day_state():
    return await state.daylife.today()


@app.post("/day-state/regenerate")
async def day_state_regenerate():
    """DEV: reroll today's hidden day state."""
    return await state.daylife.regenerate()


# ---------------------------------------------------------------------------
# Relationship progression
# ---------------------------------------------------------------------------
class RelationshipOverride(BaseModel):
    stage: str | None = None
    affection: float | None = Field(default=None, ge=0, le=100)


@app.get("/relationship")
async def get_relationship():
    return state.relationship.state()


@app.post("/relationship/override")
async def relationship_override(body: RelationshipOverride):
    """DEV ONLY: force stage/affection for testing (exposed in Settings)."""
    if body.stage is not None and body.stage not in STAGES:
        raise HTTPException(status_code=400, detail=f"stage must be one of {STAGES}")
    return state.relationship.override(stage=body.stage, affection=body.affection)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{state.settings.ollama_base_url}/api/tags")
            ollama_ok = r.status_code == 200
    except httpx.HTTPError:
        ollama_ok = False
    return {"status": "ok", "ollama_reachable": ollama_ok}


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
