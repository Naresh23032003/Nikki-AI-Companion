"""Shared application state and per-request profile resolution.

Extracted from app/main.py (N2 phase 2) so route modules can reach the running
services without importing `main` — which would be a circular import, since
`main` includes those routers.

Nothing heavy is imported at runtime: the service types are annotations only,
resolved under TYPE_CHECKING. Importing this module must stay cheap, because
every route module does.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from app.covers import CoverPipeline
    from app.dayseed import DayLife
    from app.db import Database
    from app.gpu_queue import GPUJobQueue
    from app.llm import OllamaClient
    from app.memory import MemoryStore
    from app.profiles import Profile, ProfileRegistry
    from app.providers import CloudBrain
    from app.relationship import RelationshipTracker
    from app.router import Router


class AppState:
    """Services wired up during lifespan startup.

    Populated by app.main's lifespan handler; read everywhere. Kept as a plain
    class with class-level defaults, exactly as it was in main.py, so the
    startup assignment order and `None`-until-ready semantics are unchanged.
    """

    settings = None
    persona = None
    db: Database | None = None
    llm: OllamaClient | None = None
    memory: MemoryStore | None = None
    stt: Any = None            # STTEngine — optional voice tier
    tts: Any = None            # TTSEngine — optional voice tier
    proactive = None           # ProactiveEngine
    relationship: RelationshipTracker | None = None
    brain: CloudBrain | None = None
    router: Router | None = None
    tools: Any = None          # ToolRunner
    tool_ctx: Any = None       # ToolContext
    daylife: DayLife | None = None
    rvc: Any = None            # RVCConverter — optional voice tier
    studio: Any = None         # StudioTTS — optional voice tier
    covers: CoverPipeline | None = None
    gpu_queue: GPUJobQueue | None = None
    profiles: ProfileRegistry | None = None


state = AppState()

# The profile (persona + its isolated db/memory/relationship) this request
# belongs to. WhatsApp sets it per incoming message from the sender's number;
# everything else (web app, calls) leaves it unset and gets the default
# profile, which IS state.db/state.persona - i.e. unchanged behavior.
_current_profile: ContextVar[Profile | None] = ContextVar(
    "current_profile", default=None)


def P() -> Profile:
    """The profile serving this request. Never None once startup has run."""
    p = _current_profile.get()
    if p is not None:
        return p
    return state.profiles.default if state.profiles else None


def persona_voice() -> str | None:
    """The active persona's voice, or None to fall back to the default."""
    p = P()
    persona = p.persona if p else state.persona
    return (persona.voice or None) if persona else None


def db_for_session(session_id: str):
    """Resolve which database a session_id actually lives in - each profile
    has its OWN db file (see app/profiles.py), so these must not silently
    fall through to the default profile's db when the session belongs to
    someone else's profile."""
    profile = state.profiles.by_session(session_id) if state.profiles else None
    return profile.db if profile else state.db
