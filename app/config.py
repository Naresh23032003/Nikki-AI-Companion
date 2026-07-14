"""Loads config.yaml into a simple, typed-ish settings object."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

# Project root = the directory containing this file's parent (app/ -> project).
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Tiny .env loader (KEY=value lines) - no extra dependency."""
    import os

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass
class Settings:
    ollama_base_url: str
    ollama_model: str
    ollama_embed_model: str
    ollama_extract_model: str
    ollama_options: Dict[str, Any]
    ollama_keep_alive: str
    persona_active: str
    persona_folder: Path
    db_path: Path
    max_messages: int
    # Voice
    stt_model_size: str
    tts_default_voice: str
    tts_lang_code: str
    tts_speed: float
    # Long-term memory
    chroma_path: Path
    memory_collection: str
    memory_top_k: int
    memory_retrieval_threshold: float
    memory_recent_count: int
    memory_dedup_threshold: float
    # WhatsApp bridge
    wa_bridge_url: str
    wa_voice_ratio: float
    wa_session_id: str
    # Two-brain / router / behavior (kept as dicts - consumed by their modules)
    brain: Dict[str, Any] = field(default_factory=dict)
    router: Dict[str, Any] = field(default_factory=dict)
    behavior: Dict[str, Any] = field(default_factory=dict)
    journal: Dict[str, Any] = field(default_factory=dict)
    host: str = "0.0.0.0"
    port: int = 8000
    raw: Dict[str, Any] = field(default_factory=dict)


def load_settings(config_path: Path = CONFIG_PATH) -> Settings:
    import os

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    ollama = data.get("ollama", {})
    # Env overrides win over config.yaml so the same image runs anywhere.
    # In Docker the companion reaches Ollama at http://ollama:11434, not
    # localhost (see docker-compose.yml).
    if os.environ.get("OLLAMA_BASE_URL"):
        ollama["base_url"] = os.environ["OLLAMA_BASE_URL"]
    if os.environ.get("OLLAMA_MODEL"):
        ollama["model"] = os.environ["OLLAMA_MODEL"]
    if os.environ.get("OLLAMA_EMBED_MODEL"):
        ollama["embed_model"] = os.environ["OLLAMA_EMBED_MODEL"]
    # Let the DB + vector store live on a mounted volume in Docker.
    db_override = os.environ.get("COMPANION_DB_PATH")
    chroma_override = os.environ.get("CHROMA_PATH")
    persona = data.get("persona", {})
    database = data.get("database", {})
    context = data.get("context", {})
    stt = data.get("stt", {})
    tts = data.get("tts", {})
    memory = data.get("memory", {})
    whatsapp = data.get("whatsapp", {})
    server = data.get("server", {})

    return Settings(
        ollama_base_url=ollama.get("base_url", "http://localhost:11434").rstrip("/"),
        ollama_model=ollama.get("model", "llama3.2:3b"),
        ollama_embed_model=ollama.get("embed_model", "nomic-embed-text"),
        ollama_extract_model=ollama.get("extract_model")
        or ollama.get("model", "llama3.2:3b"),
        ollama_options=ollama.get("options", {}) or {},
        ollama_keep_alive=str(ollama.get("keep_alive", "2h")),
        persona_active=persona.get("active", "luna"),
        persona_folder=ROOT / persona.get("folder", "personas"),
        db_path=Path(db_override) if db_override else ROOT / database.get("path", "companion.db"),
        max_messages=int(context.get("max_messages", 20)),
        stt_model_size=stt.get("model_size", "small"),
        tts_default_voice=tts.get("default_voice", "af_heart"),
        tts_lang_code=tts.get("lang_code", "a"),
        tts_speed=float(tts.get("speed", 0.92)),
        chroma_path=Path(chroma_override) if chroma_override else ROOT / memory.get("chroma_path", "chroma_db"),
        memory_collection=memory.get("collection", "companion_memories"),
        memory_top_k=int(memory.get("top_k", 6)),
        memory_retrieval_threshold=float(memory.get("retrieval_threshold", 0.35)),
        memory_recent_count=int(memory.get("recent_count", 3)),
        memory_dedup_threshold=float(memory.get("dedup_threshold", 0.92)),
        wa_bridge_url=whatsapp.get("bridge_url", "http://localhost:3001").rstrip("/"),
        wa_voice_ratio=float(whatsapp.get("voice_reply_ratio", 0.3)),
        wa_session_id=whatsapp.get("session_id", "main"),
        brain=data.get("brain", {}) or {},
        router=data.get("router", {}) or {},
        behavior=data.get("behavior", {}) or {},
        journal=data.get("journal", {}) or {},
        host=server.get("host", "0.0.0.0"),
        port=int(server.get("port", 8000)),
        raw=data,
    )
