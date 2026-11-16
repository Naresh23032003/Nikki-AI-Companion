"""Persona identity and photo endpoints.

Extracted from app/main.py (N2 phase 2, round 2). Bodies unchanged; the two
private helpers (`_persona_public`, `_resolve_photo_path`) moved with their
only callers since nothing else in the app used them.
"""
from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import ROOT
from app.deps import state
from app.persona import list_persona_ids, load_persona

logger = logging.getLogger("companion")

router = APIRouter(tags=["personas"])

MEDIA_DIR = ROOT / "media"
AVATAR_DIR = MEDIA_DIR / "avatars"
DEFAULT_AVATAR = AVATAR_DIR / "luna.svg"


class ActivePersona(BaseModel):
    id: str = Field(..., min_length=1)


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


@router.get("/persona")
async def get_persona():
    return _persona_public(state.persona)


@router.get("/personas")
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


@router.post("/personas/active")
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


@router.get("/personas/{persona_id}/photo")
async def get_persona_photo(persona_id: str):
    path = _resolve_photo_path(persona_id)
    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-cache"})


@router.post("/persona/photo")
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
