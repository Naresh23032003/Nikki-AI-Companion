"""Data portability + erasure endpoints (N10: privacy hardening).

Scoped to the CURRENT profile (P()) - never state.db directly - so a
multi-persona setup can never export or erase the wrong person's data.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.deps import P

router = APIRouter(tags=["privacy"])


@router.get("/privacy/export")
async def export_data():
    """Everything stored about the active profile: messages, memories,
    entities/relations, mood journal, reminders, event follow-ups and
    relationship state. A JSON download, not sent anywhere - GDPR-style data
    portability for a local-first app that holds real relationship history."""
    profile = P()
    if not profile or not profile.db:
        raise HTTPException(status_code=503, detail="profile not ready")
    return profile.db.export_all_data()


class DeleteAllRequest(BaseModel):
    confirm: bool = False


@router.post("/privacy/delete-all")
async def delete_all_data(body: DeleteAllRequest):
    """Irreversibly erase EVERYTHING stored about the active profile:
    messages, memories (SQLite rows AND their Chroma vectors), entities,
    relations, mood journal, reminders, event follow-ups, and resets
    relationship state to a fresh start. Requires an explicit
    {"confirm": true} body - this cannot be triggered by an empty POST."""
    if not body.confirm:
        raise HTTPException(
            status_code=400,
            detail="set confirm=true to erase all data for this profile - "
                   "this cannot be undone")
    profile = P()
    if not profile or not profile.db:
        raise HTTPException(status_code=503, detail="profile not ready")
    vectors_removed = 0
    if profile.memory:
        vectors_removed = profile.memory.wipe_all()
    profile.db.delete_all_data()
    return {"deleted": True, "vectors_removed": vectors_removed}
