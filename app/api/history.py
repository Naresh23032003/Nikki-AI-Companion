"""Conversation history endpoints.

Extracted from app/main.py (N2 phase 2). Bodies unchanged.

Note the use of db_for_session: each profile has its OWN database file, so
these must not fall through to the default profile's db when the session
belongs to another persona.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.deps import db_for_session as _db_for_session

router = APIRouter(tags=["history"])

@router.get("/history/{session_id}")
async def get_history(session_id: str):
    db = _db_for_session(session_id)
    return {"session_id": session_id, "messages": db.get_all_messages(session_id)}


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    db = _db_for_session(session_id)
    removed = db.clear_session(session_id)
    return {"session_id": session_id, "cleared": removed}
