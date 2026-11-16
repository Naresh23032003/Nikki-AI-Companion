"""Proactive scheduler control endpoints (pause, status, manual trigger).

Extracted from app/main.py (N2 phase 2, round 2). Bodies unchanged.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.deps import state

router = APIRouter(tags=["proactive"])


class PauseBody(BaseModel):
    hours: float = Field(..., ge=0, le=168)


@router.post("/proactive/pause")
async def proactive_pause(body: PauseBody):
    if not state.proactive:
        raise HTTPException(status_code=503, detail="Proactive scheduler not running")
    until = state.proactive.pause_for(body.hours)
    return {"paused_until": until}


@router.get("/proactive/status")
async def proactive_status():
    if not state.proactive:
        return {"enabled": False, "running": False}
    return {"running": True, **state.proactive.status()}


@router.post("/proactive/trigger/{profile_id}")
async def proactive_trigger(profile_id: str, intent: str = "random_thought"):
    """Fire one proactive check-in RIGHT NOW for a given profile (main/friend/...),
    bypassing the random daily schedule - for testing that she actually texts
    first and that it reaches the right person. Still honors the normal skip
    conditions (paused, outside active hours, mid-conversation already)."""
    profile = state.profiles.by_id(profile_id) if state.profiles else None
    if not profile or not profile.proactive:
        raise HTTPException(status_code=404, detail=f"no running profile '{profile_id}'")
    fired = await profile.proactive.fire_checkin(intent=intent)
    return {"profile": profile.id, "persona": profile.persona.name,
            "fired": fired, "intent": intent}
