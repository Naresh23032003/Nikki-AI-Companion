"""Day-state, brain diagnostics and health-check endpoints.

Extracted from app/main.py (N2 phase 2, round 2). Bodies unchanged.
"""
from __future__ import annotations

import httpx
from fastapi import APIRouter

from app.deps import state
from app.guards import guard_stats

router = APIRouter(tags=["status"])


@router.get("/day-state")
async def day_state():
    return await state.daylife.today()


@router.post("/day-state/regenerate")
async def day_state_regenerate():
    """DEV: reroll today's hidden day state."""
    return await state.daylife.regenerate()


@router.get("/brain/status")
async def brain_status():
    return {
        **(state.brain.status() if state.brain else {}),
        "routing": state.router.stats() if state.router else {},
        "guards": guard_stats(state.db),
        "deferred": state.db.list_deferred(10),
        "reminders_pending": state.db.list_reminders(pending_only=True),
    }


@router.get("/health")
async def health():
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{state.settings.ollama_base_url}/api/tags")
            ollama_ok = r.status_code == 200
    except httpx.HTTPError:
        ollama_ok = False
    return {"status": "ok", "ollama_reachable": ollama_ok}
