"""Journal endpoints.

Extracted from app/main.py (N2 phase 2). Bodies unchanged.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.deps import state
from app.journal import run_nightly_extraction, run_weekly_patterns

router = APIRouter(tags=["journal"])

class MoodEntryEdit(BaseModel):
    mood_label: str | None = None
    intensity: int | None = Field(default=None, ge=1, le=5)
    why: str | None = None


@router.get("/journal")
async def list_journal(since: str | None = None, mood: str | None = None):
    return {"entries": state.db.list_mood_entries(since_date=since, mood_filter=mood)}


@router.put("/journal/{entry_id}")
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


@router.delete("/journal/{entry_id}")
async def delete_journal_entry(entry_id: int):
    if not state.db.delete_mood_entry(entry_id):
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"deleted": entry_id}


@router.post("/journal/run-now")
async def journal_run_now(day_offset: int = 0):
    """DEV: manually trigger nightly extraction (default: today so far, not
    yesterday - for testing without waiting for the scheduled time)."""
    count = await run_nightly_extraction(state.db, state.llm, state.settings, day_offset=day_offset)
    return {"stored": count}


@router.post("/journal/run-weekly-now")
async def journal_run_weekly_now():
    """DEV: manually trigger the weekly pattern-awareness pass."""
    count = await run_weekly_patterns(state.db, state.memory, state.settings)
    return {"stored": count}
