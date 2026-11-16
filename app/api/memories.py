"""Memory CRUD endpoints.

Extracted from app/main.py (N2 phase 2). Bodies are unchanged; only the
decorator (`@app` -> `@router`) and the imports differ.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.deps import state
from app.memory import VALID_CATEGORIES

router = APIRouter(tags=["memories"])

class MemoryCreate(BaseModel):
    fact: str = Field(..., min_length=1)
    category: str = Field(default="personal_info")


@router.get("/memories")
async def list_memories():
    return {"memories": state.db.list_memories()}


@router.post("/memories", status_code=201)
async def create_memory(mem: MemoryCreate):
    category = mem.category if mem.category in VALID_CATEGORIES else "personal_info"
    memory_id = state.db.add_memory(mem.fact.strip(), category)
    state.memory.sync_from_row(memory_id)
    return state.db.get_memory(memory_id)


@router.post("/memories/{memory_id}/complete")
async def complete_memory(memory_id: int):
    """Mark an event/plan memory completed: it stops being injected into
    prompts (used by the ✓ button in Settings and future follow-up tools)."""
    if not state.db.get_memory(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    state.db.complete_memory(memory_id)
    state.db.mark_event_resolved_by_memory(memory_id)
    return state.db.get_memory(memory_id)


@router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: int):
    if not state.db.delete_memory(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    state.memory.remove(memory_id)
    return {"deleted": memory_id}
