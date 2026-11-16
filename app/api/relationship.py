"""Relationship progression endpoints.

Extracted from app/main.py (N2 phase 2, round 2). Bodies unchanged.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.deps import state
from app.relationship import STAGES

router = APIRouter(tags=["relationship"])


class RelationshipOverride(BaseModel):
    stage: str | None = None
    affection: float | None = Field(default=None, ge=0, le=100)


@router.get("/relationship")
async def get_relationship():
    return state.relationship.state()


@router.post("/relationship/override")
async def relationship_override(body: RelationshipOverride):
    """DEV ONLY: force stage/affection for testing (exposed in Settings)."""
    if body.stage is not None and body.stage not in STAGES:
        raise HTTPException(status_code=400, detail=f"stage must be one of {STAGES}")
    return state.relationship.override(stage=body.stage, affection=body.affection)
