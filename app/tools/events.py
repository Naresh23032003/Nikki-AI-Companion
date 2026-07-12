"""Upcoming-events tool: surfaces dated events from temporal memory."""
from __future__ import annotations

from datetime import datetime, timezone

from app.tools import Tool, ToolContext


async def execute(args: dict, ctx: ToolContext) -> dict:
    rows = [m for m in ctx.db.list_memories() if m.get("kind") == "event"]
    upcoming = []
    now = datetime.now(timezone.utc)
    for m in rows:
        try:
            dt = datetime.fromisoformat(m["event_datetime"]) if m.get("event_datetime") else None
        except (ValueError, TypeError):
            dt = None
        if dt and dt.replace(tzinfo=dt.tzinfo or timezone.utc) >= now:
            upcoming.append(f"{m['fact']} ({dt.astimezone():%a %I:%M %p})")
    if not upcoming:
        return {"ok": True, "result": "nothing upcoming on their schedule"}
    return {"ok": True, "result": "upcoming: " + "; ".join(upcoming[:5])}


TOOL = Tool(
    name="events",
    description="List the user's upcoming dated events/plans (exams, flights, appointments, etc).",
    parameters={},
    required=[],
    execute=execute,
)
