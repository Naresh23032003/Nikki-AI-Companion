"""Reminder tool: "remind me to call mom at 6" -> confirm in-character ->
APScheduler-polled delivery (main.py's _deliver_due_reminders) -> delivered
via the delivery-routing hierarchy (device -> WhatsApp -> web mirror), always
in-character. Recurring reminders re-arm instead of completing. Persisted in
SQLite, survives restarts. List/cancel are also actions on this same tool."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from app.tools import Tool, ToolContext

_RECURRENCE_WORDS = re.compile(
    r"\bevery\s+day\b|\bdaily\b|\bevery\s+(mon|tues?|wed(nes)?|thur?s?|fri|sat(ur)?|sun)\w*\b|\bweekly\b",
    re.I)


def parse_when(text: str, now: datetime | None = None) -> datetime | None:
    """Parse a natural time reference to an aware local datetime."""
    now = now or datetime.now().astimezone()
    t = text.lower()

    m = re.search(r"\bin\s+(\d+)\s*(min(ute)?s?|hours?|hrs?)\b", t)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = timedelta(minutes=n) if unit.startswith(("min",)) else timedelta(hours=n)
        return now + delta

    m = re.search(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", t)
    hour = minute = None
    if m and ("at" in t or m.group(3)):
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        mer = m.group(3)
        if mer == "pm" and hour < 12:
            hour += 12
        elif mer == "am" and hour == 12:
            hour = 0
        if hour > 23:
            hour = None

    target = now
    if "tomorrow" in t:
        target = now + timedelta(days=1)
        if hour is None:
            hour, minute = 9, 0
    if hour is not None:
        candidate = target.replace(hour=hour, minute=minute or 0, second=0, microsecond=0)
        if candidate <= now and "tomorrow" not in t:
            candidate += timedelta(days=1)
        return candidate
    if "tonight" in t:
        return now.replace(hour=21, minute=0, second=0, microsecond=0)
    return None


def parse_recurrence(text: str) -> str | None:
    m = _RECURRENCE_WORDS.search(text)
    if not m:
        return None
    hit = m.group(0).lower()
    if "daily" in hit or "every day" in hit:
        return "daily"
    if "weekly" in hit:
        return "weekly"
    return f"every {hit.split()[-1]}"  # "every monday" etc.


async def execute(args: dict, ctx: ToolContext) -> dict:
    action = (args.get("action") or "set").lower()
    text = (args.get("text") or "").strip()
    when_text = (args.get("when") or "").strip()

    if action == "list":
        pending = ctx.db.list_reminders(pending_only=True)
        if not pending:
            return {"ok": True, "result": "no pending reminders"}
        items = "; ".join(f"'{r['text']}' at {r['due_at'][11:16]}" for r in pending[:8])
        return {"ok": True, "result": f"pending reminders: {items}"}

    if action == "cancel":
        target = text or when_text
        if not target:
            return {"ok": False, "result": "no reminder description given to cancel"}
        cancelled = ctx.db.cancel_reminder_like(target)
        if not cancelled:
            return {"ok": False, "result": f"no pending reminder matching '{target}'"}
        return {"ok": True, "result": f"cancelled reminder: '{cancelled['text']}'"}

    # action == "set" (default)
    due = parse_when(when_text) or parse_when(text)
    if due is None:
        return {"ok": False, "result": "could not understand the time - ask them when exactly"}
    if not text:
        text = "the thing they asked about"
    recurrence = parse_recurrence(when_text) or parse_recurrence(text)
    session_id = getattr(ctx.settings, "wa_session_id", "main")
    rid = ctx.db.add_reminder(text, due.astimezone(timezone.utc).isoformat(),
                              recurrence_rule=recurrence, session_id=session_id)
    local = due.strftime("%I:%M %p on %b %d").lstrip("0")
    suffix = f" (repeats {recurrence})" if recurrence else ""
    return {"ok": True, "result": f"reminder #{rid} saved: '{text}' at {local}{suffix}"}


TOOL = Tool(
    name="reminder",
    description=("Set, list, or cancel a reminder for the user. Use action='set' when "
                "they ask to be reminded of something at a time; action='list' when "
                "they ask what reminders are pending; action='cancel' when they want "
                "one removed."),
    parameters={
        "action": {"type": "string", "enum": ["set", "list", "cancel"],
                  "description": "set | list | cancel"},
        "text": {"type": "string", "description": "what to remind them about (or, for cancel, which reminder to remove)"},
        "when": {"type": "string", "description": "when it's due, e.g. '6pm', 'tomorrow at 9am', 'in 20 minutes', 'every monday'"},
    },
    required=["action"],
    execute=execute,
)
