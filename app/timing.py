"""Time and formatting helpers extracted from app/main.py.

Pure functions: `quiet_hours` and `hhmm` previously read module globals or were
buried in the monolith, so the midnight-crossing window — the one case most
likely to be wrong — had no test.
"""
from __future__ import annotations

from datetime import datetime
from datetime import time as dtime


def sse(data: str, event: str | None = None) -> str:
    """Format a server-sent-events frame."""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"


def now_context(now: datetime | None = None) -> str:
    """Human-readable local date+time for the system prompt, e.g.
    'Saturday, July 11, 12:14 PM (afternoon)'. Weekday, date and time all
    come from this ONE line - when the date lived in a different note the
    model couldn't bind them and would get the weekday wrong."""
    now = now or datetime.now()
    h = now.hour
    tod = (
        "the middle of the night" if h < 4 else
        "early morning" if h < 7 else
        "morning" if h < 12 else
        "afternoon" if h < 17 else
        "evening" if h < 21 else
        "night"
    )
    return (f"{now.strftime('%A')}, {now.strftime('%B')} {now.day}, "
            f"{now.strftime('%I:%M %p').lstrip('0')} ({tod})")


def in_quiet_hours(raw: str | None, now: datetime | None = None) -> bool:
    """Whether `now` falls inside a '01:00-07:30' style window.

    Used to hold self-initiated deliveries (reminders, deferred answers, event
    follow-ups) until it's over instead of firing at 3am; the item stays queued
    and fires on the next scheduler tick after the window ends.

    A malformed or empty window means "never quiet" — deliveries continue
    rather than being silently suppressed forever by a typo in config.
    """
    raw = (raw or "").strip()
    if not raw:
        return False
    try:
        s, e = raw.split("-", 1)
        sh, sm = (int(x) for x in s.strip().split(":"))
        eh, em = (int(x) for x in e.strip().split(":"))
        start, end = dtime(sh, sm), dtime(eh, em)
    except (ValueError, AttributeError):
        return False

    t = (now or datetime.now()).time()
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end  # window crossing midnight


def hhmm(s: str | None, default: str = "23:45") -> tuple[int, int]:
    """Parse 'HH:MM' into (hour, minute), falling back to `default`."""
    try:
        h, m = (s or default).split(":")
        return int(h), int(m)
    except ValueError:
        h, m = default.split(":")
        return int(h), int(m)
