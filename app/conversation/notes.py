"""Prompt notes — the per-turn behavioural nudges appended to Nikki's prompt.

Moved verbatim from app/main.py. The logic, thresholds and prompt wording are
unchanged; the only transformation is mechanical: module globals (`P()`,
`state.settings`) and the global RNG became explicit parameters on
`NoteContext`.

That change is the whole point. These functions decide whether Nikki offers to
help, whether she brings up something she noticed, and how she reacts to
nonsense — and none of it could be tested, because every one of them reached
for a global profile registry and called `random.random()` directly. With the
dependencies injected, each note builder is now deterministic under test.

Design note for N3: collectively these ARE the current conversation engine, and
they work by string-concatenating instructions onto the system prompt. That is
the approach the conversation-engine task replaces with an explicit turn
planner. This module is the honest starting point for that work, not the
finished shape.
"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

logger = logging.getLogger("companion.conversation")

# Relationship stages at which Nikki does not yet volunteer offers or
# observations — she hasn't earned the familiarity.
RESERVED_STAGES = ("stranger", "acquaintance")


class _DB(Protocol):
    """The slice of app.db.Database these notes actually use."""

    def get_setting(self, key: str) -> str | None: ...
    def set_setting(self, key: str, value: str) -> None: ...
    def list_memories_by_category(self, category: str) -> list[dict]: ...
    def delete_memory(self, memory_id: int) -> bool: ...
    def get_awaiting_followup(self, session_id: str) -> dict | None: ...
    def resolve_event_followup(self, followup_id: int) -> None: ...


@dataclass
class NoteContext:
    """Everything a note builder needs, passed in rather than reached for.

    `rng` is injectable so the throttled/probabilistic notes can be tested
    deterministically instead of being effectively untestable.
    """

    db: Any
    behavior: dict = field(default_factory=dict)
    stage: str | None = None
    rng: random.Random = field(default_factory=random.Random)

    @property
    def eagerness(self) -> float:
        return float(self.behavior.get("eagerness", 0.2))

    @property
    def offer_min_gap(self) -> int:
        return int(self.behavior.get("offer_min_gap", 6))

    @property
    def exchange_count(self) -> int:
        return int(self.db.get_setting("exchange_count") or 0)

    def is_reserved(self) -> bool:
        return (self.stage or "stranger") in RESERVED_STAGES

    def _int_setting(self, key: str, default: int = -999) -> int:
        try:
            return int(self.db.get_setting(key) or default)
        except (TypeError, ValueError):
            return default


# ---------------------------------------------------------------------------
# Event follow-up
# ---------------------------------------------------------------------------


def awaiting_followup_note(ctx: NoteContext, session_id: str) -> str | None:
    """If she's waiting on an answer to an event check-in ('how did it go?'),
    resolve it now - the NEXT user message is treated as the answer (see
    app/db.py get_awaiting_followup's docstring for the intended design) -
    and tell her this reply is likely that answer so she doesn't re-ask."""
    awaiting = ctx.db.get_awaiting_followup(session_id)
    if not awaiting:
        return None
    ctx.db.resolve_event_followup(awaiting["id"])
    return (f"NOTE: You recently asked them about '{awaiting['event_fact']}' - this "
            f"message is likely their answer. React naturally to what they say now; "
            f"don't ask again.")


# ---------------------------------------------------------------------------
# Offers
# ---------------------------------------------------------------------------

OFFER_TOPICS = {
    "food": r"\b(hungry|starving|haven'?t eaten|no food|skip(ping)? (lunch|dinner))\b",
    "weather": r"\b(so (hot|cold)|freezing|melting|raining|weather)\b",
    "money": r"\b(broke|money'?s tight|expensive|overspent)\b",
}


def offer_note(ctx: NoteContext, message: str) -> str | None:
    """Offer throttling: after a relevant MENTION she may offer an action -
    rate-limited, stage-gated, and permanently dropped once declined."""
    if ctx.is_reserved():
        return None

    topic = next((t for t, p in OFFER_TOPICS.items()
                  if re.search(p, message, re.I)), None)
    if not topic:
        return None

    try:
        declined = set(json.loads(ctx.db.get_setting("declined_offers") or "[]"))
    except json.JSONDecodeError:
        declined = set()
    if topic in declined:
        return None

    n = ctx.exchange_count
    last = ctx._int_setting("last_offer_at")
    if n - last < ctx.offer_min_gap or ctx.rng.random() > ctx.eagerness:
        return None

    ctx.db.set_setting("last_offer_at", str(n))
    # track_offer_decline() reads this back if the next message declines -
    # without it, a decline always recorded the literal string "last"
    # instead of the actual topic, so "permanently dropped once declined"
    # silently never worked.
    ctx.db.set_setting("last_offer_topic", topic)
    return (f"They just mentioned something about {topic}. Respond like a person "
            f"first (empathy/teasing/curiosity). You MAY casually offer to help "
            f"with it mid-conversation if it feels natural - one soft offer, "
            f"never as your opening line, and drop it instantly if declined.")


DECLINE_PATTERN = re.compile(
    r"^\s*(no+|nah|nope|don'?t|it'?s ok(ay)?|i'?m (fine|good))\b", re.I)


def track_offer_decline(ctx: NoteContext, message: str) -> None:
    """If she offered last turn and this reply is a decline, drop that topic."""
    if ctx._int_setting("last_offer_at", -1) != ctx.exchange_count - 1:
        return
    if DECLINE_PATTERN.match(message.strip()):
        try:
            declined = set(json.loads(ctx.db.get_setting("declined_offers") or "[]"))
        except json.JSONDecodeError:
            declined = set()
        declined.add(ctx.db.get_setting("last_offer_topic") or "last")
        ctx.db.set_setting("declined_offers", json.dumps(sorted(declined)))


# ---------------------------------------------------------------------------
# Long-term patterns and short-term streaks
# ---------------------------------------------------------------------------


def pattern_note(ctx: NoteContext, message: str = "") -> str | None:
    """Occasional, throttled reference to a weekly-detected mood-journal
    pattern (category="relationship" memories prefixed "Pattern: ") - the
    girlfriend part of the journal: gentle noticing, never a report. Gated by
    the same eagerness dial as offer_note, but content-independent and much
    rarer (patterns aren't tied to any one message)."""
    if ctx.is_reserved():
        return None

    gap = ctx.offer_min_gap * 4
    # Cheap gate check BEFORE the DB query - this runs on every message but
    # the roll usually says no, so querying memories first was pure waste.
    n = ctx.exchange_count
    last_n = ctx._int_setting("last_pattern_ref_at")
    if n - last_n < gap or ctx.rng.random() > ctx.eagerness * 0.4:
        return None

    patterns = [m for m in ctx.db.list_memories_by_category("relationship")
                if (m.get("fact") or "").startswith("Pattern:")]
    if not patterns:
        return None

    last_id = ctx._int_setting("last_pattern_ref_id", 0)
    candidate = next((p for p in patterns if p["id"] != last_id), patterns[0])
    ctx.db.set_setting("last_pattern_ref_at", str(n))
    ctx.db.set_setting("last_pattern_ref_id", str(candidate["id"]))
    fact = candidate["fact"][len("Pattern:"):].strip()
    return (f"NOTE: from quietly paying attention over time you've noticed this about "
            f"them: {fact}. You MAY bring it up naturally if the moment fits - as one "
            f"gentle, caring observation, never as a report/stats/list, never mentioning "
            f"a 'journal' or that you track anything. Skip it entirely if it doesn't fit.")


def streak_note(
    ctx: NoteContext,
    message: str = "",
    on_consume: Callable[[int], None] | None = None,
) -> str | None:
    """A SHORT-TERM rough-streak flag (run_recent_streak_check, chained onto
    the NIGHTLY job - distinct from the long-term weekly Pattern: system
    above). Surfaces promptly: a short gap (not pattern_note's *4 throttle),
    since 'you've seemed off the last few days' is time-sensitive - bringing
    it up two weeks late would feel odd. One-shot: consumed (deleted) the
    moment it's used, unlike a genuine Pattern: which stays referenceable.

    `on_consume` receives the deleted memory id so the caller can drop the
    matching vector from the ANN index (previously an inline P().memory.remove).
    """
    if ctx.is_reserved():
        return None

    n = ctx.exchange_count
    last_n = ctx._int_setting("last_streak_ref_at")
    if n - last_n < ctx.offer_min_gap or ctx.rng.random() > ctx.eagerness:
        return None

    streaks = [m for m in ctx.db.list_memories_by_category("relationship")
               if (m.get("fact") or "").startswith("Streak:")]
    if not streaks:
        return None

    candidate = streaks[0]
    ctx.db.set_setting("last_streak_ref_at", str(n))
    ctx.db.delete_memory(candidate["id"])
    if on_consume is not None:
        try:
            on_consume(int(candidate["id"]))
        except Exception as e:  # noqa: BLE001 - never let cleanup break a reply
            logger.warning("streak vector cleanup failed: %s", e)

    fact = candidate["fact"][len("Streak:"):].strip()
    return (f"NOTE: you've quietly noticed this about how they've been the last few "
            f"days: {fact}. Bring it up naturally as one gentle, caring check-in if "
            f"the moment fits - never as a report, never mentioning a 'journal' or "
            f"that you track anything. Skip it entirely if it doesn't fit right now.")


def unresolved_note(
    ctx: NoteContext,
    message: str = "",
    on_consume: Callable[[int], None] | None = None,
) -> str | None:
    """A specific concern (run_unresolved_check, chained onto the nightly job
    after the streak check - N6) that was logged once and never mentioned
    again. Distinct from streak_note (aggregate mood over recent days) and
    pattern_note (a label recurring across weeks): this is ONE named thing
    that seems to have gone quiet with no resolution ever surfacing.

    Same throttle/one-shot-consumption shape as streak_note deliberately -
    the underlying mechanics (rate-limited, deletes on use) are identical;
    only the source memory prefix and framing differ.
    """
    if ctx.is_reserved():
        return None

    n = ctx.exchange_count
    last_n = ctx._int_setting("last_unresolved_ref_at")
    if n - last_n < ctx.offer_min_gap or ctx.rng.random() > ctx.eagerness:
        return None

    unresolved = [m for m in ctx.db.list_memories_by_category("relationship")
                 if (m.get("fact") or "").startswith("Unresolved:")]
    if not unresolved:
        return None

    candidate = unresolved[0]
    ctx.db.set_setting("last_unresolved_ref_at", str(n))
    ctx.db.delete_memory(candidate["id"])
    if on_consume is not None:
        try:
            on_consume(int(candidate["id"]))
        except Exception as e:  # noqa: BLE001 - never let cleanup break a reply
            logger.warning("unresolved vector cleanup failed: %s", e)

    fact = candidate["fact"][len("Unresolved:"):].strip()
    return (f"NOTE: something you noticed and haven't checked in on since: {fact}. "
            f"If the moment fits naturally, ask how that ended up going - gently, "
            f"like you actually remembered, not like a report. Skip it entirely if "
            f"it doesn't fit right now.")


# ---------------------------------------------------------------------------
# Nonsense/spam realism: 30x "rrrrrr" used to make her confabulate an entire
# fake evening (invented plans, times, random memory fragments) because the
# model had no signal the input was noise. A real person notices immediately.
# ---------------------------------------------------------------------------

_LETTERS = re.compile(r"[a-zA-Z]+")
# Real texting tokens that survive collapsing but have no vowel (or are
# single letters) - must never count as keyboard-mash.
_SHORT_REAL = {"i", "u", "y", "k", "hm", "mhm", "ty", "np", "gm", "gn",
               "idk", "tbh", "btw", "rn", "pls", "plz", "thx", "xd",
               "shh", "psst", "tsk", "brb", "wtf", "smh", "fr", "ngl"}


def is_gibberish(text: str) -> bool:
    """Keyboard-mash detector: no plausible word in the message. Pure
    emoji/punctuation ('??', '😂😂') is a real signal, NOT gibberish.
    Elongations ('noooo', 'hmmmm') collapse first so they read as words."""
    t = text.strip().lower()
    if not t or not re.search(r"[a-zA-Z]", t):
        return False
    for w in _LETTERS.findall(t):
        w = re.sub(r"(.)\1{2,}", r"\1", w)  # nooo -> no, hmmm -> hm
        if w in _SHORT_REAL or (len(w) >= 2 and set(w) & set("aeiou")):
            return False
    return True


def nonsense_note(ctx: NoteContext, message: str) -> str | None:
    """Track consecutive gibberish/identical messages and hand the model a
    human way out. Returns a prompt note while a streak is active."""
    norm = re.sub(r"\s+", " ", message.strip().lower())
    last = ctx.db.get_setting("last_user_msg") or ""
    ctx.db.set_setting("last_user_msg", norm)

    gib = is_gibberish(message)
    repeat = bool(norm) and norm == last
    streak = ctx._int_setting("nonsense_streak", 0)
    streak = streak + 1 if (gib or repeat) else 0
    ctx.db.set_setting("nonsense_streak", str(streak))
    if streak == 0:
        return None

    what = "keyboard-mash gibberish" if gib else "the exact same message again"
    if streak >= 4:
        return (
            f"NOTE: they've now sent {what} {streak} times in a row. Stop playing "
            "along like it means something: reply with ONE very short dry line "
            "('ok you're clearly just mashing your keyboard 😂' / '...say something "
            "real and i'll answer'). Do NOT invent any events, people, plans or "
            "times. No questions.")
    return (
        f"NOTE: their message is just {what}. React like a real person would - "
        "confused or teasing ('did your cat walk on your keyboard?'), one short "
        "line. Do NOT treat it as meaningful, and do NOT invent plans, people, "
        "times or topics to fill the silence.")


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------


def persona_other_names(persona) -> list[str]:
    """Names of people from HER OWN backstory (life.friends) - never valid
    names for the person she's actually texting. See scan_identity_confusion."""
    friends = getattr(persona, "life", None) or {}
    return [f.get("name", "") for f in (friends.get("friends") or [])
            if isinstance(f, dict) and f.get("name")]
