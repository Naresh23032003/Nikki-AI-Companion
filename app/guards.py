"""Code-enforced guards: she never fakes competence, never sounds like a bot.

- Capability manifest: injected into the system prompt - the things she CANNOT
  do herself and must get from tools.
- Forbidden-claims scan: replies claiming completed actions with no matching
  tool result this turn get blocked -> one regeneration with a correction note
  -> stubborn claims get the sentence replaced with an honest line.
- Honeypots: specific prices/news-ish claims with no tool run -> flag.
- Assistant-speak ban: helper phrases, bullet lists, multi-questions.
- Reaction scan: react-then-deliver reactions must contain zero status talk.

All violations are logged (counter in app_settings under guard_stats).
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger("companion.guards")

CAPABILITY_MANIFEST = """\
THINGS YOU CANNOT DO YOURSELF (they require your tools; NEVER pretend
otherwise, never fake their results):
- Math beyond trivial arithmetic; any calculation someone would use a
  calculator for.
- Ordering/booking/paying for anything.
- Current facts: news, prices, weather, sports scores, anything that changes.
- Setting reminders/alarms - only confirm one AFTER the tool result says it
  was saved.
If asked for these and no tool result is provided in this turn, say honestly
that you'll sort it out / you're not sure - never invent an answer."""

# action-claim verbs that require a matching tool result this turn
_ACTION_CLAIMS = re.compile(
    r"\b(i('ve| have)? (just )?(ordered|booked|paid|bought|scheduled|reserved)|"
    r"reminder('s| is)? (set|saved|done)|i set (a |the )?(reminder|alarm)|"
    r"i checked (the )?(weather|news|price)|i looked it up|i searched)\b", re.I)

# honeypots: precise fact-claims that need a tool behind them
_HONEYPOTS = re.compile(
    r"(₹|\$|£|€)\s?\d{2,}|\b\d+(\.\d+)?\s?(USD|INR|EUR)\b|"
    r"\b(breaking|announced today|just released|latest version is)\b|"
    r"\bit('s| is) (\d+|-?\d+\.?\d*)\s?°", re.I)

_ASSISTANT_SPEAK = [
    (re.compile(r"\bhow can i (help|assist)\b", re.I), "helper-phrase"),
    (re.compile(r"\bi can assist\b", re.I), "helper-phrase"),
    (re.compile(r"\bas an ai\b", re.I), "ai-disclosure"),
    (re.compile(r"^\s*would you like me to\b", re.I), "opener-offer"),
    # Support-desk closers - nobody texts their friend "glad i could help".
    (re.compile(r"\bglad i could\b", re.I), "helper-closer"),
    (re.compile(r"\bit('s| is) all set( now)?\b", re.I), "helper-closer"),
    (re.compile(r"\blet me know if (you|there)\b", re.I), "helper-closer"),
    (re.compile(r"\bhope (this|that) helps\b", re.I), "helper-closer"),
    (re.compile(r"^\s*[-*•]\s+\S", re.M), "bullet-list"),
    (re.compile(r"^\s*\d+[.)]\s+\S", re.M), "numbered-list"),
    (re.compile(r"^#{1,4}\s", re.M), "markdown-header"),
]

# react-then-deliver reactions must not leak status language
_STATUS_WORDS = re.compile(
    r"\b(check(ing)?|look(ing)? (it |that )?up|search(ing)?|fetch(ing)?|"
    r"one sec|gimme a sec|a moment|hold on|wait while i|let me (see|find|check)|"
    r"i'll (find|get|look)|working on it)\b", re.I)


def _bump(db, key: str) -> None:
    if not db:
        return
    try:
        stats = json.loads(db.get_setting("guard_stats") or "{}")
    except json.JSONDecodeError:
        stats = {}
    stats[key] = stats.get(key, 0) + 1
    db.set_setting("guard_stats", json.dumps(stats))


def scan_forbidden_claims(reply: str, tool_ran: bool, db=None) -> list[str]:
    """Claimed completed actions with no tool result this turn."""
    if tool_ran:
        return []
    hits = [m.group(0) for m in _ACTION_CLAIMS.finditer(reply)]
    if hits:
        _bump(db, "forbidden_claim")
        logger.warning("guard: forbidden action-claims %s in %r", hits, reply[:80])
    return hits


def scan_honeypots(reply: str, tool_ran: bool, db=None) -> list[str]:
    """Specific price/news/measurement claims with no tool behind them."""
    if tool_ran:
        return []
    hits = [m.group(0) for m in _HONEYPOTS.finditer(reply)]
    if hits:
        _bump(db, "honeypot")
        logger.warning("guard: honeypot claims %s in %r", hits, reply[:80])
    return hits


def scan_assistant_speak(reply: str, db=None) -> list[str]:
    hits = []
    for pat, name in _ASSISTANT_SPEAK:
        if pat.search(reply):
            hits.append(name)
    if reply.count("?") > 1:
        hits.append("multi-question")
    if hits:
        _bump(db, "assistant_speak")
        logger.warning("guard: assistant-speak %s in %r", hits, reply[:80])
    return hits


def scan_reaction(reply: str, db=None) -> list[str]:
    """Zero task/status language allowed in a react-then-deliver reaction."""
    hits = [m.group(0) for m in _STATUS_WORDS.finditer(reply)]
    if hits:
        _bump(db, "reaction_status")
    return hits


def strip_violating_sentences(reply: str, patterns: list[re.Pattern],
                              replacement: str | None = None) -> str:
    """Remove (or replace) sentences containing any of the given patterns."""
    parts = re.split(r"(?<=[.!?])\s+", reply)
    kept = []
    replaced = False
    for s in parts:
        if any(p.search(s) for p in patterns):
            if replacement and not replaced:
                kept.append(replacement)
                replaced = True
            continue
        kept.append(s)
    out = " ".join(kept).strip()
    return out or (replacement or "hmm, lost my train of thought 😅")


HONEST_LINE = "okay wait - I actually haven't done that yet, let me not get ahead of myself 😅"

CLAIM_PATTERNS = [_ACTION_CLAIMS]
STATUS_PATTERNS = [_STATUS_WORDS]


def guard_stats(db) -> dict:
    try:
        return json.loads(db.get_setting("guard_stats") or "{}")
    except json.JSONDecodeError:
        return {}
