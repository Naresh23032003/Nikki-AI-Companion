"""AI-tell detection (N3).

These are the structural habits that make a reply read as machine-generated.
None of them are about vocabulary — they survive any amount of "sound natural"
prompting, because they come from a model optimising each turn in isolation
with no memory of the conversation's shape.

Used two ways: as an eval over generated replies, and as a live guard that can
ask for a regeneration when a reply scores badly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence

# Each tell costs this much naturalness. Three tells puts a reply under 0.6.
TELL_WEIGHT = 0.15


@dataclass
class TurnContext:
    user_message: str = ""
    recent_replies: Sequence[str] = field(default_factory=list)


@dataclass
class Tell:
    name: str
    detail: str = ""


def question_streak(replies: Sequence[str]) -> int:
    """How many of the most recent replies ended in a question, consecutively."""
    n = 0
    for reply in reversed(list(replies)):
        if reply.strip().endswith("?"):
            n += 1
        else:
            break
    return n


_VALIDATION_OPENERS = re.compile(
    r"^\s*(that'?s\s+(so|really|completely)?\s*(valid|fair|understandable|true)"
    r"|it\s+makes\s+(total|complete|perfect)?\s*sense"
    r"|i\s+(totally|completely|really)\s+(get|understand)"
    r"|you'?re\s+(so|absolutely|totally)\s+right"
    r"|that\s+sounds\s+(so|really)\s+(valid|hard|tough)\s*$)", re.I)

_GENERIC_QUESTIONS = re.compile(
    r"(how\s+(does|did)\s+that\s+make\s+you\s+feel"
    r"|what'?s\s+on\s+your\s+mind"
    r"|do\s+you\s+want\s+to\s+talk\s+about\s+it"
    r"|tell\s+me\s+more\s+about"
    r"|how\s+are\s+you\s+feeling\s+about\s+(that|it)"
    r"|is\s+there\s+anything\s+else)", re.I)

_SUMMARISING = re.compile(
    r"^\s*(so|it\s+sounds\s+like|what\s+i'?m\s+hearing)\s+"
    r"(you'?re\s+saying|you\s+(feel|felt|mean)|like\s+you)", re.I)

_ENTHUSIASM = re.compile(
    r"\b(that'?s\s+(amazing|incredible|fantastic|wonderful|awesome)"
    r"|how\s+exciting|so\s+happy\s+for\s+you)\b[!]*", re.I)

_HEDGES = re.compile(
    r"\b(i\s+think|maybe|might|perhaps|sort\s+of|kind\s+of|possibly|"
    r"probably|it\s+seems|i\s+guess|somewhat)\b", re.I)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_STOP = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at",
         "is", "are", "was", "it", "that", "this", "you", "i", "so"}


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _content(text: str) -> set[str]:
    return {w for w in _words(text) if w not in _STOP and len(w) > 2}


def _user_is_excited(text: str) -> bool:
    """Caps or multiple exclamation marks — enthusiasm she can legitimately match."""
    letters = [c for c in text if c.isalpha()]
    caps_ratio = (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0
    return text.count("!") >= 2 or caps_ratio > 0.5


def detect_tells(reply: str, ctx: TurnContext | None = None) -> List[Tell]:
    """All structural tells present in `reply`."""
    ctx = ctx or TurnContext()
    reply = (reply or "").strip()
    if not reply:
        return []

    tells: List[Tell] = []
    reply_words = _words(reply)
    user_words = _words(ctx.user_message or "")

    # A question is fine. A third question in a row is an interrogation.
    if reply.endswith("?") and question_streak(ctx.recent_replies) >= 2:
        tells.append(Tell("reflexive_question",
                          "third consecutive question"))

    # A paragraph in reply to "ok" is a mismatch of register.
    if len(user_words) <= 2 and len(reply_words) > 12:
        tells.append(Tell("length_mismatch",
                          f"{len(reply_words)} words replying to {len(user_words)}"))

    if _VALIDATION_OPENERS.search(reply):
        tells.append(Tell("over_agreement", "opens by validating"))

    if _GENERIC_QUESTIONS.search(reply):
        tells.append(Tell("generic_question", "content-free question"))

    if _SUMMARISING.search(reply):
        tells.append(Tell("summarising_back", "reflects the message back"))

    if _ENTHUSIASM.search(reply) and not _user_is_excited(ctx.user_message or ""):
        tells.append(Tell("unearned_enthusiasm", "enthusiasm the user didn't show"))

    if len(_HEDGES.findall(reply)) >= 3:
        tells.append(Tell("stacked_hedging", "three or more hedges"))

    sentences = [s for s in _SENTENCE_SPLIT.split(reply) if s.strip()]
    if len(sentences) >= 3:
        lengths = [len(_words(s)) for s in sentences]
        spread = max(lengths) - min(lengths)
        openers = [(_words(s) or [""])[0] for s in sentences]
        if spread <= 2 or len(set(openers)) == 1:
            tells.append(Tell("symmetrical_rhythm",
                              f"sentence lengths {lengths}"))

    for previous in ctx.recent_replies:
        shared = _content(reply) & _content(previous)
        if len(shared) >= 3:
            tells.append(Tell("repetition", f"repeats: {' '.join(sorted(shared))}"))
            break

    return tells


def tell_names(reply: str, ctx: TurnContext | None = None) -> List[str]:
    return [t.name for t in detect_tells(reply, ctx)]


def naturalness_score(reply: str, ctx: TurnContext | None = None) -> float:
    """1.0 is clean. Each tell costs TELL_WEIGHT, floored at 0."""
    return max(0.0, 1.0 - TELL_WEIGHT * len(detect_tells(reply, ctx)))
