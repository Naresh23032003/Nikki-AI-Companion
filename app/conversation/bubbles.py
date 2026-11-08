"""Splitting a reply into separate text bubbles.

Moved verbatim from app/main.py — logic and constants unchanged, module-level
constants now overridable per call so the thresholds can be exercised directly.

This is texting realism: a person sends a reaction and then the real thought as
two messages, rather than one paragraph. Small local models rarely emit the
blank-line signal the persona prompt asks for, so length-based sentence
splitting is the fallback.
"""
from __future__ import annotations

import re

MAX_BUBBLES = 4
AUTO_SPLIT_MIN_CHARS = 80
MIN_BUBBLE_CHARS = 12  # "right?" / "haha" rides along with its neighbor
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def sentence_bubbles(text: str, *, max_bubbles: int = MAX_BUBBLES,
                     min_bubble_chars: int = MIN_BUBBLE_CHARS) -> list[str]:
    """One bubble per sentence; tiny fragments fold into the previous one.

    Replies with more sentences than `max_bubbles` get the sentences grouped
    into `max_bubbles` roughly length-balanced bubbles — folding all overflow
    into the LAST bubble re-created the exact wall-of-text this exists to
    prevent (observed: a 333-char final bubble).
    """
    sentences = [s.strip() for s in SENTENCE_END.split(text) if s.strip()]
    if len(sentences) < 2:
        return [text]

    bubbles: list[str] = []
    for s in sentences:
        if bubbles and len(s) < min_bubble_chars:
            bubbles[-1] = f"{bubbles[-1]} {s}"
        else:
            bubbles.append(s)

    if len(bubbles) <= max_bubbles:
        return bubbles

    per_bubble = sum(len(b) for b in bubbles) / max_bubbles
    grouped: list[str] = []
    current = ""
    for b in bubbles:
        if (current and len(grouped) < max_bubbles - 1
                and len(current) + len(b) > per_bubble * 1.15):
            grouped.append(current)
            current = b
        else:
            current = f"{current} {b}".strip()
    if current:
        grouped.append(current)
    return grouped


def split_bubbles(text: str, *, max_bubbles: int = MAX_BUBBLES,
                  auto_split_min_chars: int = AUTO_SPLIT_MIN_CHARS,
                  min_bubble_chars: int = MIN_BUBBLE_CHARS) -> list[str]:
    """Split a reply into separate text bubbles.

    Primary signal: blank lines — what the behaviour rules tell her to use for
    'texted twice' (a reaction, then the real thought). Small local models
    rarely emit that signal though, so multi-sentence replies over
    ~`auto_split_min_chars` additionally get split one-sentence-per-bubble — a
    thought per send, like real texting. Short replies stay one bubble.
    """
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]

    if len(parts) <= 1:
        cleaned = text.strip()
        if not cleaned:
            return []
        if len(cleaned) >= auto_split_min_chars:
            return sentence_bubbles(cleaned, max_bubbles=max_bubbles,
                                    min_bubble_chars=min_bubble_chars)
        return [cleaned]

    if len(parts) > max_bubbles:
        # Overflow folds into the last bubble rather than being dropped.
        parts = parts[:max_bubbles - 1] + [" ".join(parts[max_bubbles - 1:])]
    return parts
