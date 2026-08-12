"""Pure logic for day-state generation coherence (N5).

Problem: `app/dayseed.py` generated each day's hidden state from only
YESTERDAY's single state dict + a hardcoded `trend = "steady"`. Two concrete
coherence failures followed:

1. `ongoing_threads` (6 items in a typical persona) were "progressed" by
   letting the model freely pick any one of them each day with no memory of
   which it touched when. A thread could restart, duplicate, or go silent for
   weeks at random - "disconnected diary events" instead of a story that
   actually moves. Fixed here with explicit least-recently-touched selection
   and a persisted per-thread status, so "done" storylines stop recurring.
2. The "relationship mood trend" fed to the generator was never real data -
   `trend = "steady"` was a literal constant in every single call, dressed up
   in a prompt that reads as if it reflects something. Fixed here with a real
   two-snapshot comparison.

Kept as pure functions (no db/llm/datetime.now() calls) so every branch is
testable without a database or a model, matching the app.memory_core /
app.conversation.planner precedent already established in this codebase.
"""
from __future__ import annotations

from dataclasses import dataclass

TREND_RISING = "warming up"
TREND_FALLING = "cooling off"
TREND_STEADY = "steady"

# Affection points of movement below which the trend reads as noise, not a
# real trend - the exchange-level clamp is ±2 (see app/relationship.py), so a
# single normal exchange must not itself register as "warming up".
TREND_DELTA_THRESHOLD = 1.5


def compute_mood_trend(current_affection: float | None,
                       previous_affection: float | None) -> str:
    """Real relationship trend from two affection snapshots.

    Either snapshot missing (first run, or affection tracking unavailable)
    degrades to STEADY - the same value the old hardcoded constant always
    produced, so behaviour only improves, never regresses to something worse
    than before."""
    if current_affection is None or previous_affection is None:
        return TREND_STEADY
    delta = current_affection - previous_affection
    if delta >= TREND_DELTA_THRESHOLD:
        return TREND_RISING
    if delta <= -TREND_DELTA_THRESHOLD:
        return TREND_FALLING
    return TREND_STEADY


@dataclass(frozen=True)
class ThreadState:
    text: str
    status: str = "active"           # "active" | "done"
    last_touched: str | None = None  # ISO date string; None = never touched

    def to_dict(self) -> dict:
        return {"text": self.text, "status": self.status,
                "last_touched": self.last_touched}

    @staticmethod
    def from_dict(d: dict) -> "ThreadState":
        return ThreadState(text=d.get("text", ""),
                           status=d.get("status") or "active",
                           last_touched=d.get("last_touched"))


def sync_threads(existing: list[ThreadState],
                 thread_texts: list[str]) -> list[ThreadState]:
    """Reconcile persisted thread state against the persona's current
    `ongoing_threads` list from the YAML.

    Threads still listed keep their persisted status/last_touched; threads no
    longer listed are dropped; new ones start fresh. Without this, editing
    personas/*.yaml would either orphan old state forever or (worse) let a
    stale index silently point at the wrong thread text after a reorder."""
    by_text = {t.text: t for t in existing}
    return [by_text.get(text, ThreadState(text=text)) for text in thread_texts]


def select_thread_to_progress(threads: list[ThreadState]) -> ThreadState | None:
    """The active thread least recently touched - round-robin coherence.

    Replaces "the model freely picks any of N threads each call" with a
    deterministic choice made BEFORE generation, the same before-not-during
    principle N3's turn planner uses for conversation shape. Never-touched
    threads (last_touched=None) sort first."""
    active = [t for t in threads if t.status == "active"]
    if not active:
        return None
    return min(active, key=lambda t: t.last_touched or "")


def advance_thread(threads: list[ThreadState], text: str, *,
                   done: bool, today: str) -> list[ThreadState]:
    """Record that `text`'s thread was progressed today.

    Marking it done removes it from `select_thread_to_progress`'s rotation
    for good, so a finished storyline actually stops reappearing - the
    complement of the old bug where nothing tracked completion at all."""
    return [
        ThreadState(text=t.text, status=("done" if done else "active"),
                   last_touched=today)
        if t.text == text else t
        for t in threads
    ]
