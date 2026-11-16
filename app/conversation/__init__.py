"""Conversation layer: what Nikki says, and why.

Extracted from app/main.py, where these decisions lived as private helpers
tangled with module globals. Splitting them out is the prerequisite for the
conversation engine (N3): you cannot redesign turn-taking behaviour that has no
tests, and this logic could not be tested while it reached for `P()`, `state`
and the global RNG.
"""
from app.conversation.notes import (  # noqa: F401
    NoteContext,
    awaiting_followup_note,
    is_gibberish,
    nonsense_note,
    offer_note,
    pattern_note,
    persona_other_names,
    streak_note,
    track_offer_decline,
)
