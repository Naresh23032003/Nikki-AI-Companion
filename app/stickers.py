"""Sticker library: pick a sticker file matching an emotion / intent.

Drop WebP/PNG stickers into stickers/<kind>/ at the project root. Empty folders
are simply skipped - the system degrades to text-only until art exists.
"""
from __future__ import annotations

import random
from pathlib import Path

from app.config import ROOT

STICKER_ROOT = ROOT / "stickers"

KINDS = [
    "happy", "laughing", "shy", "love", "sad",
    "miss_you", "good_morning", "good_night", "angry_cute",
]

# Emotion tag / proactive intent -> sticker folder. None = no sticker.
KIND_FOR = {
    # emotions
    "happy": "happy",
    "laughing": "laughing",
    "shy": "shy",
    "love": "love",
    "sad": "sad",
    "surprised": "happy",
    "neutral": None,
    # proactive intents
    "good_morning": "good_morning",
    "goodnight": "good_night",
    "miss_you": "miss_you",
    "silence_react": "miss_you",
}

_EXTS = {".webp", ".png", ".jpg", ".jpeg"}


def ensure_dirs() -> None:
    for kind in KINDS:
        (STICKER_ROOT / kind).mkdir(parents=True, exist_ok=True)


def pick_sticker(emotion_or_intent: str) -> tuple[Path, str] | None:
    """Return (absolute path, public url) for a random matching sticker, or None."""
    kind = KIND_FOR.get(emotion_or_intent)
    if not kind:
        return None
    folder = STICKER_ROOT / kind
    if not folder.is_dir():
        return None
    files = [p for p in folder.iterdir() if p.suffix.lower() in _EXTS]
    if not files:
        return None
    chosen = random.choice(files)
    return chosen, f"/stickers/{kind}/{chosen.name}"
