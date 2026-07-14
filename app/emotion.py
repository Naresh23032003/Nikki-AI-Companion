"""Emotion tags for Call-mode avatar reactions.

In Call mode the model appends a trailing JSON emotion tag to each reply, e.g.

    hey! good to hear your voice {"emotion": "happy"}

The backend strips the tag from the spoken/stored text and forwards the emotion
to the frontend, which maps it to avatar eye variants, head motion, and
particle effects (hearts/sparkles).

IMPORTANT: models sometimes emit emotions outside our set ("curious",
"concerned", …) or add tags even in chat mode (by imitation). So the stripping
matches ANY `{"emotion": "<word>"}` tag and unknown values collapse to neutral -
that way a stray tag never leaks into a text bubble, a voice note, or history.
"""
from __future__ import annotations

import re
from typing import Tuple

EMOTIONS = {"happy", "laughing", "shy", "sad", "surprised", "neutral", "love"}
DEFAULT_EMOTION = "neutral"

# Appended to prompts (call mode has its own copy; WhatsApp uses this) so the
# reply carries silent emotion metadata for avatar reactions / sticker choice.
EMOTION_TAG_INSTRUCTION = """\
At the VERY END of your reply, append your current emotion as a JSON tag on its
own, exactly like: {"emotion": "happy"}
Allowed values: happy, laughing, shy, sad, surprised, neutral, love.
This tag is silent metadata - never reference it or say it out loud."""

# Matches an emotion tag with ANY word value (not just the allowed set) so stray
# emotions are still stripped. Quoting/spacing is flexible.
_EMO_ANY = re.compile(r'\{\s*"?emotion"?\s*:\s*"?([A-Za-z_]+)"?\s*\}', re.IGNORECASE)

# Small models sometimes emit the tag WITHOUT braces ('... for now! "emotion":
# "shy"'), which the brace-matching regex misses - that leaked verbatim into a
# WhatsApp bubble. Requiring the QUOTED key keeps this from ever touching a
# genuine sentence that happens to contain the word emotion.
_EMO_BARE = re.compile(r'["“]emotion["”]\s*:\s*["“]?([A-Za-z_]+)["”]?',
                       re.IGNORECASE)

# A bare trailing JSON object (fallback: strip any `{...}` left dangling at the end).
_TRAILING_JSON = re.compile(r'\{[^{}]*\}\s*$')

# Small models imitate stage-direction fragments like "{love}" / "{morning}" /
# "{yeah, it's nice}" once one appears in history. Strip any short brace group -
# nobody texts literal {braces}.
_BRACE_FRAGMENT = re.compile(r'\{[^{}]{0,60}\}')

# Roleplay-style action asides like "*checks*" / "*giggles*" / "*hugs you*"
# read as pure bot; strip short single-asterisk fragments too. (Losing a rare
# *emphasis* word is an acceptable trade - actions are far more common.)
_ASTERISK_FRAGMENT = re.compile(r'\*[^*\n]{1,40}\*')


def parse_emotion(text: str) -> Tuple[str, str]:
    """Return (clean_text, emotion). Uses the LAST tag; unknown -> neutral."""
    matches = list(_EMO_ANY.finditer(text)) or list(_EMO_BARE.finditer(text))
    emotion = DEFAULT_EMOTION
    if matches:
        val = matches[-1].group(1).lower()
        emotion = val if val in EMOTIONS else DEFAULT_EMOTION
    return strip_tags(text), emotion


def strip_tags(text: str) -> str:
    """Remove emotion tags AND stage-direction fragments (storage/display)."""
    text = _EMO_ANY.sub("", text)
    text = _EMO_BARE.sub("", text)
    text = _BRACE_FRAGMENT.sub("", text)
    text = _ASTERISK_FRAGMENT.sub("", text)
    return re.sub(r"  +", " ", text).strip()


def strip_for_speech(text: str) -> str:
    """Remove emotion tags (and any dangling tag fragment / trailing JSON) before TTS."""
    text = _EMO_ANY.sub("", text)
    text = _EMO_BARE.sub("", text)
    text = _BRACE_FRAGMENT.sub("", text)
    text = _ASTERISK_FRAGMENT.sub("", text)
    text = _TRAILING_JSON.sub("", text)
    brace = text.rfind("{")
    if brace != -1 and "emotion" in text[brace:].lower():
        text = text[:brace]
    return text.strip()
