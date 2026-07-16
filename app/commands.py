"""Keyword commands: adjust a persona at runtime from inside the chat.

Anyone can run these, but ONLY against their own profile - the command is
executed against the profile the message arrived on, so your friend can tune
his persona and you can tune yours, and neither can touch the other's.

    /help                 what she understands
    /closeness close      force relationship stage (dev-override style)
    /closeness 80         force affection 0-100 (stage follows organically)
    /mood flirty          tone override for the next replies (/mood off clears)
    /persona aria         switch which persona answers THIS number
    /personas             list available personas
    /proactive off        stop/start her unprompted messages
    /status               current persona, stage, affection, mood

Commands never reach the LLM, never enter memory, and never get a persona
reply - they return a terse confirmation so it's obvious they're mechanical.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from app.relationship import STAGES

logger = logging.getLogger("companion.commands")

# Mood is a free-text tone nudge injected into the system prompt. Kept short -
# it's a nudge, not a personality rewrite (the persona YAML still rules).
MOOD_KEY = "mood_override"
PROACTIVE_OFF_KEY = "proactive_disabled"
MAX_MOOD_LEN = 80

_CMD = re.compile(r"^\s*[/!](?P<name>[a-z_]+)\s*(?P<arg>.*)$", re.I | re.S)


@dataclass
class CommandResult:
    handled: bool
    reply: str = ""
    # Set when the command changed which persona answers this number, so the
    # caller can re-wire the profile before the next message.
    switch_persona: Optional[str] = None


def is_command(message: str) -> bool:
    return bool(_CMD.match(message or ""))


def _help_text(persona_names: list[str]) -> str:
    return (
        "commands:\n"
        "/closeness <stranger|acquaintance|friend|close|girlfriend> — set stage\n"
        "/closeness <0-100> — set affection\n"
        "/mood <text> — tone nudge (/mood off to clear)\n"
        f"/persona <{'|'.join(persona_names) or 'id'}> — switch persona\n"
        "/personas — list personas\n"
        "/proactive <on|off> — her unprompted messages\n"
        "/status — current state\n"
        "/help — this list"
    )


def handle(message: str, profile, persona_ids: list[str]) -> CommandResult:
    """Run a command against `profile` (the one the message arrived on).

    `profile` must expose .db, .relationship, .persona, .id.
    """
    m = _CMD.match(message or "")
    if not m:
        return CommandResult(handled=False)
    name = m.group("name").lower()
    arg = (m.group("arg") or "").strip()
    db = profile.db

    try:
        if name in ("help", "commands"):
            return CommandResult(True, _help_text(persona_ids))

        if name in ("closeness", "close", "stage", "affection"):
            if not arg:
                st = profile.relationship.state()
                return CommandResult(True, f"closeness: {st['stage']} "
                                           f"(affection {st['affection']:.0f}/100)")
            # numeric -> affection, word -> stage
            if re.fullmatch(r"\d{1,3}(\.\d+)?", arg):
                value = max(0.0, min(100.0, float(arg)))
                st = profile.relationship.override(affection=value)
                return CommandResult(True, f"affection -> {st['affection']:.0f}/100 "
                                           f"(stage {st['stage']})")
            stage = arg.lower()
            if stage not in STAGES:
                return CommandResult(True, f"stage must be one of: {', '.join(STAGES)}")
            st = profile.relationship.override(stage=stage)
            logger.warning("commands[%s]: closeness -> %s", profile.id, stage)
            return CommandResult(True, f"closeness -> {st['stage']}")

        if name == "mood":
            if not arg or arg.lower() in ("off", "clear", "none", "reset"):
                db.set_setting(MOOD_KEY, "")
                return CommandResult(True, "mood cleared")
            mood = arg[:MAX_MOOD_LEN]
            db.set_setting(MOOD_KEY, mood)
            logger.info("commands[%s]: mood -> %r", profile.id, mood)
            return CommandResult(True, f"mood -> {mood}")

        if name == "personas":
            return CommandResult(True, "personas: " + ", ".join(persona_ids))

        if name == "persona":
            if not arg:
                return CommandResult(True, f"persona: {profile.persona_id}")
            target = arg.strip().lower()
            if target not in persona_ids:
                return CommandResult(
                    True, f"no persona '{target}'. available: {', '.join(persona_ids)}")
            if target == profile.persona_id:
                return CommandResult(True, f"already {target}")
            logger.warning("commands[%s]: persona -> %s", profile.id, target)
            return CommandResult(True, f"persona -> {target}", switch_persona=target)

        if name == "proactive":
            if arg.lower() in ("off", "0", "no", "stop"):
                db.set_setting(PROACTIVE_OFF_KEY, "1")
                return CommandResult(True, "proactive messages off")
            if arg.lower() in ("on", "1", "yes", "start"):
                db.set_setting(PROACTIVE_OFF_KEY, "")
                return CommandResult(True, "proactive messages on")
            state = "off" if db.get_setting(PROACTIVE_OFF_KEY) else "on"
            return CommandResult(True, f"proactive is {state} (use /proactive on|off)")

        if name in ("status", "state"):
            st = profile.relationship.state()
            mood = db.get_setting(MOOD_KEY) or "-"
            pro = "off" if db.get_setting(PROACTIVE_OFF_KEY) else "on"
            return CommandResult(True, (
                f"persona: {profile.persona_id}\n"
                f"closeness: {st['stage']} (affection {st['affection']:.0f}/100)\n"
                f"days known: {st['days_known']} · memories: {st['memory_count']}\n"
                f"mood: {mood}\nproactive: {pro}\nprofile: {profile.id}"))

    except Exception as e:  # noqa: BLE001 - a bad command must never 500 the turn
        logger.warning("commands[%s]: %r failed: %s", profile.id, message[:40], e)
        return CommandResult(True, f"couldn't do that: {e}")

    return CommandResult(True, f"unknown command /{name} — try /help")


def mood_note(db) -> str | None:
    """System-prompt note for an active /mood override, if any."""
    mood = (db.get_setting(MOOD_KEY) or "").strip()
    if not mood:
        return None
    return (f"CURRENT MOOD (temporary tone nudge, stay yourself underneath): "
            f"{mood}")


def proactive_disabled(db) -> bool:
    return bool(db.get_setting(PROACTIVE_OFF_KEY))
