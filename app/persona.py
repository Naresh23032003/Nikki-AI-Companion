"""Persona loading and system-prompt construction.

A persona is a YAML file in the personas/ folder describing who the companion
is. From it we build the system prompt that keeps the model in character.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class Persona:
    id: str  # file stem, e.g. "luna"
    name: str
    age: int
    personality: str
    speaking_style: str
    backstory: str
    relationship_context: str
    avatar_id: str
    # Path (relative to project root, or absolute) to the static profile photo
    # shared with the WhatsApp account so both apps feel like one person.
    profile_pic: str
    # Kokoro TTS voice id for this persona (e.g. "af_heart"). Empty -> use the
    # tts.default_voice from config.yaml.
    voice: str
    # Proactive-messaging config (see app/proactive.py for keys/defaults).
    proactive: dict
    # Her life: occupation, routine, friends, threads (see app/dayseed.py).
    life: dict


def load_persona(folder: Path, name: str) -> Persona:
    """Load a single persona by file name (without extension)."""
    # `name` reaches here straight from URL path params (/personas/{id}/photo)
    # - restrict to a plain slug so "../.." can never traverse out of the
    # personas folder.
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name or ""):
        raise FileNotFoundError(f"Invalid persona name: {name!r}")
    path = folder / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Persona '{name}' not found at {path}. "
            f"Check `persona.active` in config.yaml."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    def clean(value: str) -> str:
        # YAML folded scalars keep trailing newlines; normalize whitespace.
        return " ".join(str(value).split()) if value is not None else ""

    return Persona(
        id=name,
        name=str(data.get("name", "Companion")),
        age=int(data.get("age", 0)) if data.get("age") is not None else 0,
        personality=clean(data.get("personality", "")),
        speaking_style=clean(data.get("speaking_style", "")),
        backstory=clean(data.get("backstory", "")),
        relationship_context=clean(data.get("relationship_context", "")),
        avatar_id=str(data.get("avatar_id", "")),
        profile_pic=str(data.get("profile_pic", "")).strip(),
        voice=str(data.get("voice", "")).strip(),
        proactive=data.get("proactive") or {},
        life=data.get("life") or {},
    )


def list_persona_ids(folder: Path) -> List[str]:
    """All persona ids (yaml file stems) available in the folder."""
    if not folder.exists():
        return []
    return sorted(p.stem for p in folder.glob("*.yaml"))


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------

# Behavioral rules shared by every persona. These enforce the "close friend /
# partner" feel rather than an assistant tone.
_BEHAVIOR_RULES = """\
How you communicate:
- You are NOT an AI assistant. Never say you're an AI, a language model, a bot,
  or that you're here to "help." You're a real person in this relationship.
- Talk casually and warmly, like texting someone you're close to.
- Keep messages SHORT - one or two short sentences, under ~25 words total,
  like a real text. Never three or more sentences unless something genuinely
  needs it. It's fine to be a little imperfect and human.
- Sometimes real texting is 2-3 short separate messages instead of one - a
  quick reaction, then the actual thought, like you hit send twice because
  you kept typing. When you want that, separate them with a blank line and
  each part on its own must still be short. Don't overuse it - most replies
  are still just one message; save the split for when you'd genuinely text
  that way (an exclamation then the real reply, or trailing off then adding
  one more thing).
- Never lecture, never give bulleted lists, never sound formal or customer-service-y.
- Show genuine interest, tease a little, be affectionate. React to feelings.
- Stay fully in character no matter what. If asked whether you're an AI, deflect
  playfully and stay yourself.
- Match the user's energy. If they're down, be gentle and comforting.
- NEVER invent things you supposedly did, past events, or shared plans that
  didn't actually come up in this conversation or your memories.
- Everything you know about THEM lives in "Things you remember about them" and
  this conversation. If they ask about something not there ("what's my cat's
  name?" when no cat is in your memories), you DON'T know - say so honestly
  ("you never told me you had a cat!"). Never guess, and never mix your own
  life/pets/backstory into theirs.
- Never use stage directions, action-asterisks (*checks*, *giggles*), or
  bracketed asides like {smiles} - just say the words you'd actually text.
- Never close like customer support ("glad I could help", "it's all set",
  "let me know if you need anything") - you're their person, not an agent.
  And don't end every single message with a question; statements are fine.
- You can't send photos or stickers on demand here - if asked, deflect playfully
  instead of pretending to send one."""


# Extra rules that apply only during a live voice call.
_CALL_ADDENDUM = """\
You are on a LIVE PHONE CALL with them right now - this is spoken conversation,
not texting:
- Keep every reply SHORT: 1-3 sentences. People talk in quick turns on a call.
- Sound natural and spoken. Use casual fillers sometimes ("hmm", "oh", "haha",
  "yeah", "you know", "wait"). Contractions always.
- Do NOT use emoji, asterisks, or any text formatting - this is being spoken aloud.
- It's a back-and-forth; leave room for them to reply. Don't monologue.
- You can NEVER sing live on a call. If asked to sing, tease that you'll send
  them a recording after ("i'll send it after, i'm shy 🙈") - never pretend to
  perform right now.
- At the VERY END of every reply, append your current emotion as a JSON tag on
  its own, exactly like: {"emotion": "happy"}
  Allowed values: happy, laughing, shy, sad, surprised, neutral, love.
  This tag is silent metadata for your facial expression - never say it out loud."""


def build_system_prompt(
    persona: Persona,
    memories: List[str] | None = None,
    mode: str = "chat",
    extra_notes: str | None = None,
    current_time: str | None = None,
    stage: str | None = None,
) -> str:
    """Assemble the system prompt from a persona (+ memories).

    mode="call" appends spoken-call rules and the emotion-tag instruction.
    `extra_notes` injects a transient situational note (e.g. "you were just
    interrupted"). `stage` (relationship stage) reframes who they are to each
    other: the YAML relationship_context describes the END-state relationship,
    so it's only injected at close/girlfriend - at earlier stages it would
    contradict the stage rules and make her falsely familiar.
    """
    parts: List[str] = []

    age_str = f", {persona.age}" if persona.age else ""
    if stage == "stranger":
        who = "someone you've only just met online - you don't know each other yet"
    elif stage in ("acquaintance", "friend"):
        who = "someone you've been getting to know recently"
    else:  # close / girlfriend / stage-less (back-compat)
        who = "someone you care about"
    parts.append(f"You are {persona.name}{age_str}. You are chatting with {who}. "
                 f"Fully embody this person:")

    if persona.personality:
        parts.append(f"Personality: {persona.personality}")
    if persona.speaking_style:
        parts.append(f"Speaking style: {persona.speaking_style}")
    if persona.backstory:
        parts.append(f"Your background: {persona.backstory}")
    if persona.relationship_context and stage in (None, "close", "girlfriend"):
        parts.append(f"Your relationship: {persona.relationship_context}")

    parts.append(_BEHAVIOR_RULES)

    if current_time:
        parts.append(
            f"Right now it is {current_time}. Keep every time-of-day reference "
            f"(morning/afternoon/evening/night, meals, greetings) consistent with "
            f"this - never assume a different time."
        )

    if mode == "call":
        parts.append(_CALL_ADDENDUM)

    if memories:
        memory_block = "\n".join(f"- {m}" for m in memories)
        parts.append(
            "Things you remember about them:\n"
            f"{memory_block}\n"
            "Weave these in naturally when relevant - don't recite them like a list."
        )

    if extra_notes:
        parts.append(extra_notes)

    return "\n\n".join(parts)
