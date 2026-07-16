"""Multi-persona profiles: one isolated world per WhatsApp number.

A profile bundles everything that must NOT be shared between two people
talking to the app:

  - persona          (who she is for this person)
  - db               (its OWN SQLite file - messages, memories, entities,
                      relationship_state, reminders, every app_setting)
  - memory           (its OWN Chroma collection)
  - relationship     (affection/stage/upset - keyed to that db)
  - daylife          (her day, seeded per persona)
  - tools/tool_ctx   (so the draw tool draws the RIGHT persona)
  - proactive        (her own schedule of unprompted messages)

Isolation is by SEPARATE DATABASE FILE rather than a `profile` column: the
schema has a single-row relationship_state (CHECK id = 1) and no tenant
column on memories/entities/relations, so a column-based split would mean
rewriting every query and would leave cross-leaks one forgotten WHERE away.
Separate files make a leak structurally impossible, need no migration, and
leave profile #1 byte-identical to the single-persona setup.

Config (config.yaml):

    profiles:
      - id: main                 # stable key; also the DB/collection suffix
        number: "919876543210"   # digits only, country code, no +
        persona: luna
        default: true            # gets the EXISTING companion.db + collection
      - id: friend
        number: "919999999999"
        persona: aria

With no `profiles:` block the app falls back to one implicit profile using
persona.active + the existing db/collection, i.e. exactly the old behavior.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("companion.profiles")


def normalize_number(raw: str) -> str:
    """Digits only - '+91 86107-90485' and '919876543210@c.us' both collapse
    to '919876543210' so config, bridge and WhatsApp JIDs always compare."""
    return re.sub(r"[^0-9]", "", str(raw or ""))


@dataclass
class Profile:
    id: str
    number: str          # normalized digits
    persona_id: str
    session_id: str
    db_path: Path
    collection: str
    is_default: bool = False

    # Live components, wired by build_profiles() in main.py.
    db: Any = None
    persona: Any = None
    memory: Any = None
    relationship: Any = None
    daylife: Any = None
    tools: Any = None
    tool_ctx: Any = None
    proactive: Any = None

    def __repr__(self) -> str:  # keep logs readable
        return (f"<Profile {self.id} persona={self.persona_id} "
                f"number=…{self.number[-4:]} db={self.db_path.name}>")


@dataclass
class ProfileRegistry:
    """All profiles, resolvable by number or id."""
    profiles: List[Profile] = field(default_factory=list)

    def by_number(self, number: str) -> Optional[Profile]:
        n = normalize_number(number)
        if not n:
            return None
        for p in self.profiles:
            # Compare on suffix: WhatsApp may hand back a number with or
            # without the country code depending on how it was saved.
            if p.number == n or n.endswith(p.number) or p.number.endswith(n):
                return p
        return None

    def by_id(self, profile_id: str) -> Optional[Profile]:
        return next((p for p in self.profiles if p.id == profile_id), None)

    def by_session(self, session_id: str) -> Optional[Profile]:
        return next((p for p in self.profiles if p.session_id == session_id), None)

    @property
    def default(self) -> Profile:
        return next((p for p in self.profiles if p.is_default), self.profiles[0])

    def __iter__(self):
        return iter(self.profiles)

    def __len__(self) -> int:
        return len(self.profiles)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(text).lower()).strip("_") or "profile"


def load_profiles(settings) -> ProfileRegistry:
    """Build the profile list from config. Never raises on a bad entry - a
    malformed profile is skipped with a warning rather than taking the whole
    app down (she must always come up for at least the default person)."""
    raw: List[Dict[str, Any]] = list(settings.profiles or [])
    out: List[Profile] = []

    if not raw:
        # No profiles: block -> legacy single-persona mode, unchanged.
        out.append(Profile(
            id="main",
            number="",  # empty -> matches nothing explicitly; used as default
            persona_id=settings.persona_active,
            session_id=settings.wa_session_id,
            db_path=settings.db_path,
            collection=settings.memory_collection,
            is_default=True,
        ))
        return ProfileRegistry(out)

    seen_ids: set[str] = set()
    seen_numbers: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            logger.warning("profiles: skipping non-dict entry %r", entry)
            continue
        pid = _slug(entry.get("id") or entry.get("persona") or "profile")
        number = normalize_number(entry.get("number", ""))
        persona_id = str(entry.get("persona") or settings.persona_active)
        if not number:
            logger.warning("profiles: %r has no number - skipped", pid)
            continue
        if pid in seen_ids:
            logger.warning("profiles: duplicate id %r - skipped", pid)
            continue
        if number in seen_numbers:
            logger.warning("profiles: duplicate number for %r - skipped", pid)
            continue
        seen_ids.add(pid)
        seen_numbers.add(number)

        is_default = bool(entry.get("default")) or not out
        # The default profile keeps the ORIGINAL db/collection so the existing
        # history and memories carry over untouched; others get their own.
        if is_default:
            db_path = settings.db_path
            collection = settings.memory_collection
            session_id = str(entry.get("session_id") or settings.wa_session_id)
        else:
            db_path = settings.db_path.with_name(
                f"{settings.db_path.stem}_{pid}{settings.db_path.suffix}")
            collection = f"{settings.memory_collection}_{pid}"
            session_id = str(entry.get("session_id") or pid)

        out.append(Profile(
            id=pid, number=number, persona_id=persona_id, session_id=session_id,
            db_path=Path(db_path), collection=collection, is_default=is_default,
        ))

    if not out:
        logger.error("profiles: no valid entries - falling back to legacy single profile")
        return load_profiles_legacy(settings)

    # Exactly one default.
    if not any(p.is_default for p in out):
        out[0].is_default = True
    return ProfileRegistry(out)


def load_profiles_legacy(settings) -> ProfileRegistry:
    return ProfileRegistry([Profile(
        id="main", number="", persona_id=settings.persona_active,
        session_id=settings.wa_session_id, db_path=settings.db_path,
        collection=settings.memory_collection, is_default=True,
    )])
