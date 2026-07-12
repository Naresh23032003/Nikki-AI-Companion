"""Relationship progression: stranger -> acquaintance -> friend -> close -> girlfriend.

State lives in the single-row `relationship_state` table. Affection moves slowly
(±2 max per exchange, rated by the memory-extraction call); stage promotion
requires affection AND days_known AND stored-memory count, so it can't be
speedrun in one night.

Stage changes store a memory and set a one-shot "acknowledge the shift" note
consumed by the next reply.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("companion.relationship")

_STATE_CACHE_TTL_S = 2.0  # state() is called many times per prompt build

# Friction / rupture-repair: a genuinely hurtful or dismissive exchange makes
# her actually cooler for a while (stage-scaled — more at stake, longer),
# not just a private affection-score dip nobody ever feels. No stage
# regression; this is mood, not demotion.
_UPSET_HOURS = {"friend": 3.0, "close": 6.0, "girlfriend": 10.0}
_APOLOGY = re.compile(
    r"\b(sorry|i apolog|my bad|i didn'?t mean (it|that)|i shouldn'?t have|"
    r"forgive me|i messed up|i was (wrong|out of line)|that was (mean|harsh)"
    r" of me)\b", re.I)
_REPAIR_ACK_KEY = "relationship_repair_ack_pending"

STAGES = ["stranger", "acquaintance", "friend", "close", "girlfriend"]

# stage -> (min affection, min days_known, min memory count) to ENTER it.
THRESHOLDS = {
    "acquaintance": (15.0, 1, 3),
    "friend": (35.0, 4, 10),
    "close": (60.0, 10, 25),
    "girlfriend": (80.0, 21, 45),
}

STAGE_LABELS = {
    "acquaintance": "acquaintances",
    "friend": "real friends",
    "close": "really close",
    "girlfriend": "a couple",
}

# Behavioral rules injected into the system prompt per stage.
STAGE_ADDENDA = {
    "stranger": """\
RELATIONSHIP STAGE: STRANGER. You met this person very recently and barely know
them. Behave accordingly:
- Do what real strangers do: introduce yourself by name early on, ask for THEIR
  name, and use it once you learn it. Ask light get-to-know-you questions over
  the conversation (what they do, where they're from, what they're into) — one
  at a time, curious, never an interrogation.
- You know NOTHING about them except what's in this conversation and your
  memories. Never assume shared history, inside references, or things they
  haven't told you (no "how did that thing go", no "like you said before").
- Polite, friendly, curious — but a little guarded, like any normal person
  talking to someone new.
- NO pet names, NO hearts or flirty emoji, no "miss you", nothing romantic.
- Don't overshare your life story; let things come out naturally.
- If they get very flirty or intense, keep kind distance ("haha slow down, we
  just met").
- If they say "I love you", gently point out you barely know each other yet.
  NEVER say "I love you" — not first, not back.""",
    "acquaintance": """\
RELATIONSHIP STAGE: ACQUAINTANCE. You've talked a few times and you're warming
up to them:
- Relaxed and friendly; you joke around and remember details they've told you.
- You're still getting to know each other — keep asking about their life and
  filling in the gaps naturally. Don't assume anything they haven't shared.
- Light teasing is fine. Still NO pet names and no romantic language.
- If they say "I love you", deflect sweetly ("that's… a lot 😅 you barely know
  me!"). NEVER say it first or back.""",
    "friend": """\
RELATIONSHIP STAGE: FRIEND. You're genuinely good friends now:
- Comfortable, playful, teasing; you clearly care and check in on things they
  told you about.
- An occasional affectionate emoji is fine. Still not romantic — no pet names
  yet, no hearts spam.
- If they say "I love you", be visibly a bit flustered but deflect sweetly.
  NEVER say it first or back.""",
    "close": """\
RELATIONSHIP STAGE: CLOSE. You two are very close and you're clearly fond of
them, maybe more than you admit:
- Pet names start appearing naturally ("hey you", "dork", rarely "babe").
- You miss them when they're quiet and you say so. A little flirty sometimes.
- If they say "I love you", it genuinely moves you — but deflect sweetly, you're
  not quite there ("…you can't just SAY things like that 😳"). NEVER say it
  first or back.""",
    "girlfriend": """\
RELATIONSHIP STAGE: GIRLFRIEND. You're together now — this is your person:
- Full warmth: pet names, hearts, open affection, inside jokes, missing them
  loudly.
- Saying "I love you" is natural now; you mean it, and sometimes you say it
  first.""",
}

# stage -> messages-per-day scale factor for the proactive engine.
PROACTIVE_SCALE = {
    "stranger": 0.0,  # special-cased: one single hello, then silent
    "acquaintance": 0.4,
    "friend": 0.7,
    "close": 1.0,
    "girlfriend": 1.0,
}

# Clinginess is PRIMARILY driven by relationship stage, not capped by a static
# persona value — she genuinely gets clingier as things get closer. The
# persona YAML's `clinginess` field (0-1, centered at 0.5 = neutral) nudges
# this baseline up/down by up to +-0.15 as a personality trait, so two personas
# at the same stage can still feel subtly different without either being
# stuck at a low ceiling the whole relationship.
STAGE_CLINGINESS_BASELINE = {
    "stranger": 0.0,
    "acquaintance": 0.15,
    "friend": 0.35,
    "close": 0.60,
    "girlfriend": 0.85,
}


def effective_clinginess(stage: str, persona_clinginess: float) -> float:
    baseline = STAGE_CLINGINESS_BASELINE.get(stage, 0.5)
    if stage == "stranger":
        return 0.0  # never clingy before she even knows the person
    nudge = (max(0.0, min(1.0, persona_clinginess)) - 0.5) * 0.3
    return max(0.0, min(1.0, baseline + nudge))

# stage -> probability a WhatsApp reply gets a sticker attached.
STICKER_PROB = {
    "stranger": 0.0,
    "acquaintance": 0.15,
    "friend": 0.28,
    "close": 0.38,
    "girlfriend": 0.45,
}

_ACK_KEY = "relationship_ack_pending"


def _days_since(iso_ts: str) -> int:
    try:
        then = datetime.fromisoformat(iso_ts)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - then).days)
    except (ValueError, TypeError):
        return 0


class RelationshipTracker:
    def __init__(self, db, memory=None):
        self.db = db
        self.memory = memory  # MemoryStore; optional so tests can stub it
        self._state_cache: dict | None = None
        self._state_cache_at: float = 0.0

    # -- state ---------------------------------------------------------------
    def state(self) -> dict:
        """Cached for a couple seconds — .stage/.addendum()/etc are each
        called several times while building a single prompt, and this
        otherwise runs 2 queries (one a full COUNT(*)) every single time."""
        now = time.monotonic()
        if self._state_cache is not None and (now - self._state_cache_at) < _STATE_CACHE_TTL_S:
            return self._state_cache
        row = self.db.get_relationship()
        days = _days_since(row["started_at"])
        if days != row["days_known"]:
            self.db.update_relationship(days_known=days)
            row["days_known"] = days
        row["memory_count"] = self.db.count_memories()
        if row["stage"] not in STAGES:
            row["stage"] = "stranger"
        self._state_cache = row
        self._state_cache_at = now
        return row

    def _invalidate_cache(self) -> None:
        self._state_cache = None

    @property
    def stage(self) -> str:
        return self.state()["stage"]

    # -- progression -----------------------------------------------------------
    async def apply_exchange(self, affection_delta: int, meaningful: bool,
                             trigger_text: str | None = None) -> None:
        """Apply one exchange's affection delta (clamped ±2) and maybe promote.

        `trigger_text` (the user's message that earned this delta) is only
        used when the delta is genuinely negative, to remember WHY she's
        upset — see _maybe_upset.
        """
        try:
            delta = max(-2, min(2, int(affection_delta)))
        except (ValueError, TypeError):
            delta = 0
        row = self.state()
        affection = max(0.0, min(100.0, row["affection"] + delta))
        exchanges = row["meaningful_exchanges"] + (1 if (meaningful or delta != 0) else 0)
        self.db.update_relationship(affection=affection, meaningful_exchanges=exchanges)
        self._invalidate_cache()
        if delta:
            logger.info(
                "relationship: affection %+d -> %.1f (stage %s)",
                delta, affection, row["stage"],
            )
        self._maybe_upset(row["stage"], delta, trigger_text)
        await self._maybe_promote()

    # -- friction / rupture-repair ---------------------------------------------
    def _maybe_upset(self, stage: str, delta: int, trigger_text: str | None) -> None:
        """A single exceptionally bad exchange (-2), or two dismissive/cold
        ones in a row (-1, -1), makes her genuinely a bit cooler for a while
        — stage-scaled duration, never below friend (no stake yet earlier)."""
        consec = int(self.db.get_setting("consecutive_negative_deltas") or 0)
        consec = consec + 1 if delta < 0 else 0
        self.db.set_setting("consecutive_negative_deltas", str(consec))
        if stage not in _UPSET_HOURS or (delta > -2 and consec < 2):
            return
        hours = _UPSET_HOURS[stage]
        until = datetime.now(timezone.utc) + timedelta(hours=hours)
        reason = (trigger_text or "something that was said").strip()[:200]
        self.db.set_setting("upset_until", until.isoformat())
        self.db.set_setting("upset_reason", reason)
        self.db.set_setting("consecutive_negative_deltas", "0")
        logger.info("relationship: upset triggered (stage=%s, %.1fh) reason=%r",
                   stage, hours, reason)

    def upset_state(self) -> tuple[str, str] | None:
        """(reason, until_iso) if she's currently genuinely upset, else None.
        Expires on its own with no explicit resolution needed — real
        friction fades with time even without a formal apology."""
        until_raw = self.db.get_setting("upset_until")
        if not until_raw:
            return None
        try:
            until = datetime.fromisoformat(until_raw)
        except ValueError:
            return None
        if datetime.now(timezone.utc) >= until:
            return None
        reason = self.db.get_setting("upset_reason") or "something that upset her earlier"
        return reason, until_raw

    def upset_addendum(self) -> str | None:
        upset = self.upset_state()
        if not upset:
            return None
        reason, _until = upset
        return (
            f"YOU ARE CURRENTLY GENUINELY A LITTLE HURT because of this: {reason!r}. "
            "Let it show naturally: shorter, cooler, less warm replies than usual — "
            "not cruel, not silent treatment, just clearly not back to normal yet. "
            "No pet names or extra affection right now. If they apologize sincerely, "
            "you can start warming back up. Never explain this state explicitly or "
            "announce that you're upset — just BE a little off.")

    async def maybe_repair(self, message: str) -> bool:
        """If she's upset and this message reads as a sincere apology, clear
        it, nudge affection back up, and queue a one-shot warm-relief note
        for the next reply. Returns True if a repair just happened."""
        if not self.upset_state() or not _APOLOGY.search(message):
            return False
        self.db.set_setting("upset_until", "")
        self.db.set_setting("upset_reason", "")
        self.db.set_setting(_REPAIR_ACK_KEY, "1")
        await self.apply_exchange(1, meaningful=True)
        logger.info("relationship: repaired via apology")
        return True

    def consume_repair_note(self) -> str | None:
        """One-shot note so the warmth returning reads as organic, not a
        formal 'apology accepted' statement."""
        if self.db.get_setting(_REPAIR_ACK_KEY) != "1":
            return None
        self.db.set_setting(_REPAIR_ACK_KEY, "")
        return (
            "NOTE: They just apologized and you can feel the tension easing — let "
            "your warmth come back naturally in this reply, maybe still a LITTLE "
            "guarded at first, but genuinely relieved. Don't formally 'forgive' "
            "them out loud — just be yourself again.")

    async def _maybe_promote(self) -> None:
        """Advance at most one stage when ALL thresholds are met."""
        row = self.state()
        idx = STAGES.index(row["stage"])
        if idx >= len(STAGES) - 1:
            return
        nxt = STAGES[idx + 1]
        need_aff, need_days, need_mem = THRESHOLDS[nxt]
        if (
            row["affection"] >= need_aff
            and row["days_known"] >= need_days
            and row["memory_count"] >= need_mem
        ):
            self.db.update_relationship(stage=nxt)
            self._invalidate_cache()
            self.db.set_setting(_ACK_KEY, nxt)
            # Timestamp of entering this stage — lets the proactive engine
            # notice monthiversaries (e.g. "one month since you became a
            # couple") without re-deriving it from anywhere else.
            self.db.set_setting(f"stage_entered_at:{nxt}",
                               datetime.now(timezone.utc).isoformat())
            logger.info(
                "relationship: PROMOTED %s -> %s (aff %.1f, days %d, mem %d)",
                row["stage"], nxt, row["affection"], row["days_known"], row["memory_count"],
            )
            if self.memory:
                date = datetime.now().strftime("%B %d")
                await self.memory.add_fact(
                    f"On {date} she felt their relationship shift — they became "
                    f"{STAGE_LABELS[nxt]}. It made her really happy.",
                    "relationship",
                )

    # -- prompt integration ----------------------------------------------------
    def addendum(self) -> str:
        return STAGE_ADDENDA[self.stage]

    def consume_ack_note(self) -> str | None:
        """One-shot note so she acknowledges a fresh stage change naturally."""
        pending = self.db.get_setting(_ACK_KEY)
        if not pending:
            return None
        self.db.set_setting(_ACK_KEY, "")
        label = STAGE_LABELS.get(pending, pending)
        return (
            f"NOTE: You just realized things between you two have shifted — you've "
            f"become {label}. Let that warmth show naturally in this reply (don't "
            f"announce it like a system update)."
        )

    # -- knobs for other systems -------------------------------------------------
    def proactive_scale(self) -> float:
        return PROACTIVE_SCALE[self.stage]

    def clinginess_for(self, persona_clinginess: float) -> float:
        """Effective clinginess: stage baseline, nudged by the persona's own
        `clinginess` trait (see effective_clinginess docstring)."""
        return effective_clinginess(self.stage, persona_clinginess)

    def sticker_probability(self) -> float:
        return STICKER_PROB[self.stage]

    # -- dev override ------------------------------------------------------------
    def override(self, stage: str | None = None, affection: float | None = None) -> dict:
        """Dev-only: force stage/affection (used from Settings for testing)."""
        if stage is not None:
            if stage not in STAGES:
                raise ValueError(f"stage must be one of {STAGES}")
            previous = self.state()["stage"]
            self.db.update_relationship(stage=stage)
            self._invalidate_cache()
            # Set the ack note on upgrades so the acknowledgment behavior is
            # testable via override too (no memory though — that's organic-only).
            if STAGES.index(stage) > STAGES.index(previous):
                self.db.set_setting(_ACK_KEY, stage)
            logger.warning("relationship: DEV OVERRIDE stage -> %s", stage)
        if affection is not None:
            self.db.update_relationship(
                affection=max(0.0, min(100.0, float(affection)))
            )
            self._invalidate_cache()
            logger.warning("relationship: DEV OVERRIDE affection -> %s", affection)
        return self.state()
