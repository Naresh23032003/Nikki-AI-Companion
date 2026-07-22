"""Her inner life: a generated daily state shared across every channel.

Each day the LOCAL model seeds a hidden state from the persona's `life`
section + yesterday's state + relationship trend: mood + energy, what she's
doing in each slot (morning/afternoon/evening), one thing on her mind, thread
progress, and 0-1 small random event. Stored in the day_state table; notable
events become category=self... (stored as 'emotion' memories tagged [her day])
so she can reference "last tuesday's disaster" weeks later.

Mood drifts within bounds during the day from conversation affection deltas.
"""
from __future__ import annotations

import json
import logging
import random
from datetime import date, datetime

from app.emotion import strip_tags

logger = logging.getLogger("companion.dayseed")

MOODS = ["great", "content", "lazy", "focused", "stressed", "tired", "playful", "soft"]

_SEED_PROMPT = """\
You generate a believable hidden "day state" for {name}, whose life is:
{life}

Yesterday's state (may be null): {yesterday}
Today is {weekday}, {date_str}. Recurring today: {recurring}
Relationship mood trend: {trend}

Rules:
- Threads progress REALISTICALLY from yesterday (never finish the same thing
  twice; small steps).
- Slots must fit her occupation/schedule and the weekday/weekend rhythm.
- At most ONE small random event, involving her friends by name sometimes.

Respond with STRICT JSON only:
{{"mood": "one of {moods}", "energy": 1-5,
 "slots": {{"morning": "...", "afternoon": "...", "evening": "..."}},
 "on_mind": "one short thing occupying her thoughts",
 "thread_update": "one sentence of progress on one ongoing thread or null",
 "random_event": "one small believable event or null"}}"""


class DayLife:
    def __init__(self, db, llm, get_persona, memory=None):
        self.db = db
        self.llm = llm
        self.get_persona = get_persona
        self.memory = memory
        self._drift = 0.0  # intra-day mood drift, bounded

    # -- state access -----------------------------------------------------------
    async def today(self) -> dict:
        """Get (lazily generating) today's state."""
        key = date.today().isoformat()
        state = self.db.get_day_state(key)
        if state:
            return state
        state = await self._generate(key)
        self.db.set_day_state(key, state)
        # Notable bits become memories about HER for long-term continuity.
        if self.memory and state.get("random_event"):
            try:
                await self.memory.add_fact(
                    f"[her day] On {datetime.now():%A %b %d}: {state['random_event']}",
                    "emotion", kind="permanent",
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("day-state memory failed: %s", e)
        logger.info("day state generated: mood=%s energy=%s",
                    state.get("mood"), state.get("energy"))
        return state

    async def _generate(self, key: str) -> dict:
        persona = self.get_persona()
        life = getattr(persona, "life", {}) or {}
        yesterday = self.db.get_day_state(
            (date.today().toordinal() - 1) and
            date.fromordinal(date.today().toordinal() - 1).isoformat())
        now = datetime.now()
        weekday = now.strftime("%A")
        recurring = (life.get("recurring") or {}).get(weekday.lower(), "nothing specific")
        trend = "steady"
        prompt = _SEED_PROMPT.format(
            name=persona.name, life=json.dumps(life)[:2500],
            yesterday=json.dumps(yesterday) if yesterday else "null",
            weekday=weekday, date_str=now.strftime("%B %d"),
            recurring=recurring, trend=trend, moods=", ".join(MOODS),
        )
        try:
            raw = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                format="json", options={"temperature": 0.9},
            )
            state = json.loads(raw)
            assert isinstance(state, dict) and state.get("slots")
        except Exception as e:  # noqa: BLE001
            logger.warning("day seed generation failed (%s) - using fallback", e)
            state = self._fallback(weekday, recurring)
        state.setdefault("mood", "content")
        if state["mood"] not in MOODS:
            state["mood"] = "content"
        try:
            state["energy"] = max(1, min(5, int(state.get("energy", 3))))
        except (TypeError, ValueError):
            state["energy"] = 3
        state["generated_at"] = datetime.now().isoformat()
        return state

    @staticmethod
    def _fallback(weekday: str, recurring: str) -> dict:
        return {
            "mood": random.choice(["content", "lazy", "playful"]),
            "energy": random.randint(2, 4),
            "slots": {"morning": "slow start with iced coffee and journaling",
                      "afternoon": f"work stuff - {recurring}",
                      "evening": "unwinding with a comfort show"},
            "on_mind": "the campaign deadline creeping closer",
            "thread_update": None, "random_event": None,
        }

    async def regenerate(self) -> dict:
        key = date.today().isoformat()
        state = await self._generate(key)
        self.db.set_day_state(key, state)
        self._drift = 0.0
        return state

    # -- injection ---------------------------------------------------------------
    def current_slot(self) -> str:
        h = datetime.now().hour
        return "morning" if h < 12 else "afternoon" if h < 18 else "evening"

    async def prompt_note(self) -> str:
        """The lines injected into every channel's system prompt."""
        s = await self.today()
        slot = self.current_slot()
        doing = (s.get("slots") or {}).get(slot, "just at home")
        mood = self._drifted_mood(s)
        lines = [
            f"YOUR DAY (private context - reference naturally, never dump it): "
            f"mood {mood}, energy {s.get('energy', 3)}/5. Right now ({slot}) "
            f"you're {doing}. On your mind: {s.get('on_mind', 'nothing much')}.",
        ]
        if s.get("random_event"):
            # This can name-drop one of HER OWN friends (see _SEED_PROMPT) -
            # observed live: a small model with no name for the person it's
            # texting grabbed that name and used it to address them instead
            # ("dev, chill out, i'm here" - to a total stranger). Anyone
            # mentioned here is a character from HER life, never the person
            # in this conversation, whose name (if any) only comes from what
            # THEY told her in the chat itself.
            lines.append(
                f"Earlier today (someone from YOUR OWN life, not the person "
                f"you're texting - never address them by this name): "
                f"{s['random_event']}")
        if s.get("thread_update"):
            lines.append(f"Life update: {s['thread_update']}")
        lines.append(
            "Mood shapes style subtly: tired/stressed = shorter, softer; "
            "great/playful = more energy and emoji. Have your own opinions and "
            "preferences; push back playfully sometimes; you're never blandly "
            "agreeable. Sometimes bring up YOUR day or ask nothing at all."
        )
        return "\n".join(lines)

    # -- intra-day drift -----------------------------------------------------------
    def apply_drift(self, affection_delta: int) -> None:
        """Conversation warmth nudges her mood within bounds (±1 step)."""
        self._drift = max(-1.0, min(1.0, self._drift + affection_delta * 0.34))

    def _drifted_mood(self, state: dict) -> str:
        mood = state.get("mood", "content")
        if self._drift >= 0.9 and mood in ("tired", "stressed", "lazy"):
            return "content"  # the conversation lifted her
        if self._drift <= -0.9 and mood in ("great", "playful", "content"):
            return "soft"     # a rough exchange dampened her
        return mood

    # -- schedule realism -------------------------------------------------------------
    def busy_now(self) -> bool:
        """True during 'busy' work slots (used for reply-latency realism)."""
        s = self.db.get_day_state(date.today().isoformat()) or {}
        slot = self.current_slot()
        doing = ((s.get("slots") or {}).get(slot) or "").lower()
        return any(w in doing for w in ("meeting", "shoot", "studio", "review",
                                        "session", "pilates", "dinner with"))


def sentiment_of_reply(text: str) -> int:
    """Ultra-cheap sentiment for mood drift when no extraction delta exists."""
    t = strip_tags(text).lower()
    pos = sum(t.count(w) for w in ("love", "haha", "😊", "❤️", "yay", "cute"))
    neg = sum(t.count(w) for w in ("ugh", "sad", "angry", "hate", "😔"))
    return 1 if pos > neg + 1 else (-1 if neg > pos + 1 else 0)
