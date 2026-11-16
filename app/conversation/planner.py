"""Turn planner — decides the SHAPE of Nikki's reply before it is generated (N3).

The problem this solves: the previous conversation logic was ~15 heuristic
builders that each appended an instruction string to the system prompt. Every
nudge competed with every other nudge inside one blob of English, and the model
resolved the conflicts however it felt. That is why replies felt subtly off in
a way no single prompt edit ever fixed — there was no decision, only
accumulated suggestion.

`plan_turn` makes the decision explicit and inspectable *before* generation.
`render` emits ONE coherent directive instead of fifteen competing ones. Every
plan carries `reasons`, so a bad turn can be diagnosed instead of guessed at.

All randomness goes through an injected RNG so every branch is reachable under
test — the old note builders called `random.random()` directly, which is a
large part of why none of them had tests.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Sequence

from app.conversation.tells import question_streak

# Reply length bands, smallest first.
MINIMAL = "minimal"
BRIEF = "brief"
NORMAL = "normal"
EXTENDED = "extended"

# Cooldowns, in turns.
SELF_SHARE_COOLDOWN = 4
MEMORY_COOLDOWN = 3

# Stages at which she does not push back yet.
RESERVED_STAGES = ("stranger", "acquaintance")

# Probability of asking during distress. Low, not zero: sometimes the right
# thing is one gentle question, but the default is to let them talk.
DISTRESS_ASK_CHANCE = 0.25

_LOW_ENERGY = {"ok", "okay", "k", "kk", "yeah", "yep", "yup", "nah", "no",
               "lol", "lmao", "haha", "mhm", "mm", "hm", "nice", "cool",
               "sure", "fine", "night", "ty", "thx"}

_DISTRESS = re.compile(
    r"\b(exhaust\w*|overwhelm\w*|depress\w*|anxious|anxiety|panic|scared|"
    r"terrified|can'?t cope|can'?t do this|breaking down|crying|cried|"
    r"hopeless|awful|terrible|worst|hate my|so tired of|burnt? out|"
    r"died|death|funeral|hospital|cancer|fired|laid off|broke up|divorce)\b",
    re.I)

_QUESTION_START = re.compile(
    r"^\s*(what|why|how|when|where|who|which|do|does|did|are|is|was|were|"
    r"can|could|would|should|will|have|has|had)\b", re.I)


@dataclass
class TurnSignals:
    user_message: str = ""
    recent_replies: Sequence[str] = field(default_factory=list)
    stage: str = "close"
    relevant_memory: str | None = None
    turns_since_self_share: int = 99
    turns_since_memory_surfaced: int = 99
    is_reconnect: bool = False
    rng: random.Random = field(default_factory=random.Random)

    @property
    def words(self) -> List[str]:
        return self.user_message.split()

    @property
    def is_low_energy(self) -> bool:
        stripped = re.sub(r"[^a-z\s]", "", self.user_message.lower()).strip()
        return bool(stripped) and all(w in _LOW_ENERGY for w in stripped.split())

    @property
    def is_question(self) -> bool:
        text = self.user_message.strip()
        return text.endswith("?") or bool(_QUESTION_START.match(text))

    @property
    def is_distressed(self) -> bool:
        return bool(_DISTRESS.search(self.user_message))


@dataclass
class TurnPlan:
    length: str = NORMAL
    ask_question: bool = False
    share_self: bool = False
    surface_memory: str | None = None
    allow_disagreement: bool = False
    reasons: List[str] = field(default_factory=list)


def plan_turn(signals: TurnSignals) -> TurnPlan:
    """Decide the shape of this reply.

    Length is decided first because it constrains everything else — you cannot
    ask a question, share news and surface a memory inside a two-word reply.
    """
    plan = TurnPlan()
    r = plan.reasons

    # --- length: mirror their energy ------------------------------------
    n = len(signals.words)
    if signals.is_low_energy:
        plan.length = MINIMAL
        r.append("low-energy message - match it with a few words")
    elif signals.is_distressed:
        plan.length = NORMAL
        r.append("they're struggling - room to respond properly, not a wall")
    elif n >= 40:
        plan.length = EXTENDED
        r.append(f"they wrote {n} words - meet them there")
    elif n <= 4:
        plan.length = BRIEF
        r.append("short message - keep it brief")
    else:
        plan.length = NORMAL

    minimal = plan.length == MINIMAL

    # --- asking ---------------------------------------------------------
    # The single biggest AI tell is a question every turn.
    streak = question_streak(signals.recent_replies)
    if minimal:
        plan.ask_question = False
        r.append("minimal turn - no room for a question")
    elif streak >= 2:
        plan.ask_question = False
        r.append(f"asked a question {streak} turns running - stop questioning, let it breathe")
    elif signals.is_question:
        plan.ask_question = False
        r.append("they asked something - answer it rather than deflecting")
    elif signals.is_distressed:
        plan.ask_question = signals.rng.random() < DISTRESS_ASK_CHANCE
        r.append("distress - usually listen rather than question")
    else:
        plan.ask_question = True
        r.append("ordinary open turn - a question is welcome")

    # --- self-disclosure -------------------------------------------------
    if minimal:
        plan.share_self = False
    elif signals.is_reconnect and signals.turns_since_self_share >= SELF_SHARE_COOLDOWN:
        plan.share_self = True
        r.append("reconnecting - natural moment to say what she's been up to")
    elif signals.turns_since_self_share < SELF_SHARE_COOLDOWN:
        plan.share_self = False
        r.append("just shared - don't make it about her again")
    else:
        plan.share_self = True
        r.append("hasn't offered anything of her own in a while")

    # --- memory ----------------------------------------------------------
    if signals.relevant_memory and not minimal:
        if signals.turns_since_memory_surfaced < MEMORY_COOLDOWN:
            plan.surface_memory = None
            r.append("memory suppressed - surfaced one too recently to do it again")
        else:
            plan.surface_memory = signals.relevant_memory
            r.append("a relevant memory is worth bringing in")
    elif signals.relevant_memory and minimal:
        r.append("memory suppressed - minimal turn")

    # --- disagreement -----------------------------------------------------
    plan.allow_disagreement = (
        signals.stage not in RESERVED_STAGES and not signals.is_distressed)
    if plan.allow_disagreement:
        r.append("close enough to disagree if she actually does")
    elif signals.is_distressed:
        r.append("not the moment to push back")

    return plan


def render(plan: TurnPlan, extra_notes: Sequence[str] | None = None) -> str:
    """One coherent directive — the whole point of the planner.

    Kept to a handful of lines. Fifteen competing instructions is what produced
    the original problem; replacing them with fifteen better-worded ones would
    not have helped.
    """
    length_line = {
        MINIMAL: "Reply in a few words at most.",
        BRIEF: "Reply briefly - one short line.",
        NORMAL: "Reply normally - a line or two.",
        EXTENDED: "Reply at length - they wrote a lot, meet them there.",
    }[plan.length]

    lines = [length_line]
    lines.append("Ask them one thing." if plan.ask_question
                 else "Do NOT ask a question this turn.")
    if plan.share_self:
        lines.append("Offer something of your own - what you've been doing or thinking.")
    if plan.surface_memory:
        lines.append(
            f"You remember: {plan.surface_memory}. Let it colour the reply "
            "naturally - do NOT recite it back at them.")
    if not plan.allow_disagreement:
        lines.append("Don't argue or correct them right now.")
    elif plan.allow_disagreement:
        lines.append("If you see it differently, say so plainly.")

    for note in extra_notes or []:
        lines.append(note)
    return "\n".join(lines)


def summarise(plan: TurnPlan) -> str:
    """Compact, loggable form — makes a bad turn diagnosable after the fact."""
    flags = []
    if plan.ask_question:
        flags.append("ask")
    if plan.share_self:
        flags.append("share")
    if plan.surface_memory:
        flags.append("memory")
    if plan.allow_disagreement:
        flags.append("may-disagree")
    return f"turn_plan[{plan.length}{(' ' + ','.join(flags)) if flags else ''}]"
