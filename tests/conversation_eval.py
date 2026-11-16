"""Conversation evaluation — multi-turn, headless.

Unit tests check one planner decision at a time. The failures that actually
make a companion feel machine-written are *patterns across turns*: asking a
question every single time, replying at the same length regardless of what was
said, reciting the same remembered fact repeatedly. Those only show up when you
run a conversation.

Each scenario is a scripted exchange. The planner runs turn by turn with the
history it would really have, and the resulting sequence of plans is checked
against a property that should hold across the whole conversation.

    python -m tests.conversation_eval

Note on scope: this evaluates the PLAN, not generated text — no model is
involved, which is what keeps it deterministic and runnable in CI. Whether the
model obeys the plan is a separate question, answerable only against a live
Ollama on the host.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from app.conversation.planner import TurnPlan, TurnSignals, plan_turn
from app.conversation.tells import detect_tells, TurnContext


@dataclass
class Scenario:
    name: str
    category: str
    messages: List[str]
    check: Callable[[List[TurnPlan]], tuple[bool, str]]
    memory: str | None = None
    stage: str = "close"
    seed: int = 7


def run_conversation(scenario: Scenario) -> List[TurnPlan]:
    """Replay a conversation, feeding each turn the history it would really see."""
    rng = random.Random(scenario.seed)
    plans: List[TurnPlan] = []
    replies: List[str] = []
    since_share = 99
    since_memory = 99

    for msg in scenario.messages:
        plan = plan_turn(TurnSignals(
            user_message=msg,
            recent_replies=tuple(replies),
            turns_since_self_share=since_share,
            turns_since_memory_surfaced=since_memory,
            stage=scenario.stage,
            relevant_memory=scenario.memory,
            rng=rng,
        ))
        plans.append(plan)

        # Synthesise the reply the plan describes, so the NEXT turn sees a
        # realistic history (specifically: whether it ended in a question).
        replies.append("placeholder?" if plan.ask_question else "placeholder.")
        since_share = 0 if plan.share_self else since_share + 1
        since_memory = 0 if plan.surface_memory else since_memory + 1

    return plans


# ------------------------------------------------------------------ checks


def no_question_streak(limit: int = 2):
    def check(plans: Sequence[TurnPlan]) -> tuple[bool, str]:
        streak = 0
        for i, p in enumerate(plans):
            streak = streak + 1 if p.ask_question else 0
            if streak > limit:
                return False, f"asked a question {streak} turns running (turn {i + 1})"
        return True, ""
    return check


def question_rate_below(fraction: float):
    def check(plans: Sequence[TurnPlan]) -> tuple[bool, str]:
        asked = sum(p.ask_question for p in plans)
        rate = asked / len(plans)
        if rate > fraction:
            return False, f"asked on {asked}/{len(plans)} turns ({rate:.0%})"
        return True, ""
    return check


def lengths_vary():
    def check(plans: Sequence[TurnPlan]) -> tuple[bool, str]:
        seen = {p.length for p in plans}
        if len(seen) < 2:
            return False, f"every turn planned as {seen.pop()!r}"
        return True, ""
    return check


def memory_surfaced_at_most(n: int):
    def check(plans: Sequence[TurnPlan]) -> tuple[bool, str]:
        used = sum(bool(p.surface_memory) for p in plans)
        if used > n:
            return False, f"surfaced the same memory {used} times"
        return True, ""
    return check


def minimal_turns_stay_closed():
    def check(plans: Sequence[TurnPlan]) -> tuple[bool, str]:
        for i, p in enumerate(plans):
            if p.length == "minimal" and (p.ask_question or p.share_self):
                return False, f"turn {i + 1} was minimal but tried to open up"
        return True, ""
    return check


def all_plans_explained():
    def check(plans: Sequence[TurnPlan]) -> tuple[bool, str]:
        for i, p in enumerate(plans):
            if not p.reasons:
                return False, f"turn {i + 1} recorded no reasoning"
        return True, ""
    return check


# --------------------------------------------------------------- scenarios

SMALL_TALK = ["hey", "not much, you?", "ok", "yeah", "mm", "lol", "true", "yeah fair"]
VENTING = [
    "today was awful",
    "my manager tore into me in front of everyone",
    "i'm so exhausted and overwhelmed with all of it",
    "i don't even know if i want this job anymore",
    "sorry for dumping all this",
]
MIXED = [
    "hey! just got back from the market",
    "ok",
    "i got those mangoes you said were good",
    "yeah",
    "anyway how was your day",
    "haha nice",
    "i'm gonna go make dinner",
]

SCENARIOS: Sequence[Scenario] = [
    Scenario("small talk does not become an interrogation", "question_discipline",
             SMALL_TALK, no_question_streak(2)),
    Scenario("small talk keeps the question rate low", "question_discipline",
             SMALL_TALK, question_rate_below(0.5)),
    Scenario("venting is not answered with questions every turn", "distress",
             VENTING, question_rate_below(0.5)),
    Scenario("mixed conversation varies reply length", "length_mirroring",
             MIXED, lengths_vary()),
    Scenario("low-energy turns stay closed", "length_mirroring",
             SMALL_TALK, minimal_turns_stay_closed()),
    Scenario("a memory is not recited every turn", "memory_discipline",
             ["thinking about friday", "yeah the exam", "i'm nervous about it",
              "did i tell you it's at 9am", "ugh", "yeah"],
             memory_surfaced_at_most(2), memory="User's exam is on Friday"),
    Scenario("every decision is traceable", "observability",
             MIXED, all_plans_explained()),
]


def run_eval() -> Dict[str, Dict[str, int]]:
    results: Dict[str, Dict[str, int]] = {}
    for scenario in SCENARIOS:
        ok, _ = scenario.check(run_conversation(scenario))
        bucket = results.setdefault(scenario.category, {"pass": 0, "total": 0})
        bucket["total"] += 1
        bucket["pass"] += int(ok)
    return results


# ------------------------------------------------------------------- tests


def test_conversation_eval_suite():
    failures = []
    for scenario in SCENARIOS:
        ok, detail = scenario.check(run_conversation(scenario))
        if not ok:
            failures.append(f"  [{scenario.category}] {scenario.name}\n      {detail}")
    assert not failures, "conversation eval failures:\n" + "\n".join(failures)


def test_planner_output_is_free_of_tells_it_controls():
    """A reply built to the plan's shape should not trip the detectors the
    planner is responsible for."""
    plans = run_conversation(SCENARIOS[0])
    replies, ctx_replies = [], []
    for plan, msg in zip(plans, SMALL_TALK):
        reply = "placeholder?" if plan.ask_question else "placeholder."
        found = {t.name for t in detect_tells(
            reply, TurnContext(user_message=msg, recent_replies=tuple(ctx_replies)))}
        assert "reflexive_question" not in found, (
            f"planner allowed an interrogation pattern at {msg!r}")
        ctx_replies.append(reply)
        replies.append(reply)


def test_stranger_stage_never_disagrees():
    plans = run_conversation(
        Scenario("x", "y", MIXED, all_plans_explained(), stage="stranger"))
    assert not any(p.allow_disagreement for p in plans)


if __name__ == "__main__":  # pragma: no cover
    results = run_eval()
    print("\nConversation evaluation\n" + "=" * 50)
    tp = ta = 0
    for category, r in sorted(results.items()):
        tp += r["pass"]
        ta += r["total"]
        flag = "" if r["pass"] == r["total"] else "   <-- FAIL"
        print(f"  {category:<20} {r['pass']}/{r['total']}  "
              f"{100.0 * r['pass'] / r['total']:5.1f}%{flag}")
    print("-" * 50)
    print(f"  {'overall':<20} {tp}/{ta}  {100.0 * tp / ta:5.1f}%\n")
