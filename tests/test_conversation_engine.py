"""Tests for the turn planner and AI-tell detector (N3).

The planner is deterministic under an injected RNG, so every branch is
reachable. The tell detector is pure.
"""
from __future__ import annotations

import random

import pytest

from app.conversation.planner import (
    TurnSignals,
    plan_turn,
    render,
    summarise,
)
from app.conversation.tells import (
    TurnContext,
    detect_tells,
    naturalness_score,
    question_streak,
    tell_names,
)


class FixedRandom(random.Random):
    def __init__(self, value: float):
        super().__init__(0)
        self._value = value

    def random(self):
        return self._value


def sig(msg="", **kw):
    kw.setdefault("rng", FixedRandom(0.0))  # 0.0 => probabilistic gates open
    return TurnSignals(user_message=msg, **kw)


# =============================================================== tell detector


class TestQuestionStreak:
    def test_counts_consecutive_question_endings(self):
        assert question_streak(["a?", "b?", "c?"]) == 3

    def test_stops_at_a_non_question(self):
        assert question_streak(["a?", "b.", "c?"]) == 1

    def test_empty_history(self):
        assert question_streak([]) == 0


class TestReflexiveQuestion:
    def test_third_question_in_a_row_is_a_tell(self):
        ctx = TurnContext(user_message="ok", recent_replies=["how was it?", "and then?"])
        assert "reflexive_question" in tell_names("what did you do after?", ctx)

    def test_first_question_is_fine(self):
        ctx = TurnContext(user_message="ok", recent_replies=["that's rough.", "mm."])
        assert "reflexive_question" not in tell_names("did you sleep at all?", ctx)


class TestLengthMismatch:
    def test_paragraph_in_reply_to_ok_is_a_tell(self):
        ctx = TurnContext(user_message="ok")
        reply = ("That's completely fine by me and I hope the rest of your "
                 "evening goes really smoothly for you tonight.")
        assert "length_mismatch" in tell_names(reply, ctx)

    def test_short_reply_to_short_message_is_fine(self):
        assert "length_mismatch" not in tell_names("mm ok", TurnContext(user_message="ok"))

    def test_long_reply_to_long_message_is_fine(self):
        user = " ".join(["word"] * 50)
        reply = " ".join(["reply"] * 40)
        assert "length_mismatch" not in tell_names(reply, TurnContext(user_message=user))


class TestOverAgreement:
    @pytest.mark.parametrize("opener", [
        "That's so valid, honestly.",
        "It makes total sense that you'd feel that way.",
        "I totally get that.",
        "You're so right.",
    ])
    def test_validation_openers_flagged(self, opener):
        assert "over_agreement" in tell_names(opener, TurnContext(user_message="i'm tired"))

    def test_ordinary_reply_not_flagged(self):
        assert "over_agreement" not in tell_names(
            "ugh that sounds like a long day", TurnContext(user_message="i'm tired"))


class TestGenericQuestion:
    @pytest.mark.parametrize("q", [
        "How does that make you feel?",
        "What's on your mind?",
        "Do you want to talk about it?",
        "Tell me more about that.",
    ])
    def test_content_free_questions_flagged(self, q):
        assert "generic_question" in tell_names(q, TurnContext(user_message="rough day"))

    def test_specific_question_not_flagged(self):
        assert "generic_question" not in tell_names(
            "did your manager actually apologise?", TurnContext(user_message="rough day"))


class TestOtherTells:
    def test_summarising_back(self):
        assert "summarising_back" in tell_names(
            "So you're saying the meeting went badly.", TurnContext(user_message="x"))

    def test_unearned_enthusiasm(self):
        assert "unearned_enthusiasm" in tell_names(
            "That's amazing!", TurnContext(user_message="i had toast"))

    def test_matched_enthusiasm_is_not_a_tell(self):
        assert "unearned_enthusiasm" not in tell_names(
            "that's amazing!", TurnContext(user_message="I GOT THE JOB!!"))

    def test_stacked_hedging(self):
        assert "stacked_hedging" in tell_names(
            "I think maybe it might be sort of fine.", TurnContext(user_message="x"))

    def test_symmetrical_rhythm(self):
        reply = ("I went to the shop today. I bought some bread there. "
                 "I walked back home after.")
        assert "symmetrical_rhythm" in tell_names(reply, TurnContext(user_message="x"))

    def test_varied_rhythm_is_not_flagged(self):
        reply = "Ha. I went to the shop and ended up buying way too much bread. Oops."
        assert "symmetrical_rhythm" not in tell_names(reply, TurnContext(user_message="x"))

    def test_repetition_across_replies(self):
        ctx = TurnContext(user_message="x",
                          recent_replies=["you should get some rest tonight"])
        assert "repetition" in tell_names("seriously, get some rest tonight ok", ctx)


class TestNaturalnessScore:
    def test_clean_reply_scores_one(self):
        assert naturalness_score("ha, fair enough", TurnContext(user_message="ok")) == 1.0

    def test_score_drops_with_tells(self):
        ctx = TurnContext(user_message="ok", recent_replies=["a?", "b?"])
        bad = "That's so valid! How does that make you feel?"
        assert naturalness_score(bad, ctx) < 0.7

    def test_empty_reply_has_no_tells(self):
        assert detect_tells("", TurnContext()) == []


# ================================================================== planner


class TestLengthMirroring:
    @pytest.mark.parametrize("msg", ["ok", "k", "yeah", "lol", "mhm", "nice"])
    def test_low_energy_gets_minimal(self, msg):
        assert plan_turn(sig(msg)).length == "minimal"

    def test_short_message_gets_brief(self):
        assert plan_turn(sig("just got home")).length == "brief"

    def test_long_message_gets_extended(self):
        assert plan_turn(sig(" ".join(["word"] * 50))).length == "extended"

    def test_ordinary_message_gets_normal(self):
        assert plan_turn(sig("i finally finished that report today")).length == "normal"


class TestQuestionSuppression:
    def test_never_asks_after_a_question_streak(self):
        """The headline anti-tell rule."""
        plan = plan_turn(sig("cool", recent_replies=["how was it?", "and then?"]))
        assert plan.ask_question is False
        assert any("question" in r for r in plan.reasons)

    def test_does_not_ask_on_a_minimal_turn(self):
        assert plan_turn(sig("ok")).ask_question is False

    def test_answers_rather_than_redirects_when_asked(self):
        plan = plan_turn(sig("what did you do today?"))
        assert plan.ask_question is False
        assert any("answer" in r for r in plan.reasons)

    def test_usually_avoids_questioning_during_distress(self):
        plan = plan_turn(sig("i'm so exhausted and overwhelmed with everything",
                             rng=FixedRandom(0.9)))
        assert plan.ask_question is False

    def test_can_ask_in_an_ordinary_open_turn(self):
        plan = plan_turn(sig("i finally finished that report today",
                             recent_replies=["nice.", "mm."]))
        assert plan.ask_question is True


class TestSelfSharing:
    def test_suppressed_right_after_sharing(self):
        assert plan_turn(sig("mm interesting", turns_since_self_share=1)).share_self is False

    def test_allowed_after_the_cooldown(self):
        assert plan_turn(sig("mm interesting", turns_since_self_share=9)).share_self is True

    def test_natural_on_reconnect(self):
        plan = plan_turn(sig("hey", is_reconnect=True, turns_since_self_share=9,
                             rng=FixedRandom(0.99)))
        assert plan.share_self is True

    def test_never_on_a_minimal_turn(self):
        assert plan_turn(sig("ok", turns_since_self_share=99)).share_self is False


class TestMemorySurfacing:
    def test_relevant_memory_surfaces_when_not_recently_used(self):
        plan = plan_turn(sig("thinking about the exam",
                             relevant_memory="User's exam is on Friday",
                             turns_since_memory_surfaced=9))
        assert plan.surface_memory == "User's exam is on Friday"

    def test_recently_used_memory_is_suppressed(self):
        """Prevents her reciting the same fact back turn after turn."""
        plan = plan_turn(sig("thinking about the exam",
                             relevant_memory="User's exam is on Friday",
                             turns_since_memory_surfaced=1))
        assert plan.surface_memory is None
        assert any("suppressed" in r for r in plan.reasons)

    def test_no_memory_on_a_minimal_turn(self):
        plan = plan_turn(sig("ok", relevant_memory="User's exam is on Friday",
                             turns_since_memory_surfaced=9))
        assert plan.surface_memory is None


class TestDisagreement:
    @pytest.mark.parametrize("stage", ["stranger", "acquaintance"])
    def test_not_allowed_early(self, stage):
        assert plan_turn(sig("i think that's fine", stage=stage)).allow_disagreement is False

    @pytest.mark.parametrize("stage", ["close", "partner"])
    def test_allowed_once_close(self, stage):
        assert plan_turn(sig("i think that's fine", stage=stage)).allow_disagreement is True


class TestRender:
    def test_minimal_turn_forbids_questions(self):
        out = render(plan_turn(sig("ok")))
        assert "Do NOT ask a question" in out
        assert "few words" in out

    def test_memory_is_included_with_a_recitation_guard(self):
        plan = plan_turn(sig("about the exam", relevant_memory="exam on Friday",
                             turns_since_memory_surfaced=9))
        out = render(plan)
        assert "exam on Friday" in out
        assert "recite" in out

    def test_extra_notes_are_appended(self):
        out = render(plan_turn(sig("ok")), extra_notes=["NOTE: they seem off today."])
        assert "NOTE: they seem off today." in out

    def test_render_is_one_block_not_fifteen(self):
        """The whole point: a single coherent directive."""
        out = render(plan_turn(sig("i finally finished that report today")))
        assert len(out.splitlines()) <= 8

    def test_summarise_is_loggable(self):
        assert summarise(plan_turn(sig("ok"))).startswith("turn_plan[")


class TestPlanReasons:
    def test_every_plan_explains_itself(self):
        """Decisions must be traceable — that was impossible before."""
        for msg in ["ok", "hey", "i'm so overwhelmed right now", " ".join(["w"] * 50)]:
            assert plan_turn(sig(msg)).reasons, f"no reasons recorded for {msg!r}"
