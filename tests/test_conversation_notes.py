"""Tests for the prompt-note builders extracted from main.py.

These had no tests before extraction — they were unreachable behind a global
profile registry and an un-seeded RNG. Dependencies are now injected, so the
throttles, stage gates and decline tracking are all deterministic here.
"""
from __future__ import annotations

import json
import random

import pytest

from app.conversation.notes import (
    NoteContext,
    awaiting_followup_note,
    is_gibberish,
    nonsense_note,
    offer_note,
    pattern_note,
    persona_other_names,
    streak_note,
    track_offer_decline,
    unresolved_note,
)


class FakeDB:
    """In-memory stand-in for the slice of Database these notes touch."""

    def __init__(self, settings=None, memories=None, awaiting=None):
        self.settings = dict(settings or {})
        self.memories = list(memories or [])
        self.awaiting = awaiting
        self.resolved: list[int] = []
        self.deleted: list[int] = []

    def get_setting(self, key):
        return self.settings.get(key)

    def set_setting(self, key, value):
        self.settings[key] = value

    def list_memories_by_category(self, category):
        return [m for m in self.memories if m.get("category") == category]

    def delete_memory(self, memory_id):
        self.deleted.append(memory_id)
        self.memories = [m for m in self.memories if m["id"] != memory_id]
        return True

    def get_awaiting_followup(self, session_id):
        return self.awaiting

    def resolve_event_followup(self, followup_id):
        self.resolved.append(followup_id)


class AlwaysRoll(random.Random):
    """Deterministic RNG: always returns `value` from random()."""

    def __init__(self, value: float):
        super().__init__(0)
        self._value = value

    def random(self):  # noqa: D102
        return self._value


def ctx(db=None, *, stage="close", roll=0.0, behavior=None):
    return NoteContext(
        db=db or FakeDB(),
        behavior=behavior if behavior is not None else {"eagerness": 0.5, "offer_min_gap": 6},
        stage=stage,
        rng=AlwaysRoll(roll),
    )


# ------------------------------------------------------------------ gibberish


class TestGibberish:
    @pytest.mark.parametrize("text", ["rrrrrrr", "qwrtp", "zxcvbn", "hjkl", "bcdfg"])
    def test_keyboard_mash_detected(self, text):
        assert is_gibberish(text)

    @pytest.mark.parametrize("text", ["asdfgh", "qwerty", "asdasd"])
    def test_known_gap_mash_containing_a_vowel_is_not_detected(self, text):
        """Documented limitation, not a regression.

        The detector treats any token containing a vowel as a plausible word,
        so vowel-bearing keyboard mash ('asdfgh' has an 'a') reads as real
        text. The original bug this guarded against was 'rrrrrr', which has no
        vowels and is caught. Tightening this is conversation-engine work (N3)
        and was deliberately not attempted during a behaviour-preserving
        extraction — this test pins the current behaviour so a future change
        is visible rather than accidental.
        """
        assert not is_gibberish(text)

    @pytest.mark.parametrize("text", ["hello", "noooo", "hmmmm", "idk", "u up",
                                      "ok", "brb", "ngl that was wild"])
    def test_real_text_is_not_gibberish(self, text):
        assert not is_gibberish(text)

    @pytest.mark.parametrize("text", ["??", "😂😂", "...", "!!!", ""])
    def test_punctuation_and_emoji_are_signals_not_gibberish(self, text):
        """Pure emoji/punctuation is a real reaction and must be answered."""
        assert not is_gibberish(text)

    def test_elongation_collapses_to_a_word(self):
        assert not is_gibberish("noooooooo")


class TestNonsenseNote:
    def test_normal_message_produces_no_note(self):
        assert nonsense_note(ctx(), "how was your day") is None

    def test_gibberish_produces_a_note(self):
        note = nonsense_note(ctx(), "rrrrrrrr")
        assert note and "keyboard-mash" in note

    def test_repeat_of_the_same_message_is_flagged(self):
        db = FakeDB(settings={"last_user_msg": "hello"})
        note = nonsense_note(ctx(db), "hello")
        assert note and "exact same message" in note

    def test_streak_escalates_to_the_blunt_response(self):
        db = FakeDB()
        c = ctx(db)
        for _ in range(4):
            note = nonsense_note(c, "zxcvbn")
        assert "Stop playing" in note
        assert "4 times in a row" in note

    def test_streak_resets_on_a_real_message(self):
        db = FakeDB()
        c = ctx(db)
        nonsense_note(c, "zxcvbn")
        assert nonsense_note(c, "sorry, my phone was in my pocket") is None
        assert db.settings["nonsense_streak"] == "0"

    def test_never_invents_content(self):
        """The bug this existed to fix: she confabulated a whole evening."""
        note = nonsense_note(ctx(), "rrrrrrrrrr")
        assert "Do NOT" in note and "invent" in note


# ---------------------------------------------------------------------- offers


class TestOfferNote:
    def test_reserved_stages_do_not_offer(self):
        for stage in ("stranger", "acquaintance"):
            assert offer_note(ctx(stage=stage), "i'm starving") is None

    def test_relevant_mention_produces_an_offer(self):
        note = offer_note(ctx(), "i'm starving and haven't eaten")
        assert note and "food" in note

    def test_irrelevant_message_produces_nothing(self):
        assert offer_note(ctx(), "just finished a book") is None

    def test_declined_topic_is_never_offered_again(self):
        db = FakeDB(settings={"declined_offers": json.dumps(["food"])})
        assert offer_note(ctx(db), "i'm starving") is None

    def test_throttled_by_recent_offer(self):
        db = FakeDB(settings={"exchange_count": "10", "last_offer_at": "8"})
        assert offer_note(ctx(db), "i'm starving") is None, "gap of 6 not respected"

    def test_offer_allowed_once_the_gap_has_passed(self):
        db = FakeDB(settings={"exchange_count": "20", "last_offer_at": "8"})
        assert offer_note(ctx(db), "i'm starving") is not None

    def test_eagerness_roll_can_suppress_the_offer(self):
        assert offer_note(ctx(roll=0.99), "i'm starving") is None

    def test_offer_records_the_actual_topic(self):
        """Regression: a decline used to record the literal string 'last'."""
        db = FakeDB()
        offer_note(ctx(db), "i'm so broke right now")
        assert db.settings["last_offer_topic"] == "money"

    def test_corrupt_declined_json_does_not_crash(self):
        db = FakeDB(settings={"declined_offers": "{not json"})
        assert offer_note(ctx(db), "i'm starving") is not None


class TestOfferDecline:
    def test_decline_after_an_offer_drops_the_topic(self):
        db = FakeDB(settings={"exchange_count": "11", "last_offer_at": "10",
                              "last_offer_topic": "food"})
        track_offer_decline(ctx(db), "no i'm fine")
        assert json.loads(db.settings["declined_offers"]) == ["food"]

    def test_acceptance_does_not_drop_the_topic(self):
        db = FakeDB(settings={"exchange_count": "11", "last_offer_at": "10",
                              "last_offer_topic": "food"})
        track_offer_decline(ctx(db), "yes please that would be lovely")
        assert "declined_offers" not in db.settings

    def test_decline_unrelated_to_an_offer_is_ignored(self):
        db = FakeDB(settings={"exchange_count": "50", "last_offer_at": "2",
                              "last_offer_topic": "food"})
        track_offer_decline(ctx(db), "nope")
        assert "declined_offers" not in db.settings

    @pytest.mark.parametrize("reply", ["no", "nah", "nope", "don't", "it's okay",
                                       "i'm fine", "i'm good"])
    def test_decline_phrasings(self, reply):
        db = FakeDB(settings={"exchange_count": "11", "last_offer_at": "10",
                              "last_offer_topic": "food"})
        track_offer_decline(ctx(db), reply)
        assert "declined_offers" in db.settings


# ------------------------------------------------------------------- patterns


class TestPatternNote:
    def _db(self):
        return FakeDB(
            settings={"exchange_count": "100"},
            memories=[{"id": 1, "category": "relationship",
                       "fact": "Pattern: they go quiet when work is heavy"}],
        )

    def test_reserved_stages_never_reference_patterns(self):
        assert pattern_note(ctx(self._db(), stage="stranger")) is None

    def test_pattern_surfaces_when_the_gate_opens(self):
        note = pattern_note(ctx(self._db()))
        assert note and "go quiet when work is heavy" in note

    def test_pattern_never_mentions_the_journal(self):
        note = pattern_note(ctx(self._db()))
        assert "journal" in note, "the instruction not to mention it must be present"
        assert "never mentioning" in note

    def test_no_patterns_means_no_note(self):
        assert pattern_note(ctx(FakeDB(settings={"exchange_count": "100"}))) is None

    def test_throttle_is_four_times_the_offer_gap(self):
        db = self._db()
        db.settings["last_pattern_ref_at"] = "80"  # 20 < 6*4
        assert pattern_note(ctx(db)) is None

    def test_avoids_repeating_the_same_pattern(self):
        db = self._db()
        db.memories.append({"id": 2, "category": "relationship",
                            "fact": "Pattern: they text more on weekends"})
        db.settings["last_pattern_ref_id"] = "1"
        note = pattern_note(ctx(db))
        assert "weekends" in note


class TestStreakNote:
    def _db(self):
        return FakeDB(
            settings={"exchange_count": "100"},
            memories=[{"id": 7, "category": "relationship",
                       "fact": "Streak: rough few days at work"}],
        )

    def test_streak_surfaces_and_is_consumed(self):
        db = self._db()
        note = streak_note(ctx(db))
        assert note and "rough few days at work" in note
        assert db.deleted == [7], "a streak is one-shot and must be deleted"

    def test_consume_callback_receives_the_id(self):
        seen = []
        streak_note(ctx(self._db()), on_consume=seen.append)
        assert seen == [7]

    def test_callback_failure_does_not_break_the_reply(self):
        def boom(_):
            raise RuntimeError("chroma down")

        assert streak_note(ctx(self._db()), on_consume=boom) is not None

    def test_reserved_stages_do_not_surface_streaks(self):
        assert streak_note(ctx(self._db(), stage="acquaintance")) is None

    def test_no_streaks_means_no_note(self):
        assert streak_note(ctx(FakeDB(settings={"exchange_count": "100"}))) is None


class TestUnresolvedNote:
    """Mirrors TestStreakNote - unresolved_note (N6) has the identical
    throttle/one-shot-consumption shape, only the source prefix differs."""

    def _db(self):
        return FakeDB(
            settings={"exchange_count": "100"},
            memories=[{"id": 9, "category": "relationship",
                       "fact": "Unresolved: On 2026-08-01, they mentioned "
                               "feeling stressed - said the presentation was "
                               "a disaster. It never came up again after that."}],
        )

    def test_unresolved_surfaces_and_is_consumed(self):
        db = self._db()
        note = unresolved_note(ctx(db))
        assert note and "presentation was" in note
        assert db.deleted == [9], "unresolved is one-shot and must be deleted"

    def test_consume_callback_receives_the_id(self):
        seen = []
        unresolved_note(ctx(self._db()), on_consume=seen.append)
        assert seen == [9]

    def test_callback_failure_does_not_break_the_reply(self):
        def boom(_):
            raise RuntimeError("chroma down")

        assert unresolved_note(ctx(self._db()), on_consume=boom) is not None

    def test_reserved_stages_do_not_surface_unresolved(self):
        assert unresolved_note(ctx(self._db(), stage="acquaintance")) is None

    def test_no_unresolved_means_no_note(self):
        assert unresolved_note(ctx(FakeDB(settings={"exchange_count": "100"}))) is None

    def test_streak_and_unresolved_throttles_are_independent(self):
        """Referencing a streak must not also consume the unresolved cooldown
        (they share the mechanics but track separate 'last_ref_at' settings)."""
        db = FakeDB(
            settings={"exchange_count": "100"},
            memories=[{"id": 7, "category": "relationship", "fact": "Streak: rough week"},
                     {"id": 9, "category": "relationship",
                      "fact": "Unresolved: something specific"}],
        )
        streak_note(ctx(db))
        assert unresolved_note(ctx(db)) is not None, \
            "consuming the streak note must not throttle the unrelated unresolved note"


# ------------------------------------------------------------------- followup


class TestAwaitingFollowup:
    def test_pending_followup_produces_a_note_and_resolves(self):
        db = FakeDB(awaiting={"id": 3, "event_fact": "the exam"})
        note = awaiting_followup_note(ctx(db), "sess")
        assert note and "the exam" in note
        assert db.resolved == [3], "must resolve so she doesn't ask twice"

    def test_no_followup_produces_nothing(self):
        assert awaiting_followup_note(ctx(FakeDB()), "sess") is None


# -------------------------------------------------------------------- persona


class TestPersonaOtherNames:
    def test_extracts_backstory_friend_names(self):
        class P:
            life = {"friends": [{"name": "Riya"}, {"name": "Dev"}]}

        assert persona_other_names(P()) == ["Riya", "Dev"]

    def test_missing_life_is_safe(self):
        class P:
            pass

        assert persona_other_names(P()) == []

    def test_malformed_entries_are_skipped(self):
        class P:
            life = {"friends": [{"name": "Riya"}, "not a dict", {}]}

        assert persona_other_names(P()) == ["Riya"]
