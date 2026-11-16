"""Tests for bubble splitting and time helpers extracted from main.py.

Neither had tests before extraction. The midnight-crossing quiet-hours window
and the bubble-overflow regrouping are the two cases most likely to be wrong
and least likely to be noticed.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from app.conversation.bubbles import MAX_BUBBLES, sentence_bubbles, split_bubbles
from app.timing import hhmm, in_quiet_hours, now_context, sse


class TestSplitBubbles:
    def test_short_reply_stays_one_bubble(self):
        assert split_bubbles("hey you") == ["hey you"]

    def test_blank_line_is_the_primary_split_signal(self):
        assert split_bubbles("omg\n\nwait what happened") == ["omg", "wait what happened"]

    def test_empty_text_yields_nothing(self):
        assert split_bubbles("") == []
        assert split_bubbles("   \n  ") == []

    def test_long_multi_sentence_reply_is_split(self):
        text = ("That sounds really rough honestly. I hope you managed to get "
                "some rest afterwards. Did you eat anything at all today?")
        out = split_bubbles(text)
        assert len(out) > 1

    def test_long_single_sentence_is_not_split(self):
        text = "i really do think that " + "you should rest a bit more today " * 3
        assert len(split_bubbles(text)) == 1

    def test_never_exceeds_the_bubble_cap(self):
        text = "\n\n".join(f"line number {i} here" for i in range(12))
        assert len(split_bubbles(text)) <= MAX_BUBBLES

    def test_overflow_is_folded_not_dropped(self):
        text = "\n\n".join(["alpha", "bravo", "charlie", "delta", "echo"])
        out = split_bubbles(text)
        assert "echo" in " ".join(out), "overflow content must never be lost"


class TestSentenceBubbles:
    def test_single_sentence_returns_unchanged(self):
        assert sentence_bubbles("just the one thing") == ["just the one thing"]

    def test_tiny_fragment_folds_into_the_previous_bubble(self):
        out = sentence_bubbles("I went to the market today. Right?")
        assert len(out) == 1
        assert "Right?" in out[0]

    def test_overflow_is_length_balanced_not_dumped_in_the_last(self):
        """Regression: folding all overflow into the last bubble recreated the
        333-char wall of text this function exists to prevent."""
        sentences = " ".join(
            f"This is sentence number {i} and it is reasonably long." for i in range(12)
        )
        out = sentence_bubbles(sentences)
        assert len(out) <= MAX_BUBBLES
        longest, shortest = max(map(len, out)), min(map(len, out))
        assert longest < shortest * 3, f"bubbles badly unbalanced: {list(map(len, out))}"

    def test_no_content_is_lost(self):
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        assert "Eight" in " ".join(sentence_bubbles(text))


class TestQuietHours:
    @pytest.mark.parametrize("clock,expected", [
        ("03:00", True), ("01:00", True), ("07:30", True),
        ("07:31", False), ("00:59", False), ("14:00", False),
    ])
    def test_window_crossing_midnight(self, clock, expected):
        """01:00-07:30 spans midnight in the sense that start > end never
        happens here, but the boundary conditions still need pinning."""
        h, m = map(int, clock.split(":"))
        assert in_quiet_hours("01:00-07:30", datetime(2026, 8, 8, h, m)) is expected

    @pytest.mark.parametrize("clock,expected", [
        ("23:00", True), ("02:00", True), ("06:00", True),
        ("12:00", False), ("21:00", False),
    ])
    def test_window_that_genuinely_wraps(self, clock, expected):
        h, m = map(int, clock.split(":"))
        assert in_quiet_hours("22:00-07:00", datetime(2026, 8, 8, h, m)) is expected

    def test_empty_window_means_never_quiet(self):
        assert in_quiet_hours("") is False
        assert in_quiet_hours(None) is False

    @pytest.mark.parametrize("bad", ["nonsense", "25:00-99:99", "01:00", "-", "a:b-c:d"])
    def test_malformed_window_fails_open(self, bad):
        """A typo in config must not silently suppress every delivery forever."""
        assert in_quiet_hours(bad, datetime(2026, 8, 8, 3, 0)) is False


class TestHhmm:
    def test_parses_a_normal_time(self):
        assert hhmm("07:30") == (7, 30)

    @pytest.mark.parametrize("bad", ["", None, "garbage"])
    def test_falls_back_to_the_default(self, bad):
        assert hhmm(bad) == (23, 45)

    def test_custom_default(self):
        assert hhmm("nope", default="06:15") == (6, 15)


class TestNowContext:
    def test_includes_weekday_date_and_time_in_one_line(self):
        out = now_context(datetime(2026, 8, 8, 14, 5))
        assert "Saturday" in out and "August" in out and "8" in out
        assert "2:05 PM" in out

    @pytest.mark.parametrize("hour,label", [
        (2, "the middle of the night"), (5, "early morning"), (9, "morning"),
        (14, "afternoon"), (19, "evening"), (22, "night"),
    ])
    def test_time_of_day_labels(self, hour, label):
        assert label in now_context(datetime(2026, 8, 8, hour, 0))


class TestSSE:
    def test_plain_frame(self):
        assert sse("hello") == "data: hello\n\n"

    def test_named_event_frame(self):
        assert sse("hello", "chunk") == "event: chunk\ndata: hello\n\n"
