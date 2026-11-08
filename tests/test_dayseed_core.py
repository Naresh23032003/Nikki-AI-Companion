"""Headless tests for app.dayseed_core (N5: life-engine world-state coherence).

No db, no llm, no datetime.now() - pure functions only.
"""
from __future__ import annotations

from app.dayseed_core import (
    TREND_FALLING,
    TREND_RISING,
    TREND_STEADY,
    ThreadState,
    advance_thread,
    compute_mood_trend,
    select_thread_to_progress,
    sync_threads,
)


class TestMoodTrend:
    def test_rising_when_affection_climbed(self):
        assert compute_mood_trend(50.0, 45.0) == TREND_RISING

    def test_falling_when_affection_dropped(self):
        assert compute_mood_trend(40.0, 47.0) == TREND_FALLING

    def test_steady_within_the_noise_band(self):
        assert compute_mood_trend(50.0, 49.0) == TREND_STEADY
        assert compute_mood_trend(50.0, 51.0) == TREND_STEADY

    def test_exactly_on_the_threshold_counts(self):
        assert compute_mood_trend(51.5, 50.0) == TREND_RISING
        assert compute_mood_trend(48.5, 50.0) == TREND_FALLING

    def test_missing_snapshot_degrades_to_steady(self):
        """Must never be worse than the old hardcoded 'steady' constant."""
        assert compute_mood_trend(None, 50.0) == TREND_STEADY
        assert compute_mood_trend(50.0, None) == TREND_STEADY
        assert compute_mood_trend(None, None) == TREND_STEADY


class TestSyncThreads:
    def test_new_persona_gets_fresh_state_for_every_thread(self):
        out = sync_threads([], ["thread A", "thread B"])
        assert [t.text for t in out] == ["thread A", "thread B"]
        assert all(t.status == "active" and t.last_touched is None for t in out)

    def test_existing_state_is_preserved_for_threads_still_listed(self):
        existing = [ThreadState("thread A", status="done", last_touched="2026-08-01")]
        out = sync_threads(existing, ["thread A", "thread B"])
        a = next(t for t in out if t.text == "thread A")
        assert a.status == "done" and a.last_touched == "2026-08-01"
        b = next(t for t in out if t.text == "thread B")
        assert b.status == "active" and b.last_touched is None

    def test_removed_threads_are_dropped(self):
        existing = [ThreadState("gone now", status="active", last_touched="2026-08-01")]
        out = sync_threads(existing, ["thread A"])
        assert [t.text for t in out] == ["thread A"]


class TestSelectThreadToProgress:
    def test_never_touched_threads_come_first(self):
        threads = [
            ThreadState("touched recently", last_touched="2026-08-08"),
            ThreadState("never touched", last_touched=None),
        ]
        chosen = select_thread_to_progress(threads)
        assert chosen.text == "never touched"

    def test_least_recently_touched_wins(self):
        threads = [
            ThreadState("touched yesterday", last_touched="2026-08-08"),
            ThreadState("touched a week ago", last_touched="2026-08-01"),
        ]
        chosen = select_thread_to_progress(threads)
        assert chosen.text == "touched a week ago"

    def test_done_threads_are_never_selected(self):
        threads = [ThreadState("finished", status="done", last_touched=None)]
        assert select_thread_to_progress(threads) is None

    def test_empty_list(self):
        assert select_thread_to_progress([]) is None

    def test_all_done_returns_none(self):
        threads = [ThreadState("a", status="done"), ThreadState("b", status="done")]
        assert select_thread_to_progress(threads) is None


class TestAdvanceThread:
    def test_touching_updates_last_touched_and_keeps_active(self):
        threads = [ThreadState("a", last_touched=None)]
        out = advance_thread(threads, "a", done=False, today="2026-08-09")
        assert out[0].last_touched == "2026-08-09"
        assert out[0].status == "active"

    def test_marking_done_removes_it_from_future_rotation(self):
        threads = [ThreadState("a", last_touched="2026-08-01")]
        out = advance_thread(threads, "a", done=True, today="2026-08-09")
        assert out[0].status == "done"
        assert select_thread_to_progress(out) is None

    def test_other_threads_are_untouched(self):
        threads = [ThreadState("a", last_touched="2026-08-01"),
                   ThreadState("b", last_touched="2026-08-01")]
        out = advance_thread(threads, "a", done=False, today="2026-08-09")
        b = next(t for t in out if t.text == "b")
        assert b.last_touched == "2026-08-01"


class TestThreadStateRoundTrip:
    def test_to_dict_from_dict_round_trips(self):
        t = ThreadState("some thread", status="done", last_touched="2026-08-09")
        assert ThreadState.from_dict(t.to_dict()) == t

    def test_from_dict_defaults_missing_fields(self):
        t = ThreadState.from_dict({"text": "x"})
        assert t.status == "active" and t.last_touched is None
