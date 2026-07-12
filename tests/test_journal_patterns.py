"""Deterministic tests for the programmatic pattern/streak layer (app/journal.py).

No model involved — the whole point of making this layer pure code is that
these behaviors are now provable:
  - a single bad day never flags a streak
  - 2-3 bad days DO flag a streak (available the very next day)
  - one bad Mon-Tue-Wed week is NOT a weekly "pattern" (the original bug)
  - the same weekday recurring across weeks IS
  - streak notes never stack while one is pending

Run: PYTHONPATH=. python tests/test_journal_patterns.py
"""
import asyncio
import shutil
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import Database
from app.journal import (
    _is_negative_mood,
    run_recent_streak_check,
    run_weekly_patterns,
)


class FakeMemory:
    """Captures add_fact calls; no embeddings/Chroma/Ollama needed."""

    def __init__(self, db):
        self.db = db
        self.facts = []

    async def add_fact(self, fact, category, **kwargs):
        self.facts.append((fact, category))
        return self.db.add_memory(fact, category)


def _day(offset: int) -> str:
    return (date.today() + timedelta(days=offset)).isoformat()


class JournalPatternTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="journal_test_"))
        self.db = Database(self.tmp / "test.db")
        self.memory = FakeMemory(self.db)

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _add(self, day_offset, mood, intensity=3, why="said things felt heavy at work"):
        self.db.add_mood_entry(_day(day_offset), "21:00", mood, intensity, why, "webapp_chat")

    # -- mood classification ------------------------------------------------
    def test_mood_classification(self):
        for neg in ("sad", "stressed", "anxious", "exhausted", "overwhelmed",
                    "worried sick", "frustrated"):
            self.assertTrue(_is_negative_mood(neg), neg)
        for not_neg in ("happy", "content", "excited", "curious", "surprised", ""):
            self.assertFalse(_is_negative_mood(not_neg), not_neg)

    # -- streak (nightly) ----------------------------------------------------
    def test_single_bad_day_is_not_a_streak(self):
        self._add(0, "sad", 4)
        fired = asyncio.run(run_recent_streak_check(self.db, self.memory))
        self.assertFalse(fired)
        self.assertEqual(self.memory.facts, [])

    def test_two_of_three_bad_days_is_a_streak(self):
        self._add(-2, "sad", 3)
        self._add(0, "exhausted", 4, why="said they can't keep this up")
        fired = asyncio.run(run_recent_streak_check(self.db, self.memory))
        self.assertTrue(fired)
        fact = self.memory.facts[0][0]
        self.assertTrue(fact.startswith("Streak:"), fact)
        self.assertIn("2 of the last 3 days", fact)
        self.assertIn("can't keep this up", fact)  # strongest entry's evidence

    def test_positive_days_never_flag(self):
        self._add(-2, "happy", 4)
        self._add(-1, "excited", 5)
        self._add(0, "content", 3)
        fired = asyncio.run(run_recent_streak_check(self.db, self.memory))
        self.assertFalse(fired)

    def test_streaks_do_not_stack(self):
        self._add(-1, "sad", 3)
        self._add(0, "sad", 4)
        self.assertTrue(asyncio.run(run_recent_streak_check(self.db, self.memory)))
        # Second nightly run while the first Streak: memory is unconsumed:
        self.assertFalse(asyncio.run(run_recent_streak_check(self.db, self.memory)))
        streaks = [f for f, _ in self.memory.facts if f.startswith("Streak:")]
        self.assertEqual(len(streaks), 1)

    # -- weekly patterns -----------------------------------------------------
    def test_one_bad_week_is_not_a_pattern(self):
        """The original bug: sad Mon+Tue+Wed of ONE week reported as a
        pattern on Sunday. Now it must produce zero Pattern: memories
        (padded with neutral entries on other days to clear the >=5 gate)."""
        monday = date.today() - timedelta(days=date.today().weekday())
        for i in range(3):  # Mon, Tue, Wed of the current week
            d = (monday + timedelta(days=i)).isoformat()
            self.db.add_mood_entry(d, "21:00", "sad", 4, "said work was crushing them", "webapp_chat")
        for i in (10, 12, 15):  # neutral filler on earlier days
            self.db.add_mood_entry((monday - timedelta(days=i)).isoformat(),
                                   "12:00", "content", 2, "said the day went fine", "webapp_chat")
        stored = asyncio.run(run_weekly_patterns(self.db, self.memory))
        patterns = [f for f, _ in self.memory.facts if f.startswith("Pattern:")]
        self.assertEqual(stored, 0, patterns)

    def test_recurring_weekday_across_weeks_is_a_pattern(self):
        """Sad on 3 separate Sundays -> weekday pattern."""
        today = date.today()
        last_sunday = today - timedelta(days=(today.weekday() + 1) % 7 or 7)
        for weeks_back in (0, 1, 2):
            d = (last_sunday - timedelta(weeks=weeks_back)).isoformat()
            self.db.add_mood_entry(d, "21:30", "lonely", 3, "said the flat felt empty", "webapp_chat")
        for i in (2, 4):  # neutral filler to clear the >=5 entry gate
            self.db.add_mood_entry((today - timedelta(days=i)).isoformat(),
                                   "12:00", "content", 2, "said the day went fine", "webapp_chat")
        stored = asyncio.run(run_weekly_patterns(self.db, self.memory))
        self.assertGreaterEqual(stored, 1)
        pattern = next(f for f, _ in self.memory.facts if f.startswith("Pattern:"))
        self.assertIn("lonely", pattern)
        self.assertIn("Sunday", pattern)

    def test_fewer_than_five_entries_skips(self):
        self._add(-1, "sad", 3)
        self._add(0, "sad", 4)
        stored = asyncio.run(run_weekly_patterns(self.db, self.memory))
        self.assertEqual(stored, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
