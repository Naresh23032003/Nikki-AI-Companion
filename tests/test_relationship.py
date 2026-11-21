"""Deterministic unit tests for relationship progression and friction (app/relationship.py).

Covers:
  - stage promotion (all thresholds required, exactly one stage at a time)
  - affection clamping (±2 per exchange, bounds [0, 100])
  - upset and repair (friction, stage gating, apologies, notes)
  - effective clinginess calculation
  - dev override validation

Run: PYTHONPATH=. python tests/test_relationship.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import Database
from app.relationship import (
    STAGES,
    THRESHOLDS,
    RelationshipTracker,
    effective_clinginess,
)


class RelationshipTrackerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="relationship_test_"))
        self.db = Database(self.tmp / "test.db")
        self.tracker = RelationshipTracker(self.db, memory=None)

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _set_state(
        self,
        *,
        stage: str | None = None,
        affection: float | None = None,
        days_known: int | None = None,
        memories_count: int | None = None,
    ) -> None:
        """Helper to configure relationship state in the DB."""
        # Ensure initial row exists
        self.tracker.state()

        updates = {}
        if stage is not None:
            updates["stage"] = stage
        if affection is not None:
            updates["affection"] = affection
        if days_known is not None:
            updates["days_known"] = days_known
            past_iso = (datetime.now(timezone.utc) - timedelta(days=days_known)).isoformat()
            with self.db._lock:
                self.db._conn.execute(
                    "UPDATE relationship_state SET started_at = ? WHERE id = 1",
                    (past_iso,),
                )
                self.db._conn.commit()

        if updates:
            self.db.update_relationship(**updates)

        if memories_count is not None:
            # Clear existing memories and insert desired count
            with self.db._lock:
                self.db._conn.execute("DELETE FROM memories")
                self.db._conn.commit()
            for i in range(memories_count):
                self.db.add_memory(f"fact {i}", "personal_info")

        self.tracker._invalidate_cache()

    # -- Affection clamping ----------------------------------------------------
    async def test_affection_clamping_and_bounds(self):
        """Given a stranger with affection 0,
        When apply_exchange(5, ...) runs,
        Then affection rises by exactly 2 (clamp),
        and apply_exchange(-5, ...) at affection 0 stays at 0.
        """
        self._set_state(stage="stranger", affection=0.0)
        self.assertEqual(self.tracker.state()["affection"], 0.0)

        # Delta 5 should be clamped to +2
        await self.tracker.apply_exchange(5, meaningful=False)
        self.assertEqual(self.tracker.state()["affection"], 2.0)

        # Reset affection to 0 and apply -5: should clamp to -2 and stay bounded at 0
        self._set_state(affection=0.0)
        await self.tracker.apply_exchange(-5, meaningful=False)
        self.assertEqual(self.tracker.state()["affection"], 0.0)

    # -- Stage promotion thresholds -------------------------------------------
    async def test_acquaintance_thresholds_all_met(self):
        """Given affection, days known and memory count all at the acquaintance thresholds,
        When apply_exchange(0, ...) runs,
        Then stage becomes acquaintance.
        """
        min_aff, min_days, min_mem = THRESHOLDS["acquaintance"]
        self._set_state(
            stage="stranger",
            affection=min_aff,
            days_known=min_days,
            memories_count=min_mem,
        )
        self.assertEqual(self.tracker.stage, "stranger")

        await self.tracker.apply_exchange(0, meaningful=False)
        self.assertEqual(self.tracker.stage, "acquaintance")

    async def test_acquaintance_thresholds_any_one_short(self):
        """Given any one of the three short,
        Then the stage stays stranger.
        """
        min_aff, min_days, min_mem = THRESHOLDS["acquaintance"]

        # Case 1: affection short
        self._set_state(
            stage="stranger",
            affection=min_aff - 0.5,
            days_known=min_days,
            memories_count=min_mem,
        )
        await self.tracker.apply_exchange(0, meaningful=False)
        self.assertEqual(self.tracker.stage, "stranger")

        # Case 2: days_known short
        self._set_state(
            stage="stranger",
            affection=min_aff,
            days_known=max(0, min_days - 1),
            memories_count=min_mem,
        )
        await self.tracker.apply_exchange(0, meaningful=False)
        self.assertEqual(self.tracker.stage, "stranger")

        # Case 3: memory count short
        self._set_state(
            stage="stranger",
            affection=min_aff,
            days_known=min_days,
            memories_count=max(0, min_mem - 1),
        )
        await self.tracker.apply_exchange(0, meaningful=False)
        self.assertEqual(self.tracker.stage, "stranger")

    async def test_single_stage_promotion_per_exchange(self):
        """Given all girlfriend thresholds met while at stranger,
        When one exchange runs,
        Then it advances only one stage.
        """
        gf_aff, gf_days, gf_mem = THRESHOLDS["girlfriend"]
        self._set_state(
            stage="stranger",
            affection=gf_aff,
            days_known=gf_days,
            memories_count=gf_mem,
        )
        self.assertEqual(self.tracker.stage, "stranger")

        await self.tracker.apply_exchange(0, meaningful=False)
        # Advances stranger -> acquaintance only
        self.assertEqual(self.tracker.stage, "acquaintance")

        # Second exchange advances acquaintance -> friend
        await self.tracker.apply_exchange(0, meaningful=False)
        self.assertEqual(self.tracker.stage, "friend")

    # -- Friction and upset ---------------------------------------------------
    async def test_upset_state_trigger(self):
        """Given a friend and an exchange with delta -2,
        When upset_state() is read,
        Then it is not None and carries the trigger text;
        given a stranger, Then it is None.
        """
        trigger = "You are so annoying and useless"

        # Friend with delta -2 -> triggers upset
        self._set_state(stage="friend", affection=50.0)
        await self.tracker.apply_exchange(-2, meaningful=False, trigger_text=trigger)
        upset = self.tracker.upset_state()
        self.assertIsNotNone(upset)
        reason, until_iso = upset
        self.assertEqual(reason, trigger)
        self.assertTrue(bool(until_iso))

        # Stranger with delta -2 -> no upset (no stake yet)
        self._set_state(stage="stranger", affection=5.0)
        # Clear any prior upset settings
        self.db.set_setting("upset_until", "")
        self.db.set_setting("upset_reason", "")
        await self.tracker.apply_exchange(-2, meaningful=False, trigger_text=trigger)
        self.assertIsNone(self.tracker.upset_state())

    async def test_upset_repair(self):
        """Given an upset friend,
        When maybe_repair('sorry, my bad') runs,
        Then it returns True and upset_state() is None;
        with 'ok' it returns False.
        """
        self._set_state(stage="friend", affection=50.0)
        await self.tracker.apply_exchange(-2, meaningful=False, trigger_text="bad comment")
        self.assertIsNotNone(self.tracker.upset_state())

        # Non-apology should not repair
        repaired = await self.tracker.maybe_repair("ok")
        self.assertFalse(repaired)
        self.assertIsNotNone(self.tracker.upset_state())

        # Sincere apology repairs
        repaired = await self.tracker.maybe_repair("sorry, my bad")
        self.assertTrue(repaired)
        self.assertIsNone(self.tracker.upset_state())

        # Consuming the repair note returns the one-shot note
        note = self.tracker.consume_repair_note()
        self.assertIsNotNone(note)
        self.assertIn("NOTE: They just apologized", note)
        # Next call returns None
        self.assertIsNone(self.tracker.consume_repair_note())

    # -- Clinginess calculation -----------------------------------------------
    def test_effective_clinginess(self):
        """Given effective_clinginess('stranger', 1.0), Then 0.0;
        and for every stage the result stays within 0.0 to 1.0.
        """
        self.assertEqual(effective_clinginess("stranger", 1.0), 0.0)

        test_clinginess_values = [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0]
        for stage in STAGES:
            for persona_c in test_clinginess_values:
                eff = effective_clinginess(stage, persona_c)
                self.assertGreaterEqual(eff, 0.0, f"Failed for {stage}, {persona_c}")
                self.assertLessEqual(eff, 1.0, f"Failed for {stage}, {persona_c}")

        # Stranger is always 0.0 regardless of persona clinginess
        for persona_c in test_clinginess_values:
            self.assertEqual(effective_clinginess("stranger", persona_c), 0.0)

    # -- Dev override ---------------------------------------------------------
    def test_override_validation(self):
        """override(stage='nonsense') raises ValueError."""
        with self.assertRaises(ValueError):
            self.tracker.override(stage="nonsense")

        # Valid stage override succeeds
        state = self.tracker.override(stage="friend", affection=42.0)
        self.assertEqual(state["stage"], "friend")
        self.assertEqual(state["affection"], 42.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
