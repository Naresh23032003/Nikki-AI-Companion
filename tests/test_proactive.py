"""Tests for app/proactive.py (N7: initiative engine).

This module had ZERO test coverage despite already being architected for it
(injected now_fn, explicit db/llm/memory/relationship/tools - no globals).
Uses a real scratch app.db.Database (matching the precedent set in
tests/test_dayseed_integration.py) so get_last_activity/mood_entries_for_day/
get_setting all behave exactly as in production, plus a real
RelationshipTracker (also untested elsewhere, cheap to construct, higher
fidelity than a hand-stubbed one).

`_generate` (LLM prompt assembly + call) is monkeypatched to a canned string
in tests that need `fire_checkin`/`fire_followup` to actually fire - that
boundary keeps these tests focused on the engine's OWN orchestration logic
(skip conditions, escalation, delivery, milestone bookkeeping), not
re-testing prompt assembly which lives elsewhere.
"""
from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.db import Database
from app.proactive import ProactiveConfig, ProactiveEngine
from app.relationship import RelationshipTracker


# =============================================================== ProactiveConfig


class TestProactiveConfigParsing:
    def test_messages_per_day_range_string(self):
        cfg = ProactiveConfig.from_persona({"messages_per_day": "2-5"})
        assert (cfg.min_per_day, cfg.max_per_day) == (2, 5)

    def test_messages_per_day_list(self):
        cfg = ProactiveConfig.from_persona({"messages_per_day": [1, 3]})
        assert (cfg.min_per_day, cfg.max_per_day) == (1, 3)

    def test_messages_per_day_single_int(self):
        cfg = ProactiveConfig.from_persona({"messages_per_day": 3})
        assert (cfg.min_per_day, cfg.max_per_day) == (3, 3)

    def test_garbage_messages_per_day_falls_back_to_defaults(self):
        cfg = ProactiveConfig.from_persona({"messages_per_day": "not-a-range"})
        assert cfg.min_per_day <= cfg.max_per_day

    def test_active_hours_parsing(self):
        cfg = ProactiveConfig.from_persona({"active_hours": "09:00-22:30"})
        assert cfg.start.hour == 9 and cfg.end.hour == 22 and cfg.end.minute == 30

    def test_clinginess_clamped_to_0_1(self):
        assert ProactiveConfig.from_persona({"clinginess": 5.0}).clinginess == 1.0
        assert ProactiveConfig.from_persona({"clinginess": -2.0}).clinginess == 0.0

    def test_disabled_by_default(self):
        assert ProactiveConfig.from_persona({}).enabled is False
        assert ProactiveConfig.from_persona(None).enabled is False


class TestActiveHours:
    def test_normal_window(self):
        cfg = ProactiveConfig.from_persona({"active_hours": "08:00-23:00"})
        assert cfg.in_active_hours(datetime(2026, 1, 1, 12, 0))
        assert not cfg.in_active_hours(datetime(2026, 1, 1, 3, 0))

    def test_midnight_crossing_window(self):
        cfg = ProactiveConfig.from_persona({"active_hours": "22:00-02:00"})
        assert cfg.in_active_hours(datetime(2026, 1, 1, 23, 30))
        assert cfg.in_active_hours(datetime(2026, 1, 1, 1, 0))
        assert not cfg.in_active_hours(datetime(2026, 1, 1, 12, 0))


class TestMessagesToday:
    def test_equal_min_max_is_deterministic(self):
        cfg = ProactiveConfig.from_persona({"messages_per_day": 3})
        assert cfg.messages_today() == 3

    def test_higher_clinginess_skews_toward_the_max(self):
        import random
        rng = random.Random(42)
        low_cling = ProactiveConfig.from_persona({"messages_per_day": "0-10", "clinginess": 0.0})
        high_cling = ProactiveConfig.from_persona({"messages_per_day": "0-10", "clinginess": 1.0})
        low_draws = [low_cling.messages_today(rng) for _ in range(200)]
        high_draws = [high_cling.messages_today(rng) for _ in range(200)]
        assert sum(high_draws) > sum(low_draws)


class TestFollowupDelay:
    def test_second_attempt_is_sooner_than_the_first(self):
        import random
        rng = random.Random(7)
        cfg = ProactiveConfig.from_persona({"clinginess": 0.5})
        first = [cfg.followup_delay_hours(0, rng) for _ in range(50)]
        second = [cfg.followup_delay_hours(1, rng) for _ in range(50)]
        assert sum(second) / len(second) < sum(first) / len(first)

    def test_never_below_the_floor(self):
        import random
        rng = random.Random(1)
        cfg = ProactiveConfig.from_persona({"clinginess": 1.0})
        assert all(cfg.followup_delay_hours(1, rng) >= 0.75 for _ in range(50))


# =============================================================== ProactiveEngine


class _FakeLLM:
    async def chat(self, *a, **kw):
        return "hey, thinking about you"


class _FakeMemory:
    def __init__(self):
        self.facts = []

    async def retrieve_memories(self, query, k=4):
        return []

    async def add_fact(self, fact, category, **kw):
        self.facts.append((fact, category))
        return 1


def _persona(proactive=None):
    return SimpleNamespace(
        name="Test", proactive=proactive or {
            "enabled": True, "messages_per_day": "2-3",
            "active_hours": "08:00-23:00", "clinginess": 0.5,
            "escalate_on_silence": True,
        })


@pytest.fixture
def db():
    tmp = tempfile.mkdtemp(prefix="nikki_proactive_test_")
    database = Database(Path(tmp) / "test.db")
    yield database
    database.close()


def _engine(db, *, now, persona=None, relationship=True, tools=None):
    rel = RelationshipTracker(db) if relationship else None
    eng = ProactiveEngine(
        db=db, llm=_FakeLLM(), memory=_FakeMemory(),
        get_persona=lambda: (persona or _persona()),
        settings=SimpleNamespace(wa_session_id="main", wa_bridge_url="http://nope:1",
                                 behavior={}),
        now_fn=lambda: now, relationship=rel, tools=tools,
        session_id="main",
    )
    return eng


class TestFireCheckinSkipConditions:
    def test_skips_when_disabled(self, db):
        eng = _engine(db, now=datetime(2026, 1, 1, 12, 0),
                     persona=_persona({"enabled": False}))
        assert asyncio.run(eng.fire_checkin("random_thought")) is False

    def test_skips_when_paused(self, db):
        eng = _engine(db, now=datetime(2026, 1, 1, 12, 0))
        eng.pause_for(2.0)
        assert asyncio.run(eng.fire_checkin("random_thought")) is False

    def test_skips_outside_active_hours(self, db):
        eng = _engine(db, now=datetime(2026, 1, 1, 3, 0))  # 3am, window is 08-23
        assert asyncio.run(eng.fire_checkin("random_thought")) is False

    def test_skips_mid_conversation(self, db):
        eng = _engine(db, now=datetime(2026, 1, 1, 12, 0))
        db.ensure_session("main")
        db.add_message("main", "user", "hey", source="webapp_chat")
        assert asyncio.run(eng.fire_checkin("random_thought")) is False

    def test_milestone_fires_even_mid_conversation(self, db, monkeypatch):
        eng = _engine(db, now=datetime(2026, 1, 1, 12, 0))
        db.ensure_session("main")
        db.add_message("main", "user", "hey", source="webapp_chat")
        monkeypatch.setattr(eng, "_generate", lambda *a, **kw: asyncio.sleep(0, result="happy text"))
        assert asyncio.run(eng.fire_checkin("milestone", "days_7")) is True

    def test_resume_after_pause(self, db):
        eng = _engine(db, now=datetime(2026, 1, 1, 12, 0))
        eng.pause_for(2.0)
        assert eng.paused_until() is not None
        eng.pause_for(0)
        assert eng.paused_until() is None


class TestFireCheckinFiring:
    def test_fires_and_stores_the_message(self, db, monkeypatch):
        eng = _engine(db, now=datetime(2026, 1, 1, 12, 0))
        monkeypatch.setattr(eng, "_generate", lambda *a, **kw: asyncio.sleep(0, result="hey you"))
        fired = asyncio.run(eng.fire_checkin("random_thought"))
        assert fired is True
        activity = db.get_last_activity("main")
        assert activity["last_assistant_ts"] is not None

    def test_generation_failure_is_a_skip_not_a_crash(self, db, monkeypatch):
        eng = _engine(db, now=datetime(2026, 1, 1, 12, 0))
        monkeypatch.setattr(eng, "_generate", lambda *a, **kw: asyncio.sleep(0, result=None))
        assert asyncio.run(eng.fire_checkin("random_thought")) is False

    def test_stranger_hello_only_sent_once(self, db, monkeypatch):
        eng = _engine(db, now=datetime(2026, 1, 1, 12, 0))
        monkeypatch.setattr(eng, "_generate", lambda *a, **kw: asyncio.sleep(0, result="hi!"))
        assert asyncio.run(eng.fire_checkin("hello")) is True
        assert db.get_setting("proactive_stranger_hello_sent") == "1"


class TestFollowupEscalation:
    def test_no_followup_if_they_already_replied(self, db):
        eng = _engine(db, now=datetime(2026, 1, 1, 12, 0))
        db.set_setting("proactive_last_sent",
                       (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat())
        db.ensure_session("main")
        db.add_message("main", "user", "hi back!", source="webapp_chat")
        assert asyncio.run(eng.fire_followup(0)) is False

    def test_followup_fires_when_still_silent(self, db, monkeypatch):
        eng = _engine(db, now=datetime(2026, 1, 1, 12, 0))
        db.set_setting("proactive_last_sent",
                       (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat())
        monkeypatch.setattr(eng, "_generate", lambda *a, **kw: asyncio.sleep(0, result="hello??"))
        assert asyncio.run(eng.fire_followup(0)) is True

    def test_escalation_cap_stores_a_sulk_memory_and_stops(self, db):
        eng = _engine(db, now=datetime(2026, 1, 1, 12, 0))
        db.set_setting("proactive_last_sent",
                       (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat())
        fired = asyncio.run(eng.fire_followup(2))  # attempt >= 2 -> cap
        assert fired is False
        assert any("ignored" in f for f, _ in eng.memory.facts)


def _set_started_at(db, days_ago: int) -> None:
    """update_relationship() deliberately doesn't allow changing started_at
    (see app/db.py's `allowed` set - it's meant to be set once, at creation).
    Reach into the row directly for test setup; get_relationship() first so
    the row exists to update."""
    db.get_relationship()
    started = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    db._conn.execute("UPDATE relationship_state SET started_at = ? WHERE id = 1",
                     (started,))
    db._conn.commit()


class TestMilestones:
    def test_days_known_milestone_detected(self, db):
        eng = _engine(db, now=datetime(2026, 1, 1, 12, 0))
        _set_started_at(db, 7)
        key = eng._pending_milestone()
        assert key == "days_7"

    def test_already_sent_milestone_is_not_repeated(self, db):
        eng = _engine(db, now=datetime(2026, 1, 1, 12, 0))
        _set_started_at(db, 7)
        db.set_setting("milestone_sent:days_7", "1")
        assert eng._pending_milestone() is None

    def test_no_milestone_on_an_ordinary_day(self, db):
        eng = _engine(db, now=datetime(2026, 1, 1, 12, 0))
        _set_started_at(db, 13)
        assert eng._pending_milestone() is None


class TestPlanDay:
    def test_disabled_persona_plans_nothing(self, db):
        eng = _engine(db, now=datetime(2026, 1, 1, 8, 0),
                     persona=_persona({"enabled": False}))
        assert eng.plan_day() == []

    def test_plans_the_configured_count_within_the_window(self, db):
        # add_job() works on an unstarted scheduler (jobs queue, they just
        # don't fire yet) - AsyncIOScheduler.start() itself requires a
        # running asyncio loop, which this plain sync test doesn't have and
        # doesn't need just to verify plan_day()'s own scheduling logic.
        #
        # relationship=False deliberately: a fresh relationship defaults to
        # stage="stranger", and config() correctly caps a stranger to exactly
        # 1 message/day regardless of the persona's own messages_per_day (see
        # TestPlanDay::test_stranger_stage_caps_to_one_message_regardless_of_config
        # below) - this test isolates plan_day()'s own scheduling logic from
        # that separate, already-covered stage-scaling behaviour.
        eng = _engine(db, now=datetime(2026, 1, 1, 8, 0), relationship=False,
                     persona=_persona({"enabled": True, "messages_per_day": 3,
                                       "active_hours": "08:00-23:00"}))
        times = eng.plan_day()
        assert len(times) == 3
        for t in times:
            assert eng.now().replace(hour=8, minute=0) <= t <= eng.now().replace(hour=23, minute=0)

    def test_stranger_stage_caps_to_one_message_regardless_of_config(self, db):
        """A stranger doesn't get spammed with the persona's full
        messages_per_day - config() scales this down to exactly 1 (the
        one-shot 'hello') until the relationship warms up."""
        eng = _engine(db, now=datetime(2026, 1, 1, 8, 0),
                     persona=_persona({"enabled": True, "messages_per_day": 3,
                                       "active_hours": "08:00-23:00"}))
        assert eng.relationship.stage == "stranger"
        times = eng.plan_day()
        assert len(times) == 1

    def test_nothing_planned_once_the_window_has_passed(self, db):
        eng = _engine(db, now=datetime(2026, 1, 1, 23, 55),
                     persona=_persona({"active_hours": "08:00-23:00"}))
        assert eng.plan_day() == []
