"""Integration tests for N5's wiring: DayLife._generate() actually uses
app.dayseed_core's thread selection and trend computation, and persists state
correctly across restarts (a fresh DayLife instance over the same Database).

Uses a real scratch app.db.Database (headless SQLite, no Ollama) and a fake
LLM, following the pattern established in
tests/test_turn_planning_integration.py.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.dayseed import DayLife
from app.db import Database


class _FakeLLM:
    """Scripted JSON responses, one per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def chat(self, messages, **kw):
        self.prompts.append(messages[0]["content"])
        return json.dumps(self.responses.pop(0))


def _persona(name="Test Persona", **life_kw):
    life = {
        "ongoing_threads": ["thread A", "thread B", "thread C"],
        "friends": [{"name": "Sam"}, {"name": "Riya"}],
        "recurring": {},
        **life_kw,
    }
    return SimpleNamespace(name=name, life=life)


@pytest.fixture
def db():
    tmp = tempfile.mkdtemp(prefix="nikki_dayseed_test_")
    database = Database(Path(tmp) / "test.db")
    yield database
    database.close()


DAY1 = "2026-08-01"
DAY2 = "2026-08-02"


def _basic_response(thread_focus_hint=None, thread_done=False):
    return {
        "mood": "content", "energy": 3,
        "slots": {"morning": "coffee", "afternoon": "work", "evening": "rest"},
        "on_mind": "nothing much",
        "thread_update": f"progressed {thread_focus_hint}" if thread_focus_hint else None,
        "thread_done": thread_done,
        "random_event": None,
    }


class TestThreadRotationWiring:
    def test_first_generation_picks_a_never_touched_thread_and_names_it_in_the_prompt(self, db):
        llm = _FakeLLM([_basic_response("thread A")])
        life = DayLife(db, llm, lambda: _persona())
        state = asyncio.run(life._generate(DAY1))
        assert state["mood"] == "content"
        # The chosen thread must actually be named in the prompt sent to the LLM.
        assert any(t in llm.prompts[0] for t in ("thread A", "thread B", "thread C"))

    def test_successive_generations_rotate_through_different_threads(self, db):
        llm = _FakeLLM([_basic_response(), _basic_response(), _basic_response()])
        life = DayLife(db, llm, lambda: _persona())
        for i, day in enumerate(["2026-08-01", "2026-08-02", "2026-08-03"]):
            asyncio.run(life._generate(day))
        prompts_thread_lines = [
            next(line for line in p.splitlines() if line.startswith("The ONE thread"))
            for p in llm.prompts
        ]
        # All three prompts must have named a DIFFERENT thread - not the same
        # one three times (the old "model freely picks" behaviour could do
        # exactly that, silently).
        assert len(set(prompts_thread_lines)) == 3

    def test_marking_a_thread_done_removes_it_from_rotation(self, db):
        # First call touches "thread A" (never-touched threads sort first,
        # and the persona lists A, B, C in that order) and marks it done.
        llm = _FakeLLM([_basic_response("thread A", thread_done=True),
                        _basic_response("thread B"),
                        _basic_response("thread B or C, never A again")])
        life = DayLife(db, llm, lambda: _persona())
        asyncio.run(life._generate("2026-08-01"))  # selects+marks "thread A" done
        asyncio.run(life._generate("2026-08-02"))
        asyncio.run(life._generate("2026-08-03"))
        # Only the calls AFTER thread A was marked done must never reselect it.
        for p in llm.prompts[1:]:
            line = next(line for line in p.splitlines() if line.startswith("The ONE thread"))
            assert "thread A" not in line, "a thread marked done must never be reselected"

    def test_thread_state_survives_a_fresh_DayLife_instance(self, db):
        """Simulates an app restart: a new DayLife wraps the SAME Database."""
        llm1 = _FakeLLM([_basic_response("thread A", thread_done=True)])
        DayLife(db, llm1, lambda: _persona())
        asyncio.run(DayLife(db, llm1, lambda: _persona())._generate("2026-08-01"))

        llm2 = _FakeLLM([_basic_response("thread B")])
        life2 = DayLife(db, llm2, lambda: _persona())
        asyncio.run(life2._generate("2026-08-02"))
        line = next(l for l in llm2.prompts[0].splitlines()
                    if l.startswith("The ONE thread"))
        assert "thread A" not in line

    def test_editing_the_persona_thread_list_does_not_crash(self, db):
        """Removing/renaming ongoing_threads in the YAML must reconcile
        cleanly, not orphan state or crash on a stale reference."""
        llm1 = _FakeLLM([_basic_response("thread A")])
        asyncio.run(DayLife(db, llm1, lambda: _persona())._generate("2026-08-01"))

        llm2 = _FakeLLM([_basic_response("a brand new thread")])
        new_persona = _persona(ongoing_threads=["a brand new thread"])
        life2 = DayLife(db, llm2, lambda: new_persona)
        state = asyncio.run(life2._generate("2026-08-02"))
        assert state["mood"] == "content"  # did not raise


class TestTrendWiring:
    def test_first_ever_call_has_no_previous_snapshot_so_trend_is_steady(self, db):
        life = DayLife(db, _FakeLLM([_basic_response()]), lambda: _persona())
        trend = life._compute_trend()
        assert trend == "steady"

    def test_trend_reflects_a_real_affection_rise_on_the_next_call(self, db):
        life = DayLife(db, _FakeLLM([_basic_response()]), lambda: _persona())
        life._compute_trend()  # snapshots current affection (5.0 default)
        db.update_relationship(affection=15.0)
        assert life._compute_trend() == "warming up"

    def test_trend_is_included_in_the_generation_prompt(self, db):
        llm = _FakeLLM([_basic_response()])
        life = DayLife(db, llm, lambda: _persona())
        db.update_relationship(affection=90.0)
        asyncio.run(life._generate(DAY1))
        assert "Relationship mood trend:" in llm.prompts[0]
