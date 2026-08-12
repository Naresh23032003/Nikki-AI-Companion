"""Schema-level tests for memory scoring, supersession, revisions and FTS5.

Uses a real SQLite file in tmp_path — no Ollama, no ChromaDB, no network.
"""
from __future__ import annotations

import sqlite3

import pytest

from app.db import Database


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    yield d
    try:
        d.close()
    except Exception:
        pass


def _cols(db, table="memories"):
    return {r["name"] for r in db._conn.execute(f"PRAGMA table_info({table})")}


class TestMigration:
    def test_scoring_columns_exist(self, db):
        assert {
            "importance", "confidence", "superseded_by", "superseded_at",
            "t_invalid", "reinforced_count",
        } <= _cols(db)

    def test_defaults_are_sane(self, db):
        mid = db.add_memory("User likes tea", "preference")
        row = db.get_memory(mid)
        assert row["importance"] == pytest.approx(0.5)
        assert row["confidence"] == pytest.approx(0.7)
        assert row["superseded_by"] is None
        assert row["reinforced_count"] == 0

    def test_revisions_table_exists(self, db):
        assert db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='memory_revisions'"
        ).fetchone() is not None

    def test_migration_is_idempotent(self, tmp_path):
        path = tmp_path / "twice.db"
        a = Database(path)
        mid = a.add_memory("User likes tea", "preference")
        a.close()
        b = Database(path)  # re-open runs the migration again
        assert b.get_memory(mid)["fact"] == "User likes tea"
        b.close()

    def test_upgrades_a_pre_existing_database(self, tmp_path):
        """A companion.db written before this change must survive the upgrade."""
        path = tmp_path / "legacy.db"
        raw = sqlite3.connect(path)
        raw.executescript(
            """
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL,
                category TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed TEXT,
                access_count INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO memories (fact, category, created_at, access_count)
            VALUES ('User birthday is March 3rd', 'personal_info', '2026-01-01T00:00:00+00:00', 4);
            """
        )
        raw.commit()
        raw.close()

        db = Database(path)
        rows = db.list_memories()
        assert len(rows) == 1
        assert rows[0]["fact"] == "User birthday is March 3rd"
        assert rows[0]["importance"] == pytest.approx(0.5)  # backfilled
        assert rows[0]["access_count"] == 4                 # preserved
        db.close()


class TestSupersession:
    def test_supersede_preserves_the_old_fact(self, db):
        old = db.add_memory("User works at Google", "personal_info")
        new = db.add_memory("User works at Stripe", "personal_info")
        db.supersede_memory(old, new, reason="employer changed")

        old_row = db.get_memory(old)
        assert old_row is not None, "superseded memory must not be deleted"
        assert old_row["fact"] == "User works at Google"
        assert old_row["superseded_by"] == new
        assert old_row["t_invalid"] is not None

    def test_superseded_memory_is_excluded_from_active(self, db):
        old = db.add_memory("User works at Google", "personal_info")
        new = db.add_memory("User works at Stripe", "personal_info")
        db.supersede_memory(old, new)
        active_ids = {r["id"] for r in db.get_active_memories()}
        assert new in active_ids
        assert old not in active_ids

    def test_supersession_is_recorded_in_history(self, db):
        old = db.add_memory("User works at Google", "personal_info")
        new = db.add_memory("User works at Stripe", "personal_info")
        db.supersede_memory(old, new, reason="employer changed")
        history = db.get_memory_history(old)
        assert len(history) == 1
        assert history[0]["operation"] == "supersede"
        assert history[0]["previous_fact"] == "User works at Google"
        assert history[0]["new_fact"] == "User works at Stripe"
        assert history[0]["reason"] == "employer changed"

    def test_superseding_a_missing_memory_is_a_noop(self, db):
        db.supersede_memory(9999, 1)  # must not raise

    def test_reinforce_bumps_count_and_confidence(self, db):
        mid = db.add_memory("User dislikes coffee", "preference")
        db.reinforce_memory(mid, confidence=0.9)
        db.reinforce_memory(mid, confidence=0.95)
        row = db.get_memory(mid)
        assert row["reinforced_count"] == 2
        assert row["confidence"] == pytest.approx(0.95)

    def test_scores_are_clamped(self, db):
        mid = db.add_memory("x", "preference")
        db.set_memory_scores(mid, importance=5.0, confidence=-3.0)
        row = db.get_memory(mid)
        assert row["importance"] == 1.0
        assert row["confidence"] == 0.0


class TestLexicalSearch:
    def test_finds_an_exact_token_vector_search_might_miss(self, db):
        target = db.add_memory("User's sister is named Meera", "relationship")
        db.add_memory("User enjoys long walks", "preference")
        assert target in db.search_memories_lexical("what is my sister called")

    def test_ranks_the_match_first(self, db):
        db.add_memory("User enjoys long walks", "preference")
        target = db.add_memory("User's cat is called Pepper", "relationship")
        assert db.search_memories_lexical("Pepper")[0] == target

    def test_index_follows_updates(self, db):
        mid = db.add_memory("User has a cat", "relationship")
        db.update_memory(mid, "User has a cat named Pepper", "relationship")
        assert mid in db.search_memories_lexical("Pepper")

    def test_index_follows_deletes(self, db):
        mid = db.add_memory("User has a parrot named Kiwi", "relationship")
        db.delete_memory(mid)
        assert mid not in db.search_memories_lexical("Kiwi")

    def test_punctuation_does_not_break_the_query(self, db):
        """A natural question must not produce a malformed FTS5 MATCH."""
        db.add_memory("User's sister is named Meera", "relationship")
        for q in ["what's my sister's name?", "sister -- name!", '"quoted"', "()"]:
            db.search_memories_lexical(q)  # must not raise

    def test_empty_and_short_queries_are_safe(self, db):
        assert db.search_memories_lexical("") == []
        assert db.search_memories_lexical("a an") == []
