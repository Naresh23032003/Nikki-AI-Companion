"""Tests for N10: data export + erasure.

app.db.Database.export_all_data/delete_all_data (real scratch SQLite),
app.memory.MemoryStore.wipe_all (real scratch ChromaDB, no Ollama needed -
vectors are added directly rather than via embedding text), and the
/privacy/* HTTP routes (real app boot, TestClient).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.db import Database

# ================================================================ Database


@pytest.fixture
def db():
    tmp = tempfile.mkdtemp(prefix="nikki_privacy_test_")
    database = Database(Path(tmp) / "test.db")
    yield database
    database.close()


def _populate(db):
    db.ensure_session("main")
    db.add_message("main", "user", "hi", source="webapp_chat")
    db.add_message("main", "assistant", "hey!", source="webapp_chat")
    mid = db.add_memory("User likes filter coffee", "preference")
    db.add_mood_entry("2026-08-01", "12:00", "happy", 4, "said work went well", "webapp_chat")
    db.add_reminder("water the plants", "2026-08-10T12:00:00")
    db.add_event_followup(mid, "exam on Friday", "2026-08-14T09:00:00",
                          "2026-08-13T09:00:00", "2026-08-14T18:00:00", "main")
    db.get_relationship()  # auto-creates the row; update_relationship() alone won't
    db.update_relationship(affection=42.0, stage="close", days_known=30)
    return mid


class TestExportAllData:
    def test_export_includes_every_data_kind(self, db):
        _populate(db)
        export = db.export_all_data()
        assert len(export["messages"]) == 2
        assert len(export["memories"]) == 1
        assert export["memories"][0]["fact"] == "User likes filter coffee"
        assert len(export["mood_journal"]) == 1
        assert len(export["reminders"]) == 1
        assert len(export["event_followups"]) == 1
        assert export["relationship"]["affection"] == 42.0
        assert export["relationship"]["stage"] == "close"
        assert "exported_at" in export

    def test_export_on_a_fresh_db_is_empty_not_an_error(self, db):
        export = db.export_all_data()
        assert export["messages"] == []
        assert export["memories"] == []
        assert export["relationship"]  # row auto-created, but zeroed


class TestDeleteAllData:
    def test_wipes_every_table(self, db):
        _populate(db)
        db.delete_all_data()
        export = db.export_all_data()
        assert export["messages"] == []
        assert export["memories"] == []
        assert export["mood_journal"] == []
        assert export["reminders"] == []
        assert export["event_followups"] == []

    def test_resets_relationship_to_a_fresh_start(self, db):
        _populate(db)
        db.delete_all_data()
        rel = db.get_relationship()
        assert rel["stage"] == "stranger"
        assert rel["affection"] == 5.0
        assert rel["days_known"] == 0

    def test_idempotent_on_an_already_empty_db(self, db):
        db.delete_all_data()  # must not raise on nothing-to-delete
        db.delete_all_data()
        assert db.export_all_data()["messages"] == []


# ============================================================== MemoryStore


class TestWipeAllVectors:
    def _memory_store(self, tmp_path):
        import app.memory as memory_mod

        db = Database(tmp_path / "wipe_test.db")
        llm = SimpleNamespace()  # never called - vectors added directly below
        settings = SimpleNamespace(
            wa_session_id="main",
            chroma_path=tmp_path / "chroma",
            memory_collection="wipe_test_collection",
            memory_dedup_threshold=0.92,
        )
        store = memory_mod.MemoryStore(db, llm, settings)
        return db, store

    def test_wipe_all_removes_every_vector(self, tmp_path):
        db, store = self._memory_store(tmp_path)
        try:
            # Add vectors directly (no Ollama needed - fixed-length fake embeddings).
            store._collection.add(
                ids=["1", "2", "3"],
                embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                documents=["a", "b", "c"],
            )
            assert store._collection.count() == 3
            removed = store.wipe_all()
            assert removed == 3
            assert store._collection.count() == 0
        finally:
            db.close()

    def test_wipe_all_on_an_empty_collection_is_a_noop(self, tmp_path):
        db, store = self._memory_store(tmp_path)
        try:
            assert store.wipe_all() == 0
        finally:
            db.close()


# ================================================================ HTTP routes

pytest.importorskip("fastapi")

_SCRATCH = Path(tempfile.mkdtemp(prefix="nikki_privacy_api_test_"))
os.environ.setdefault("COMPANION_DB_PATH", str(_SCRATCH / "companion.db"))
os.environ.setdefault("CHROMA_PATH", str(_SCRATCH / "chroma"))

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client():
    import app.main as main

    saved = {k: os.environ.pop(k, None)
             for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                       "http_proxy", "https_proxy", "all_proxy")}
    headers = {}
    if getattr(main, "_AUTH_TOKEN", ""):
        headers["x-auth-token"] = main._AUTH_TOKEN
    tmp = tempfile.mkdtemp(prefix="nikki_privacy_route_test_")
    with TestClient(main.app, headers=headers) as c:
        scratch = Database(Path(tmp) / "route_test.db")
        profile = main.P()
        if profile is not None:
            profile.db = scratch
        main.state.db = scratch
        yield c
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


class TestPrivacyRoutes:
    def test_export_returns_the_full_shape(self, client):
        resp = client.get("/privacy/export")
        assert resp.status_code == 200, resp.text[:300]
        body = resp.json()
        for key in ("messages", "memories", "mood_journal", "relationship"):
            assert key in body

    def test_delete_all_without_confirm_is_rejected(self, client):
        resp = client.post("/privacy/delete-all", json={})
        assert resp.status_code == 400
        resp2 = client.post("/privacy/delete-all", json={"confirm": False})
        assert resp2.status_code == 400

    def test_delete_all_with_confirm_wipes_and_reports_result(self, client):
        client.post("/memories", json={"fact": "test fact for wipe", "category": "preference"})
        resp = client.post("/privacy/delete-all", json={"confirm": True})
        assert resp.status_code == 200, resp.text[:300]
        body = resp.json()
        assert body["deleted"] is True
        assert "vectors_removed" in body
        # Verify the wipe actually took: nothing left to export.
        export = client.get("/privacy/export").json()
        assert export["memories"] == []
