"""Characterization tests for the HTTP surface.

Written BEFORE the route extraction (N2 phase 2) to pin the existing contract —
paths, status codes and payload shapes — so the move into APIRouters is provably
behaviour-preserving rather than merely plausible.

The app is booted in-process with a temporary database and no Ollama. Anything
requiring generation is out of scope here; this covers the CRUD surface, which
is what moves.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

# Redirect persistent stores to local scratch space BEFORE app.config is
# imported (it reads these at load_settings time). Two reasons: the suite must
# never touch the real companion.db sitting in the repo root, and ChromaDB's
# SQLite backend cannot open a file on a network/9p mount.
_SCRATCH = Path(tempfile.mkdtemp(prefix="nikki_api_test_"))
os.environ.setdefault("COMPANION_DB_PATH", str(_SCRATCH / "companion.db"))
os.environ.setdefault("CHROMA_PATH", str(_SCRATCH / "chroma"))

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Boot the real app once. Lifespan runs, so services are wired as in prod."""
    import os

    import app.main as main
    from app.db import Database

    # TestClient talks to the app in-process, but httpx still reads ambient
    # proxy environment variables and will try to route through them. Strip
    # them for the duration so the suite is not hostage to the shell it runs in.
    saved = {k: os.environ.pop(k, None)
             for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                       "http_proxy", "https_proxy", "all_proxy")}

    # TestClient's client host is "testclient", not 127.0.0.1, so the LAN auth
    # middleware treats it as a remote caller. Send the configured token when
    # one is set; when it isn't, auth is a no-op and the header is ignored.
    headers = {}
    if getattr(main, "_AUTH_TOKEN", ""):
        headers["x-auth-token"] = main._AUTH_TOKEN

    tmp = tmp_path_factory.mktemp("api")
    with TestClient(main.app, headers=headers) as c:
        # Point the default profile at a scratch DB so tests never touch the
        # real companion.db sitting in the repo root.
        scratch = Database(tmp / "api_test.db")
        profile = main.P()
        if profile is not None:
            profile.db = scratch
        main.state.db = scratch
        yield c

    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


def _ok(resp):
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:300]}"
    return resp.json()


class TestPersonaRoutes:
    def test_get_persona(self, client):
        body = _ok(client.get("/persona"))
        assert isinstance(body, dict)

    def test_list_personas(self, client):
        body = _ok(client.get("/personas"))
        assert isinstance(body, (list, dict))

    def test_unknown_persona_photo_is_not_a_500(self, client):
        assert client.get("/personas/definitely-not-real/photo").status_code in (200, 404)


class TestMemoryRoutes:
    def test_list_memories(self, client):
        assert isinstance(_ok(client.get("/memories")), (list, dict))

    def test_create_memory_returns_201(self, client):
        resp = client.post("/memories", json={"fact": "User likes filter coffee",
                                              "category": "preference"})
        assert resp.status_code == 201, resp.text[:300]

    def test_created_memory_appears_in_the_list(self, client):
        client.post("/memories", json={"fact": "User's cat is called Pepper",
                                       "category": "relationship"})
        body = _ok(client.get("/memories"))
        rows = body if isinstance(body, list) else body.get("memories", [])
        assert any("Pepper" in str(r) for r in rows)

    def test_delete_unknown_memory_is_handled(self, client):
        assert client.delete("/memories/999999").status_code in (200, 204, 404)

    def test_complete_unknown_memory_is_handled(self, client):
        assert client.post("/memories/999999/complete").status_code in (200, 204, 404)


class TestJournalRoutes:
    def test_list_journal(self, client):
        assert isinstance(_ok(client.get("/journal")), (list, dict))

    def test_journal_accepts_filters(self, client):
        assert client.get("/journal", params={"since": "2020-01-01"}).status_code == 200

    def test_edit_unknown_entry_is_handled(self, client):
        resp = client.put("/journal/999999", json={"mood": "ok", "text": "x"})
        assert resp.status_code in (200, 404, 422)

    def test_delete_unknown_entry_is_handled(self, client):
        assert client.delete("/journal/999999").status_code in (200, 204, 404)


class TestHistoryRoutes:
    def test_get_history_for_a_fresh_session(self, client):
        body = _ok(client.get("/history/test-session"))
        assert isinstance(body, (list, dict))

    def test_clear_history(self, client):
        assert client.delete("/history/test-session").status_code in (200, 204)


class TestMemoryGraphRoutes:
    def test_graph_data(self, client):
        body = _ok(client.get("/memory-graph/data"))
        assert isinstance(body, dict)
        assert "entities" in body or "nodes" in body

    def test_graph_page_renders(self, client):
        assert client.get("/memory-graph").status_code == 200

    def test_delete_unknown_entity_is_handled(self, client):
        assert client.delete("/memory-graph/entities/999999").status_code in (200, 204, 404)

    def test_delete_unknown_relation_is_handled(self, client):
        assert client.delete("/memory-graph/relations/999999").status_code in (200, 204, 404)


class TestDeviceRoutes:
    def test_devices_status(self, client):
        assert isinstance(_ok(client.get("/devices/status")), (list, dict))


class TestProactiveRoutes:
    """Characterization tests written BEFORE extraction (N2 phase 2, round 2)."""

    def test_proactive_status(self, client):
        body = _ok(client.get("/proactive/status"))
        assert "running" in body

    def test_proactive_pause(self, client):
        body = _ok(client.post("/proactive/pause", json={"hours": 1}))
        assert "paused_until" in body

    def test_proactive_pause_rejects_out_of_range(self, client):
        assert client.post("/proactive/pause", json={"hours": 999}).status_code == 422

    def test_proactive_trigger_unknown_profile_is_404(self, client):
        assert client.post("/proactive/trigger/not-a-real-profile").status_code == 404


class TestRelationshipRoutes:
    def test_get_relationship(self, client):
        body = _ok(client.get("/relationship"))
        assert isinstance(body, dict)

    def test_override_rejects_bad_stage(self, client):
        resp = client.post("/relationship/override", json={"stage": "not-a-real-stage"})
        assert resp.status_code == 400

    def test_override_accepts_affection_only(self, client):
        body = _ok(client.post("/relationship/override", json={"affection": 42}))
        assert isinstance(body, dict)


class TestVoiceRoutes:
    def test_voice_status(self, client):
        body = _ok(client.get("/voice/status"))
        assert "rvc_ready" in body and "studio_installed" in body


class TestDashboardRoutes:
    def test_day_state(self, client):
        body = _ok(client.get("/day-state"))
        assert isinstance(body, dict)

    def test_brain_status(self, client):
        body = _ok(client.get("/brain/status"))
        assert "routing" in body and "guards" in body

    def test_health(self, client):
        body = _ok(client.get("/health"))
        assert body["status"] == "ok"
        assert "ollama_reachable" in body


class TestRouteInventory:
    """The registered path set must not change across the extraction."""

    EXPECTED = {
        "/persona", "/personas", "/personas/active", "/personas/{persona_id}/photo",
        "/persona/photo", "/memories", "/memories/{memory_id}",
        "/memories/{memory_id}/complete", "/journal", "/journal/{entry_id}",
        "/journal/run-now", "/journal/run-weekly-now",
        "/history/{session_id}", "/memory-graph", "/memory-graph/data",
        "/memory-graph/entities/{entity_id}", "/memory-graph/relations/{relation_id}",
        "/devices/status", "/chat", "/stt", "/tts",
        "/proactive/pause", "/proactive/status", "/proactive/trigger/{profile_id}",
        "/relationship", "/relationship/override",
        "/voice/bench", "/voice/status",
        "/day-state", "/day-state/regenerate", "/brain/status", "/health",
        "/privacy/export", "/privacy/delete-all",
        "/telephony/incoming-call",
    }

    def test_all_expected_paths_are_registered(self):
        import app.main as main

        registered = {getattr(r, "path", None) for r in main.app.routes}
        missing = self.EXPECTED - registered
        assert not missing, f"routes disappeared: {sorted(missing)}"
