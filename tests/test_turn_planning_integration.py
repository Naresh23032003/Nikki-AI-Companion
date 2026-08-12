"""Integration tests for N3: the turn planner wired into `_build_prompt`, and
the AI-tell detector wired into `_guarded_reply` as a live regeneration guard.

`app/conversation/planner.py` and `app/conversation/tells.py` already had 59
passing unit tests before this change (see tests/test_conversation_engine.py)
- what those tests could NOT cover is whether `app/main.py` actually calls
them correctly. That is what this file pins down, with a fake LLM so no
Ollama/GPU is required.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

_SCRATCH = Path(tempfile.mkdtemp(prefix="nikki_turnplan_test_"))
os.environ.setdefault("COMPANION_DB_PATH", str(_SCRATCH / "companion.db"))
os.environ.setdefault("CHROMA_PATH", str(_SCRATCH / "chroma"))

from fastapi.testclient import TestClient  # noqa: E402


class _FakeLLM:
    """Replaces state.llm for these tests. Returns replies from a queue so a
    test can script exactly what the "model" says on each successive call.

    Installed for the WHOLE module (not just per-test) so the real
    OllamaClient's httpx.AsyncClient is never constructed against Ollama in
    the first place - it would otherwise open real connections during
    lifespan's warmup, bound to TestClient's own portal event loop, which
    then fight with the separate short-lived loops asyncio.run() creates per
    test for `state.llm.close()` at teardown ("Event loop is closed").
    """

    def __init__(self, replies=("ok.",)):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, **kw):
        self.calls.append(messages)
        return self.replies.pop(0) if self.replies else "ok."

    async def close(self):
        pass


@pytest.fixture(scope="module")
def booted():
    """Boot the real app once (lifespan runs, services wired), yield the
    app.main module itself so tests can reach state/_guarded_reply/etc."""
    saved = {k: os.environ.pop(k, None)
             for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                       "http_proxy", "https_proxy", "all_proxy")}
    import app.main as main
    with TestClient(main.app):
        main.state.llm = _FakeLLM()
        yield main
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


@pytest.fixture
def fake_llm(booted, monkeypatch):
    def _install(replies):
        fake = _FakeLLM(replies)
        monkeypatch.setattr(booted.state, "llm", fake)
        return fake
    return _install


# =========================================================== turn directive


class TestTurnDirectiveWiring:
    def test_turn_directive_is_reachable_and_returns_a_string(self, booted):
        """The exact bug this pins: N3 was wired into main.py with a missing
        import, so this call raised NameError and was silently swallowed by
        the try/except - always returning None. It must actually plan now."""
        result = booted._turn_directive("test-session-td", "ok", [], "chat")
        assert isinstance(result, str)
        assert "Do NOT ask a question" in result  # "ok" -> minimal turn

    def test_turn_directive_degrades_to_none_on_failure(self, booted, monkeypatch):
        def boom(_signals):
            raise RuntimeError("planner exploded")
        monkeypatch.setattr(booted, "plan_turn", boom)
        assert booted._turn_directive("test-session-td2", "hello there", [], "chat") is None

    def test_build_prompt_includes_the_turn_directive(self, booted):
        messages = asyncio.run(booted._build_prompt("test-session-bp", "ok"))
        system = messages[0]["content"]
        assert "Do NOT ask a question" in system

    def test_recent_assistant_replies_reads_history(self, booted):
        # No history yet for a fresh session -> empty, not an exception.
        assert booted._recent_assistant_replies("brand-new-session-xyz") == []


# =============================================================== tell guard


class TestTellGuardWiring:
    def test_a_tell_ridden_reply_triggers_one_regeneration(self, booted, fake_llm):
        # First draft: over_agreement + generic_question. Clean under every
        # OTHER guard (no claims, no honeypot, no refusal, no identity
        # confusion) so this isolates the new tell-detection path.
        fake = fake_llm(["That's so valid, honestly. How does that make you feel?",
                         "ha, fair enough."])
        reply, _ = asyncio.run(booted._guarded_reply(
            [{"role": "system", "content": "x"}], tool_ran=False,
            session_id="tell-test-session", user_message="i'm tired"))
        assert len(fake.calls) == 2, "tells should trigger exactly one retry"
        assert reply == "ha, fair enough."
        correction = fake.calls[1][-1]["content"]
        assert "CORRECTION NOTE" in correction
        assert "valid" in correction.lower() or "generic" in correction.lower()

    def test_a_clean_reply_does_not_regenerate(self, booted, fake_llm):
        fake = fake_llm(["ha, fair enough."])
        reply, _ = asyncio.run(booted._guarded_reply(
            [{"role": "system", "content": "x"}], tool_ran=False,
            session_id="tell-test-session-2", user_message="i'm tired"))
        assert len(fake.calls) == 1
        assert reply == "ha, fair enough."

    def test_omitting_session_id_skips_tell_detection(self, booted, fake_llm):
        """Pre-N3 call sites don't pass session_id/user_message - must keep
        working exactly as before (no tell checking, no behaviour change)."""
        fake = fake_llm(["That's so valid, honestly. How does that make you feel?"])
        reply, _ = asyncio.run(booted._guarded_reply(
            [{"role": "system", "content": "x"}], tool_ran=False))
        assert len(fake.calls) == 1, "no session_id -> tells must not gate regeneration"
        assert reply == "That's so valid, honestly. How does that make you feel?"
