"""Tests for app/telephony.py (N9: telephony provider abstraction).

Everything here is testable WITHOUT a real Twilio account: the audio codec
(round-trips against synthetic samples), Twilio's PUBLISHED signature
algorithm (computed both ways and compared), TwiML generation (XML string
assertions), and the turn-endpointing buffer (synthetic energy patterns).
What is NOT testable without a live account, a real phone number and a
publicly reachable webhook URL: an actual phone call. See
IMPLEMENTATION_LOG.md for what remains for a human to verify that part.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.telephony import (
    TELEPHONY_SAMPLE_RATE,
    TwilioProvider,
    TwilioStreamBuffer,
    get_telephony_provider,
    pcm16_to_ulaw,
    resample_linear,
    ulaw_to_pcm16,
)


class TestMuLawCodec:
    def test_round_trip_preserves_signal_shape(self):
        t = np.linspace(0, 1, TELEPHONY_SAMPLE_RATE, dtype=np.float32)
        original = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        encoded = pcm16_to_ulaw(original)
        decoded = ulaw_to_pcm16(encoded)
        assert len(decoded) == len(original)
        # mu-law is lossy (8-bit companded) - correlation, not exact equality.
        correlation = np.corrcoef(original, decoded)[0, 1]
        assert correlation > 0.95

    def test_silence_round_trips_to_silence(self):
        silence = np.zeros(160, dtype=np.float32)
        decoded = ulaw_to_pcm16(pcm16_to_ulaw(silence))
        assert np.abs(decoded).max() < 0.01

    def test_encoded_output_is_half_the_byte_length(self):
        # mu-law is 1 byte/sample vs 2 bytes/sample for 16-bit PCM.
        samples = np.random.uniform(-0.5, 0.5, 320).astype(np.float32)
        encoded = pcm16_to_ulaw(samples)
        assert len(encoded) == len(samples)

    def test_clipping_does_not_raise(self):
        # Values outside [-1, 1] must be clamped, not wrap/overflow.
        samples = np.array([2.0, -3.0, 0.5], dtype=np.float32)
        encoded = pcm16_to_ulaw(samples)
        assert len(encoded) == 3


class TestResample:
    def test_same_rate_is_a_noop(self):
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        out = resample_linear(samples, 8000, 8000)
        assert np.array_equal(out, samples)

    def test_upsampling_produces_more_samples(self):
        samples = np.ones(100, dtype=np.float32)
        out = resample_linear(samples, 8000, 16000)
        assert len(out) == 200

    def test_empty_input(self):
        assert len(resample_linear(np.array([], dtype=np.float32), 8000, 16000)) == 0


class TestTwilioSignatureVerification:
    def _provider(self):
        return TwilioProvider("ACtest", "test_auth_token", "+15550001111")

    def test_valid_signature_is_accepted(self):
        import base64
        import hashlib
        import hmac as hmac_mod

        provider = self._provider()
        url = "https://example.com/telephony/incoming-call"
        params = {"CallSid": "CA123", "From": "+15551234567"}
        data = url + "".join(k + str(params[k]) for k in sorted(params))
        real_signature = base64.b64encode(
            hmac_mod.new(b"test_auth_token", data.encode(), hashlib.sha1).digest()
        ).decode()
        assert provider.verify_webhook_signature(url, params, real_signature) is True

    def test_forged_signature_is_rejected(self):
        provider = self._provider()
        assert provider.verify_webhook_signature(
            "https://example.com/telephony/incoming-call",
            {"CallSid": "CA123"}, "not-a-real-signature") is False

    def test_missing_signature_is_rejected(self):
        provider = self._provider()
        assert provider.verify_webhook_signature(
            "https://example.com/x", {}, None) is False

    def test_tampered_params_invalidate_a_previously_valid_signature(self):
        import base64
        import hashlib
        import hmac as hmac_mod

        provider = self._provider()
        url = "https://example.com/telephony/incoming-call"
        original_params = {"CallSid": "CA123"}
        data = url + "".join(k + str(original_params[k]) for k in sorted(original_params))
        signature = base64.b64encode(
            hmac_mod.new(b"test_auth_token", data.encode(), hashlib.sha1).digest()
        ).decode()
        tampered_params = {"CallSid": "CA999-attacker-controlled"}
        assert provider.verify_webhook_signature(url, tampered_params, signature) is False


class TestTwiMLGeneration:
    def test_answer_call_produces_valid_connect_stream(self):
        provider = TwilioProvider("ACtest", "tok", "+15550001111")
        twiml = provider.answer_call_twiml("wss://example.com/telephony/media-stream")
        assert "<Connect>" in twiml
        assert "<Stream" in twiml
        assert 'url="wss://example.com/telephony/media-stream"' in twiml

    def test_stream_url_is_xml_escaped(self):
        provider = TwilioProvider("ACtest", "tok", "+15550001111")
        twiml = provider.answer_call_twiml("wss://example.com/x?a=1&b=2")
        assert "&amp;" in twiml
        assert "&b=2" not in twiml  # raw & must not survive unescaped


class TestProviderFactory:
    def _settings(self, enabled=True, provider="twilio"):
        return SimpleNamespace(telephony={"enabled": enabled, "provider": provider})

    def test_disabled_returns_none(self):
        assert get_telephony_provider(self._settings(enabled=False)) is None

    def test_enabled_without_env_credentials_returns_none(self, monkeypatch):
        monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("TWILIO_PHONE_NUMBER", raising=False)
        assert get_telephony_provider(self._settings()) is None

    def test_enabled_with_credentials_returns_a_provider(self, monkeypatch):
        monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACtest")
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tok")
        monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+15550001111")
        provider = get_telephony_provider(self._settings())
        assert provider is not None
        assert provider.name == "twilio"

    def test_unknown_provider_name_returns_none(self, monkeypatch):
        monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACtest")
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tok")
        monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+15550001111")
        assert get_telephony_provider(self._settings(provider="acme_telecom")) is None


class TestStreamBufferEndpointing:
    def _silence(self, n=160):
        return np.zeros(n, dtype=np.float32)

    def _speech(self, n=160):
        return (np.random.uniform(-0.3, 0.3, n)).astype(np.float32)

    def test_no_flush_while_still_speaking(self):
        buf = TwilioStreamBuffer(silence_frames=5, min_speech_frames=2)
        for _ in range(8):
            buf.add_chunk(self._speech())
        assert buf.should_flush() is False

    def test_flushes_after_speech_then_enough_silence(self):
        buf = TwilioStreamBuffer(silence_frames=5, min_speech_frames=2)
        for _ in range(3):
            buf.add_chunk(self._speech())
        for _ in range(5):
            buf.add_chunk(self._silence())
        assert buf.should_flush() is True

    def test_pure_silence_never_flushes(self):
        """No speech at all (min_speech_frames never reached) must not
        trigger a turn - an open line with nobody talking isn't an utterance."""
        buf = TwilioStreamBuffer(silence_frames=3, min_speech_frames=5)
        for _ in range(50):
            buf.add_chunk(self._silence())
        assert buf.should_flush() is False

    def test_hard_cap_flushes_a_stuck_open_mic(self):
        buf = TwilioStreamBuffer(max_buffer_frames=10)
        for _ in range(11):
            buf.add_chunk(self._speech())
        assert buf.should_flush() is True

    def test_flush_resets_state(self):
        buf = TwilioStreamBuffer(silence_frames=2, min_speech_frames=1)
        buf.add_chunk(self._speech())
        buf.add_chunk(self._silence())
        buf.add_chunk(self._silence())
        assert buf.should_flush() is True
        out = buf.flush()
        assert len(out) > 0
        assert buf.is_empty()
        assert buf.should_flush() is False

    def test_empty_buffer_never_flushes(self):
        assert TwilioStreamBuffer().should_flush() is False


# ================================================================ HTTP routes

import os  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("fastapi")

_SCRATCH = Path(tempfile.mkdtemp(prefix="nikki_telephony_route_test_"))
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
    with TestClient(main.app, headers=headers) as c:
        yield c
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


class TestIncomingCallWebhook:
    def test_not_configured_returns_503(self, client, monkeypatch):
        # Telephony is off by default (no config.yaml `telephony:` block in
        # the test fixture's settings, and no TWILIO_* env vars) - the
        # webhook must degrade cleanly, never 500.
        resp = client.post("/telephony/incoming-call", data={"CallSid": "CA123"})
        assert resp.status_code == 503

    def test_configured_but_unsigned_request_is_rejected(self, client, monkeypatch):
        monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACtest")
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test_token")
        monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+15550001111")
        import app.main as main
        monkeypatch.setitem(main.state.settings.telephony, "enabled", True)
        monkeypatch.setitem(main.state.settings.telephony, "provider", "twilio")
        resp = client.post("/telephony/incoming-call", data={"CallSid": "CA123"})
        assert resp.status_code == 403

    def test_media_stream_closes_immediately_when_not_configured(self, client):
        with pytest.raises(Exception):  # noqa: B017 - TestClient raises on the 4404 close
            with client.websocket_connect("/telephony/media-stream"):
                pass
