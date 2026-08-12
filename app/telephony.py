"""Telephony: a real, dialable phone number for her (N9).

Provider-abstracted (config `telephony.provider`, credentials in `.env`,
never in the tracked config.yaml - see app/profiles.py's N10 fix for why)
so a different provider can be swapped in without touching call-handling
logic. Architecture deliberately mirrors app/providers.py's CloudBrain
multi-provider pattern: config-driven selection, credentials only from env,
never silently trusted without verification.

Twilio Voice + Media Streams is the current (2026) standard shape for
bridging a real phone call to a custom STT/TTS pipeline (confirmed via
Twilio's own current documentation and sample code): Twilio answers the PSTN
call, POSTs a webhook we respond to with TwiML, then opens a WebSocket
("Media Stream") carrying raw audio to our server in 20ms mu-law/8kHz
frames. That bridges into the SAME turn-handling architecture
app/main.py's `/ws/call` already runs for browser-based Call mode - just
with mu-law/8kHz framing instead of the browser's webm/opus, and Twilio's
own start/media/stop control-message protocol instead of a direct
connection.

NOT LIVE-VERIFIED. This needs a real Twilio account, a purchased phone
number, and a PUBLICLY REACHABLE HTTPS/WSS URL for the webhook (this app is
LAN-only by design) - none of which exist in this environment. See
IMPLEMENTATION_LOG.md for the exact remaining steps to go live.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
from abc import ABC, abstractmethod
from urllib.parse import urlencode
from xml.sax.saxutils import escape

import numpy as np

logger = logging.getLogger("companion.telephony")

TELEPHONY_SAMPLE_RATE = 8000  # Twilio Media Streams: mu-law, 8kHz, mono


# ---------------------------------------------------------------------------
# Audio codec: G.711 mu-law <-> 16-bit linear PCM
# ---------------------------------------------------------------------------
# Uses the stdlib `audioop` module (bit-exact G.711, battle-tested) rather
# than a hand-rolled reimplementation: a subtly wrong hand-rolled codec would
# be effectively undetectable without a live call to test against, and
# `audioop` is the one part of this that's actually been exercised against
# real G.711 streams for decades. Known migration risk: `audioop` is
# deprecated and slated for removal in Python 3.13 (this project runs 3.12).
# If/when this project upgrades, the drop-in fix is the `audioop-lts` PyPI
# backport (same stdlib C implementation, published for exactly this
# migration) - not a rewrite.
import audioop  # noqa: E402


def ulaw_to_pcm16(mulaw_bytes: bytes) -> np.ndarray:
    """Decode mu-law bytes (Twilio's wire format) to float32 PCM in [-1, 1]."""
    pcm16 = audioop.ulaw2lin(mulaw_bytes, 2)  # 2 = output sample width (bytes)
    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def pcm16_to_ulaw(samples: np.ndarray) -> bytes:
    """Encode float32 PCM in [-1, 1] to mu-law bytes for Twilio."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16).tobytes()
    return audioop.lin2ulaw(pcm16, 2)


def resample_linear(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Cheap linear resample - good enough for 8kHz telephony <-> Kokoro's
    24kHz/faster-whisper's 16kHz; a full sinc resampler is overkill for
    voice-band telephony audio and would add a new dependency for no
    audible benefit at this bandwidth."""
    if from_rate == to_rate or len(samples) == 0:
        return samples
    duration = len(samples) / from_rate
    target_len = max(1, int(round(duration * to_rate)))
    return np.interp(
        np.linspace(0, len(samples) - 1, target_len),
        np.arange(len(samples)), samples,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------


class TelephonyProvider(ABC):
    name: str = "base"

    @abstractmethod
    def verify_webhook_signature(self, url: str, params: dict,
                                 signature: str | None) -> bool:
        """True if an incoming webhook genuinely came from the provider."""

    @abstractmethod
    def answer_call_twiml(self, stream_url: str) -> str:
        """TwiML/equivalent response telling the provider to open a media
        stream to `stream_url` for the rest of the call."""


class TwilioProvider(TelephonyProvider):
    name = "twilio"

    def __init__(self, account_sid: str, auth_token: str, phone_number: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.phone_number = phone_number

    def verify_webhook_signature(self, url: str, params: dict,
                                 signature: str | None) -> bool:
        """Twilio's documented request-validation algorithm: HMAC-SHA1 of the
        full URL + sorted-and-concatenated POST param key/value pairs, using
        the auth token as the key, base64-encoded, compared to the
        X-Twilio-Signature header. See Twilio's "Validating Requests" docs -
        this is their published algorithm, not a guess."""
        if not signature:
            return False
        data = url
        for key in sorted(params.keys()):
            data += key + str(params[key])
        computed = base64.b64encode(
            hmac.new(self.auth_token.encode("utf-8"),
                     data.encode("utf-8"), hashlib.sha1).digest()
        ).decode("utf-8")
        return hmac.compare_digest(computed, signature)

    def answer_call_twiml(self, stream_url: str) -> str:
        # <Connect><Stream> hands the whole call over to the Media Stream
        # WebSocket for full-duplex audio - the shape Twilio's own current
        # docs use for this exact "bridge to a custom voice pipeline" case,
        # as opposed to <Start><Stream> (one-way, call stays on TwiML).
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            "<Connect>"
            f'<Stream url="{escape(stream_url)}" />'
            "</Connect>"
            "</Response>"
        )

    def signed_request_url(self, base_url: str, params: dict | None = None) -> str:
        """Helper for tests/tools: the exact URL Twilio would sign against."""
        if not params:
            return base_url
        return f"{base_url}?{urlencode(sorted(params.items()))}"


# ---------------------------------------------------------------------------
# Turn endpointing: Twilio streams continuous audio with no "they stopped
# talking" event (unlike the browser Call mode client, which sends discrete
# utterance blobs) - something has to decide when a turn is over.
# ---------------------------------------------------------------------------


class TwilioStreamBuffer:
    """Accumulates decoded PCM from Twilio's 20ms mu-law frames and decides
    when a user turn has ended, via a simple RMS-energy silence gate.

    Deliberately NOT WebRTC VAD or anything ML-based: this is a stopgap
    endpointer, easy to reason about and unit-test without audio fixtures,
    matching the accuracy this stage actually needs - faster-whisper's own
    `vad_filter=True` (already used in app/stt.py) does the real speech/noise
    discrimination once a candidate turn is handed to it; this layer only
    decides WHEN to hand one over.
    """

    def __init__(self, silence_frames: int = 25, min_speech_frames: int = 10,
                energy_floor: float = 0.01, max_buffer_frames: int = 750):
        # Twilio frames are 20ms each: 25 frames of silence = 0.5s, a natural
        # end-of-utterance pause; max_buffer_frames=750 = 15s hard cap so a
        # stuck-open mic can never block a turn forever.
        self.silence_frames = silence_frames
        self.min_speech_frames = min_speech_frames
        self.energy_floor = energy_floor
        self.max_buffer_frames = max_buffer_frames
        self._chunks: list[np.ndarray] = []
        self._speech_frames = 0
        self._trailing_silence = 0

    def add_chunk(self, samples: np.ndarray) -> None:
        self._chunks.append(samples)
        rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
        if rms >= self.energy_floor:
            self._speech_frames += 1
            self._trailing_silence = 0
        else:
            self._trailing_silence += 1

    def should_flush(self) -> bool:
        if not self._chunks:
            return False
        if len(self._chunks) >= self.max_buffer_frames:
            return True
        return (self._speech_frames >= self.min_speech_frames
                and self._trailing_silence >= self.silence_frames)

    def flush(self) -> np.ndarray:
        out = (np.concatenate(self._chunks) if self._chunks
               else np.zeros(0, dtype=np.float32))
        self._chunks = []
        self._speech_frames = 0
        self._trailing_silence = 0
        return out

    def is_empty(self) -> bool:
        return not self._chunks


def get_telephony_provider(settings) -> TelephonyProvider | None:
    """Build the configured provider, or None if telephony isn't configured -
    callers must degrade gracefully (matching the `available` pattern already
    used by app.stt/app.tts/app.rvc_layer for optional, credential-gated
    tiers), never crash the app over a missing phone feature."""
    import os

    cfg = getattr(settings, "telephony", {}) or {}
    if not cfg.get("enabled", False):
        return None
    provider_name = (cfg.get("provider") or "twilio").lower()
    if provider_name != "twilio":
        logger.warning("telephony: unknown provider %r", provider_name)
        return None
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    number = os.environ.get("TWILIO_PHONE_NUMBER", "")
    if not (sid and token and number):
        logger.info("telephony: enabled in config but TWILIO_* env vars "
                    "not set - see .env.example")
        return None
    return TwilioProvider(sid, token, number)
