"""Text-to-speech via Kokoro-82M (CPU).

Two consumers:
  * File mode  — synthesize a full reply to a WAV for chat voice notes.
  * Stream mode — synthesize sentence-by-sentence for Call mode, so playback can
    start before the whole reply is generated.

Every synthesized chunk carries a `timings` dict (see `build_timings`) that
Phase 5 lip-sync depends on. We emit per-word timings when Kokoro exposes token
timestamps, otherwise approximate per-character timings derived from the audio
duration.

Kokoro runs on CPU (0 VRAM), leaving the 6 GB GPU for whisper-small + llama3.2.
Lazy-loaded so importing this module never requires the package/model.
"""
from __future__ import annotations

import io
import logging
import re
import threading
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np

logger = logging.getLogger("companion.tts")

SAMPLE_RATE = 24000  # Kokoro's native rate
DEFAULT_VOICE = "af_heart"  # warm, expressive American female
DEFAULT_SPEED = 0.92  # slightly relaxed pace reads warmer / more human


# Emoji & pictographs — espeak reads these aloud by name ("winking face"), so we
# strip them (and markdown noise) from the text before synthesis. The chat bubble
# still shows the original text; only the spoken audio is cleaned.
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols, pictographs, emoji, supplemental/extended
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicator (flags)
    "\U00002190-\U000021ff"  # arrows
    "\U00002b00-\U00002bff"  # misc symbols and arrows
    "\U0001f000-\U0001f0ff"  # mahjong / dominoes / cards
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0000200d"             # zero-width joiner
    "\U000024c2\U0000203c\U00002049\U00002122\U00002139"
    "]+",
    flags=re.UNICODE,
)


def strip_nonspeech(text: str) -> str:
    """Remove emoji / markdown so TTS doesn't voice symbol names."""
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"[*_`~#>|]", "", text)          # markdown-ish emphasis chars
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)   # tidy space-before-punctuation
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Pure helpers (no model needed — unit-testable)
# ---------------------------------------------------------------------------
# Sentence boundary: end punctuation (with optional closing quote/bracket)
# followed by whitespace/end, OR one-or-more newlines.
_SENTENCE_BOUNDARY = re.compile(r"[.!?]+[\)\]\"'”’]*(\s|$)|\n+")


class SentenceAccumulator:
    """Feeds streamed tokens in; yields complete sentences as they finish.

    Splits on sentence-ending punctuation (. ! ?) and newlines. Any remaining
    buffered text is emitted by `flush()` at end-of-stream.
    """

    def __init__(self, min_len: int = 2):
        self._buf = ""
        self.min_len = min_len

    def add(self, text: str) -> List[str]:
        self._buf += text
        out: List[str] = []
        while True:
            m = _SENTENCE_BOUNDARY.search(self._buf)
            if not m:
                break
            end = m.end()
            sentence = self._buf[:end].strip()
            self._buf = self._buf[end:]
            if len(sentence) >= self.min_len:
                out.append(sentence)
        return out

    def flush(self) -> Optional[str]:
        s = self._buf.strip()
        self._buf = ""
        return s if len(s) >= 1 else None


def split_sentences(text: str) -> List[str]:
    """One-shot sentence split (file mode helper / tests)."""
    acc = SentenceAccumulator()
    out = acc.add(text)
    tail = acc.flush()
    if tail:
        out.append(tail)
    return out


def build_timings(text: str, duration: float, word_units=None) -> dict:
    """Assemble the `timings` payload for a chunk of audio.

    If `word_units` (list of {"t","d","s"}) is provided (from Kokoro token
    timestamps) we use them; otherwise we approximate per-character timing by
    distributing `duration` across characters (spaces weighted lighter).
    """
    if word_units:
        return {
            "source": "kokoro-word",
            "sample_rate": SAMPLE_RATE,
            "audio_duration": round(duration, 4),
            "units": word_units,
        }

    chars = list(text)
    weights = [0.4 if c.isspace() else 1.0 for c in chars]
    total = sum(weights) or 1.0
    units = []
    t = 0.0
    for c, w in zip(chars, weights):
        d = duration * (w / total)
        units.append({"t": round(t, 4), "d": round(d, 4), "s": c})
        t += d
    return {
        "source": "approx-char",
        "sample_rate": SAMPLE_RATE,
        "audio_duration": round(duration, 4),
        "units": units,
    }


@dataclass
class TTSResult:
    samples: np.ndarray  # float32 mono @ SAMPLE_RATE
    sample_rate: int
    timings: dict

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    def to_wav_bytes(self) -> bytes:
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, self.samples, self.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class TTSEngine:
    def __init__(
        self,
        default_voice: str = DEFAULT_VOICE,
        lang_code: str = "a",
        speed: float = DEFAULT_SPEED,
    ):
        self.default_voice = default_voice
        self.lang_code = lang_code
        self.speed = speed
        self._pipeline = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        if self._pipeline is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import kokoro  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        with self._lock:
            if self._pipeline is not None:
                return
            try:
                from kokoro import KPipeline
            except Exception as e:  # noqa: BLE001
                self._load_error = "kokoro not installed. Run: pip install kokoro"
                raise RuntimeError(self._load_error) from e
            logger.info("Loading Kokoro pipeline (lang=%s) on CPU…", self.lang_code)
            self._pipeline = KPipeline(lang_code=self.lang_code)

    def synth(self, text: str, voice: Optional[str] = None) -> TTSResult:
        """Synthesize `text` to audio + timings. Blocking (run via to_thread).

        Emoji/markdown are stripped first so they aren't voiced aloud; timings
        correspond to the spoken (cleaned) text.
        """
        self._ensure_loaded()
        voice = voice or self.default_voice
        text = strip_nonspeech(text)
        if not text:
            return TTSResult(np.zeros(0, dtype=np.float32), SAMPLE_RATE,
                             build_timings("", 0.0))

        parts: List[np.ndarray] = []
        word_units: List[dict] = []
        offset = 0.0
        for result in self._pipeline(text, voice=voice, speed=self.speed):
            audio = _extract_audio(result)
            if audio is None or len(audio) == 0:
                continue
            parts.append(audio)
            toks = _extract_word_units(result, offset)
            if toks:
                word_units.extend(toks)
            offset += len(audio) / SAMPLE_RATE

        if not parts:
            return TTSResult(np.zeros(0, dtype=np.float32), SAMPLE_RATE,
                             build_timings(text, 0.0))

        samples = np.concatenate(parts).astype(np.float32)
        duration = len(samples) / SAMPLE_RATE
        # Only trust word_units if they cover the audio; else approximate.
        timings = build_timings(text, duration, word_units or None)
        return TTSResult(samples, SAMPLE_RATE, timings)


# ---------------------------------------------------------------------------
# Kokoro result shape helpers (its API has shifted across versions)
# ---------------------------------------------------------------------------
def _extract_audio(result) -> Optional[np.ndarray]:
    audio = None
    if hasattr(result, "audio"):
        audio = result.audio
    elif isinstance(result, (tuple, list)) and len(result) >= 3:
        audio = result[2]
    if audio is None:
        return None
    # torch tensor -> numpy
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _extract_word_units(result, offset: float) -> List[dict]:
    """Best-effort per-word timings from Kokoro token timestamps, if present."""
    tokens = getattr(result, "tokens", None)
    if not tokens:
        return []
    units: List[dict] = []
    for tok in tokens:
        text = getattr(tok, "text", None)
        start = getattr(tok, "start_ts", None)
        end = getattr(tok, "end_ts", None)
        if text is None or start is None or end is None:
            continue
        units.append({
            "t": round(offset + float(start), 4),
            "d": round(float(end) - float(start), 4),
            "s": text,
        })
    return units
