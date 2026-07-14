"""PART 3 - Studio TTS: high-quality renders in HER cloned voice.

Two engines behind one interface (config `voice.studio_engine`):
- "xtts": Coqui XTTS-v2 - clones from a reference clip; the message's EMOTION
  metadata selects ref/<emotion>/ (fallback: neutral, logged).
- "chatterbox": Resemble Chatterbox - same neutral clone, emotion mapped to
  exaggeration/intensity presets (tunable in config `voice.emotion_presets`).

GPU, lazily loaded and unloaded after idle; renders go through the shared GPU
job queue (voice notes outrank covers). 10-30s per render is expected.

Speakable-text pass: optional local-LLM rewrite so the AUDIO sounds spoken
(interjections, ellipses, written-out laughs) while the text bubble keeps the
original words.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from pathlib import Path

import numpy as np

from app.config import ROOT
from app.emotion import EMOTIONS, strip_for_speech

logger = logging.getLogger("companion.studio")

REF_ROOT = ROOT / "voice_tracks" / "ref"

DEFAULT_PRESETS = {  # chatterbox knobs per emotion (overridable in config)
    "neutral": {"exaggeration": 0.45, "cfg_weight": 0.5},
    "happy": {"exaggeration": 0.7, "cfg_weight": 0.45},
    "laughing": {"exaggeration": 0.9, "cfg_weight": 0.35},
    "shy": {"exaggeration": 0.35, "cfg_weight": 0.6},
    "sad": {"exaggeration": 0.4, "cfg_weight": 0.6},
    "surprised": {"exaggeration": 0.85, "cfg_weight": 0.4},
    "love": {"exaggeration": 0.6, "cfg_weight": 0.5},
}

_SPEAKABLE_PROMPT = (
    "Rewrite this text so it SOUNDS natural spoken aloud as a voice note from "
    "a girlfriend: add small interjections, ellipses for pauses, write out "
    "laughs ('haha'), expand emoji into vocal tone words or drop them. Keep "
    "the meaning and length similar. Reply with ONLY the rewritten text.\n\n{t}"
)


def ref_clip_for(emotion: str) -> Path | None:
    """Pick a handpicked reference clip for the emotion (neutral fallback)."""
    emo = (emotion or "neutral").lower()
    if emo not in EMOTIONS:
        emo = "neutral"
    for candidate in (emo, "neutral"):
        folder = next((d for d in REF_ROOT.iterdir()
                       if d.is_dir() and d.name.lower() == candidate), None) \
            if REF_ROOT.exists() else None
        if folder:
            clips = [p for p in folder.iterdir()
                     if p.suffix.lower() in {".wav", ".flac", ".mp3"}]
            if clips:
                if candidate != emo:
                    logger.info("studio: no ref for %r - using neutral", emo)
                return random.choice(clips)
    logger.warning("studio: no reference clips found under %s", REF_ROOT)
    return None


class StudioTTS:
    def __init__(self, llm, settings):
        self.llm = llm
        vcfg = (settings.raw or {}).get("voice", {})
        self.engine_name = vcfg.get("studio_engine", "xtts")
        self.speakable_enabled = bool(vcfg.get("speakable_rewrite", True))
        self.presets = {**DEFAULT_PRESETS, **(vcfg.get("emotion_presets") or {})}
        self.idle_unload_s = int(vcfg.get("studio_idle_unload_s", 300))
        # Minimum free VRAM to load the engine on CUDA. XTTS-v2 takes ~3.4GB;
        # loading it into a card without that much headroom doesn't OOM on
        # Windows - the driver silently spills into shared system memory and
        # EVERYTHING on the GPU (Ollama included) crawls: a measured render
        # went from RTF 1.04 to RTF 24 (510s), starving chat into 120s
        # ReadTimeouts. Better to refuse and let callers fall back to
        # Kokoro+RVC, which still sounds like her.
        self.min_free_vram_mb = int(vcfg.get("studio_min_free_vram_mb", 3800))
        self._engine = None
        self._engine_kind = None
        self._last_used = 0.0
        self._lock = threading.Lock()
        # Held for the WHOLE render; maybe_idle_unload only acts when it can
        # take it without blocking, so an in-flight render can never be
        # unloaded out from under itself.
        self._render_lock = threading.Lock()
        # Injected by main: returns True while RVC training owns the GPU.
        self.prefer_cpu = None

    @property
    def available(self) -> bool:
        try:
            if self.engine_name == "chatterbox":
                import chatterbox  # noqa: F401
            else:
                import TTS  # noqa: F401
            return True
        except ImportError:
            return False

    # -- engine lifecycle (lazy load, idle unload) ---------------------------
    @staticmethod
    def _vram_mb() -> float | None:
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                return (total - free) / 1e6
        except Exception:  # noqa: BLE001
            pass
        return None

    @staticmethod
    def _vram_free_mb() -> float | None:
        try:
            import torch
            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info()
                return free / 1e6
        except Exception:  # noqa: BLE001
            pass
        return None

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def _load(self):
        with self._lock:
            if self._engine is not None and self._engine_kind == self.engine_name:
                return
            self.unload()
            device = "cpu" if (self.prefer_cpu and self.prefer_cpu()) else "cuda"
            if device == "cuda":
                free = self._vram_free_mb()
                if free is not None and free < self.min_free_vram_mb:
                    # Deliberately raise instead of quietly using CPU: a CPU
                    # XTTS render takes minutes per sentence - the Kokoro
                    # fallback in the callers is the right degradation.
                    raise RuntimeError(
                        f"only {free:.0f}MB VRAM free (< {self.min_free_vram_mb}MB "
                        f"needed) - skipping studio load so the GPU doesn't "
                        f"over-commit and stall chat; falling back to Kokoro")
            before = self._vram_mb()
            logger.info("studio: loading %s on %s… (VRAM used: %s MB)",
                        self.engine_name, device,
                        f"{before:.0f}" if before is not None else "n/a")
            try:
                self._engine = self._build(device)
            except Exception as e:  # noqa: BLE001 - OOM etc -> CPU fallback
                if device == "cuda":
                    logger.warning("studio: GPU load failed (%s) - using CPU", e)
                    self._engine = self._build("cpu")
                else:
                    raise
            self._engine_kind = self.engine_name
            after = self._vram_mb()
            if before is not None and after is not None:
                logger.info("studio: loaded (VRAM used: %.0f MB, +%.0f MB)",
                            after, after - before)

    def _build(self, device: str):
        import os
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        if self.engine_name == "chatterbox":
            from chatterbox.tts import ChatterboxTTS
            return ChatterboxTTS.from_pretrained(device=device)
        from TTS.api import TTS as CoquiTTS
        return CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def unload(self) -> None:
        if self._engine is not None:
            logger.info("studio: unloading %s", self._engine_kind)
            self._engine = None
            self._engine_kind = None
            try:
                import torch
                torch.cuda.empty_cache()
                after = self._vram_mb()
                if after is not None:
                    logger.info("studio: unloaded (VRAM used: %.0f MB)", after)
            except Exception:  # noqa: BLE001
                pass

    def maybe_idle_unload(self) -> None:
        """Unload after idle - but NEVER while a render holds _render_lock.

        The old version stamped _last_used at render START, so any render
        longer than idle_unload_s got 'unloaded' mid-flight: engine nulled
        and torch.cuda.empty_cache() fired against a GPU mid-forward-pass
        (observed blocking ~2 minutes and starving the event loop). Now the
        stamp is at render END and this check skips entirely when busy."""
        if self._engine is None:
            return
        if not self._render_lock.acquire(blocking=False):
            return  # render in progress - not idle, try again next tick
        try:
            if time.time() - self._last_used > self.idle_unload_s:
                self.unload()
        finally:
            self._render_lock.release()

    # -- speakable pass --------------------------------------------------------
    async def speakable(self, text: str) -> str:
        if not self.speakable_enabled:
            return text
        try:
            out = await self.llm.chat(
                messages=[{"role": "user",
                           "content": _SPEAKABLE_PROMPT.format(t=text)}],
                options={"temperature": 0.6})
            out = out.strip().strip('"')
            return out if 0 < len(out) < len(text) * 3 else text
        except Exception as e:  # noqa: BLE001
            logger.warning("speakable rewrite failed: %s", e)
            return text

    # -- render (BLOCKING, run inside the GPU queue) ------------------------------
    def render(self, text: str, emotion: str = "neutral") -> tuple[np.ndarray, int]:
        """Render text -> (samples, sample_rate) in her cloned voice.

        Holds _render_lock for the duration (blocks maybe_idle_unload), and
        stamps _last_used at the END - the idle countdown starts when the
        render finishes, not when it began."""
        with self._render_lock:
            try:
                text = strip_for_speech(text)
                self._load()
                ref = ref_clip_for(emotion)
                if ref is None:
                    raise RuntimeError("no reference clips in voice_tracks/ref/")

                if self.engine_name == "chatterbox":
                    preset = self.presets.get((emotion or "neutral").lower(),
                                              self.presets["neutral"])
                    wav = self._engine.generate(
                        text, audio_prompt_path=str(ref),
                        exaggeration=float(preset.get("exaggeration", 0.5)),
                        cfg_weight=float(preset.get("cfg_weight", 0.5)))
                    samples = wav.squeeze().cpu().numpy()
                    return samples.astype(np.float32), self._engine.sr

                # XTTS-v2: the emotion lives in the reference clip itself.
                samples = self._engine.tts(text=text, speaker_wav=str(ref), language="en")
                return np.asarray(samples, dtype=np.float32), 24000
            finally:
                self._last_used = time.time()
