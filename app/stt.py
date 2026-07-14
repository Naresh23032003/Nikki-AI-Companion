"""Speech-to-text via faster-whisper (small, int8).

Lazy-loaded so importing this module never requires the model or the package to
be present - the endpoints degrade gracefully with a clear message if STT isn't
installed. Uses the GPU (CUDA, int8) when available and falls back to CPU.

VRAM: whisper-small int8 is ~0.5 GB, chosen so it can coexist with llama3.2:3b
on a 6 GB GPU (see the README "VRAM budget" section).
"""
from __future__ import annotations

import io
import logging
import threading

logger = logging.getLogger("companion.stt")


class STTEngine:
    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self._model = None
        self._device = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        """True if the model can be (or already is) loaded."""
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import faster_whisper  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from faster_whisper import WhisperModel
            except Exception as e:  # noqa: BLE001
                self._load_error = (
                    "faster-whisper not installed. Run: pip install faster-whisper"
                )
                raise RuntimeError(self._load_error) from e

            device, compute_type = self._pick_device()
            logger.info(
                "Loading faster-whisper '%s' on %s (%s)…",
                self.model_size, device, compute_type,
            )
            try:
                self._model = WhisperModel(
                    self.model_size, device=device, compute_type=compute_type
                )
                self._device = device
            except Exception as e:  # noqa: BLE001 - GPU present but unusable -> CPU
                if device == "cuda":
                    logger.warning("CUDA load failed (%s); falling back to CPU.", e)
                    self._model = WhisperModel(
                        self.model_size, device="cpu", compute_type="int8"
                    )
                    self._device = "cpu"
                else:
                    self._load_error = str(e)
                    raise

    @staticmethod
    def _pick_device():
        """(device, compute_type): CUDA int8 if available AND there's headroom,
        else CPU int8. whisper-small int8 needs ~0.5GB; if the GPU is already
        near-full (Ollama + XTTS render in flight), loading it there would OOM
        someone else - CPU int8 transcribes a short utterance fine."""
        try:
            import torch
            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info()
                if free / 1e9 >= 1.0:
                    return "cuda", "int8"
                logger.info("STT: only %.1fGB VRAM free - using CPU", free / 1e9)
        except Exception:  # noqa: BLE001
            pass
        return "cpu", "int8"

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        """Transcribe a webm/opus (or any ffmpeg-decodable) audio blob to text.

        This is CPU/GPU-bound and blocking; call it via asyncio.to_thread from
        async handlers.
        """
        self._ensure_loaded()
        # faster-whisper accepts a binary file-like object; it decodes via PyAV,
        # which handles webm/opus.
        segments, _info = self._model.transcribe(
            io.BytesIO(audio_bytes),
            language=language,
            beam_size=5,
            vad_filter=True,  # drop silence/noise for cleaner short utterances
        )
        return "".join(seg.text for seg in segments).strip()
