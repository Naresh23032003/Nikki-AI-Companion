"""Tests for app/stt.py (N8: voice pipeline latency).

Does not load a real faster-whisper model (that requires downloading model
weights, which hung on what looked like a blocked network path in this
sandbox during this session - documented in IMPLEMENTATION_LOG.md rather
than skipped silently). Instead mocks STTEngine._model directly to verify
the beam_size plumbing this task is actually about.
"""
from __future__ import annotations

from app.stt import STTEngine


class _FakeSegment:
    def __init__(self, text):
        self.text = text


class _FakeModel:
    """Records the beam_size it was called with."""

    def __init__(self, segments=("hello there",)):
        self.calls: list[dict] = []
        self._segments = segments

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return [_FakeSegment(s) for s in self._segments], None


class TestBeamSizePlumbing:
    def _engine_with_fake_model(self):
        engine = STTEngine()
        engine._model = _FakeModel()  # bypass _ensure_loaded()
        return engine

    def test_default_beam_size_is_5(self):
        engine = self._engine_with_fake_model()
        engine.transcribe(b"fake-audio-bytes")
        assert engine._model.calls[-1]["beam_size"] == 5

    def test_call_mode_passes_beam_size_1(self):
        """Matches the call site in app/main.py's _run_call_turn - greedy
        decoding for the live-call latency-critical path."""
        engine = self._engine_with_fake_model()
        engine.transcribe(b"fake-audio-bytes", None, 1)
        assert engine._model.calls[-1]["beam_size"] == 1

    def test_result_text_is_joined_and_stripped(self):
        engine = self._engine_with_fake_model()
        engine._model = _FakeModel(segments=("hello ", "there "))
        text = engine.transcribe(b"fake-audio-bytes")
        assert text == "hello there"

    def test_vad_filter_always_on(self):
        engine = self._engine_with_fake_model()
        engine.transcribe(b"fake-audio-bytes")
        assert engine._model.calls[-1]["vad_filter"] is True

    def test_language_is_passed_through(self):
        engine = self._engine_with_fake_model()
        engine.transcribe(b"fake-audio-bytes", language="en")
        assert engine._model.calls[-1]["language"] == "en"


class TestDevicePicking:
    def test_cpu_when_torch_unavailable(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        device, compute_type = STTEngine._pick_device()
        assert device == "cpu"
        assert compute_type == "int8"


class TestAvailability:
    def test_available_true_once_model_loaded(self):
        engine = STTEngine()
        engine._model = _FakeModel()
        assert engine.available is True

    def test_available_false_after_a_load_error(self):
        engine = STTEngine()
        engine._load_error = "faster-whisper not installed"
        assert engine.available is False
