"""Warm RVC inference worker — runs INSIDE the Applio env, serves the app.

    applio/env/Scripts/python.exe tools/rvc_server.py   (port 3002)

Loads Applio's VoiceConverter once (model stays warm on the GPU) and converts
wav->wav over local HTTP. The main app's RVC layer calls this instead of
bundling RVC deps into its own environment (rvc-python is broken on py3.12).
"""
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APPLIO = ROOT / "applio"
MODEL_DIR = ROOT / "voices" / "rvc" / "nikki"
sys.path.insert(0, str(APPLIO))

import uvicorn  # noqa: E402  (applio env ships fastapi+uvicorn via gradio)
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s rvc_server %(message)s")
logger = logging.getLogger("rvc_server")

app = FastAPI(title="RVC worker")
_converter = None
_tmp = Path(tempfile.mkdtemp(prefix="rvc_worker_"))
_patched = False


def _patch_rvc_hot_paths() -> None:
    """Cache the RMVPE pitch model + FAISS index across calls.

    Applio's pipeline recreates RMVPE from scratch (disk weight load + a
    one-time CUDA kernel-selection warmup) on EVERY conversion, then discards
    it, and separately re-reads + reconstructs the whole FAISS index from disk
    every call too. Measured on this machine: ~0.7-1.1s RMVPE reload + ~0.1-0.4s
    warmup + ~0.14s FAISS reload — nearly all of the ~2s per-chunk latency,
    since a WARM RMVPE instance answers in ~40-100ms and a cached index is
    free. This patches only OUR worker process (not Applio's vendored files),
    so `git pull`-ing Applio later doesn't conflict with it.
    """
    global _patched
    if _patched:
        return
    import faiss
    from rvc.lib.predictors.f0 import RMVPE

    # RMVPE: one instance per (device, model_name, sample_rate, hop_size).
    # No __del__ on this class, so the pipeline's `del model` after each use
    # is just a refcount decrement — the cached instance survives untouched.
    rmvpe_cache: dict = {}
    rmvpe_ready: set = set()
    orig_new = RMVPE.__new__
    orig_init = RMVPE.__init__

    def cached_new(cls, device, model_name="rmvpe.pt", sample_rate=16000, hop_size=160):
        key = (device, model_name, sample_rate, hop_size)
        if key in rmvpe_cache:
            return rmvpe_cache[key]
        obj = orig_new(cls)
        rmvpe_cache[key] = obj
        return obj

    def cached_init(self, device, model_name="rmvpe.pt", sample_rate=16000, hop_size=160):
        key = (device, model_name, sample_rate, hop_size)
        if key in rmvpe_ready:
            return  # already loaded on GPU — skip the expensive reload
        orig_init(self, device, model_name, sample_rate, hop_size)
        rmvpe_ready.add(key)
        logger.info("RMVPE loaded and cached for %s", key)

    RMVPE.__new__ = cached_new
    RMVPE.__init__ = cached_init

    # FAISS index: cache by (path, mtime) so retraining still invalidates it.
    # pipeline.py only reads/reconstructs the index (no mutation), so sharing
    # one loaded instance across calls is safe.
    faiss_cache: dict = {}
    orig_read_index = faiss.read_index

    def cached_read_index(path, *a, **kw):
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return orig_read_index(path, *a, **kw)
        key = (path, mtime)
        if key not in faiss_cache:
            faiss_cache[key] = orig_read_index(path, *a, **kw)
            logger.info("FAISS index loaded and cached: %s", path)
        return faiss_cache[key]

    faiss.read_index = cached_read_index
    _patched = True
    logger.info("hot-path caching enabled (RMVPE + FAISS)")


def _model_files():
    pth = next(iter(MODEL_DIR.glob("*.pth")), None)
    idx = next(iter(MODEL_DIR.glob("*.index")), None)
    return pth, idx


def _get_converter():
    global _converter
    if _converter is None:
        os.chdir(APPLIO)  # Applio resolves its assets relative to its root
        _patch_rvc_hot_paths()
        from rvc.infer.infer import VoiceConverter
        _converter = VoiceConverter()
    return _converter


@app.get("/health")
def health():
    pth, idx = _model_files()
    return {"ok": True, "model": pth.name if pth else None,
            "index": idx.name if idx else None, "warm": _converter is not None}


@app.on_event("startup")
async def _warmup() -> None:
    """Pre-load the model + RMVPE + FAISS index, priming a few common chunk
    lengths (typical TTS sentence durations) so real calls don't pay the
    per-shape kernel-selection cost mid-conversation."""
    pth, idx = _model_files()
    if not (pth and idx):
        logger.warning("no model found in %s — skipping warmup", MODEL_DIR)
        return
    import numpy as np
    import soundfile as sf
    vc = _get_converter()
    for secs in (1.5, 2.5, 3.5, 4.5):
        clip = _tmp / f"warmup_{secs}.wav"
        out = _tmp / f"warmup_{secs}_out.wav"
        sf.write(clip, np.zeros(int(24000 * secs), dtype=np.float32), 24000)
        t0 = time.perf_counter()
        vc.convert_audio(audio_input_path=str(clip), audio_output_path=str(out),
                         model_path=str(pth), index_path=str(idx), f0_method="rmvpe")
        logger.info("warmup %.1fs chunk: %.0fms", secs, (time.perf_counter() - t0) * 1000)
    logger.info("worker ready — common chunk lengths primed")


@app.post("/convert")
async def convert(request: Request, pitch: int = 0):
    """Raw WAV bytes in, raw WAV bytes out — no multipart parsing, no
    FileResponse disk-streaming. This is an internal, single-caller worker
    (the app's RVC layer), so skipping both saves ~500-900ms per call that
    had nothing to do with the actual GPU conversion (measured: multipart
    UploadFile parsing + FileResponse's chunked async disk read accounted for
    essentially ALL of the gap between server-side and client-observed time)."""
    import soundfile as sf

    pth, idx = _model_files()
    if not (pth and idx):
        return JSONResponse({"error": "no model in voices/rvc/nikki"}, 503)
    src = _tmp / f"in_{time.time_ns()}.wav"
    dst = _tmp / f"out_{time.time_ns()}.wav"
    src.write_bytes(await request.body())
    in_sr = sf.info(str(src)).samplerate
    t0 = time.perf_counter()
    vc = _get_converter()
    vc.convert_audio(
        audio_input_path=str(src),
        audio_output_path=str(dst),
        model_path=str(pth),
        index_path=str(idx),
        pitch=pitch,
        f0_method="rmvpe",
    )
    ms = (time.perf_counter() - t0) * 1000
    # The RVC model's own sample rate (e.g. 40k/48k) almost never matches the
    # caller's input rate (Kokoro's 24k) — resample back here, INSIDE the
    # Applio env where librosa/scipy are already guaranteed present, rather
    # than requiring the lean main-app venv to carry a version-pinned
    # torch/torchaudio just for this one incidental step.
    out, out_sr = sf.read(str(dst))
    logger.info("convert: in_sr=%d out_sr=%d", in_sr, out_sr)
    if out_sr != in_sr:
        import librosa
        if out.ndim > 1:
            out = out.mean(axis=1)
        out = librosa.resample(out.astype("float32"), orig_sr=out_sr, target_sr=in_sr)
        out_sr = in_sr
        import io
        buf_out = io.BytesIO()
        sf.write(buf_out, out, out_sr, format="WAV")
        out_bytes = buf_out.getvalue()
    else:
        out_bytes = dst.read_bytes()
    src.unlink(missing_ok=True)
    dst.unlink(missing_ok=True)
    return Response(content=out_bytes, media_type="audio/wav",
                    headers={"X-Latency-Ms": f"{ms:.0f}"})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3002, log_level="warning")
