"""RVC conversion layer: any audio -> her timbre.

Talks to the warm RVC worker (tools/rvc_server.py, running in the Applio env
on 127.0.0.1:3002) - rvc-python is broken on Python 3.12, and the worker keeps
the model loaded so per-chunk call latency stays low. If the worker or model
is missing, everything degrades to the raw voice.
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import httpx
import numpy as np

logger = logging.getLogger("companion.rvc")

WORKER_URL = "http://127.0.0.1:3002"


class RVCConverter:
    def __init__(self, model_dir: Path, worker_url: str = WORKER_URL):
        self.model_dir = Path(model_dir)
        self.worker_url = worker_url.rstrip("/")
        self.last_latency_ms: float | None = None
        self._worker_ok_at: float = 0.0
        # ONE persistent client (keep-alive connection), reused across every
        # call. A fresh httpx.get/post() call opens a NEW TCP connection each
        # time - measured at ~550-570ms of pure connection-establishment
        # overhead on Windows loopback, dwarfing the ~250ms of actual GPU
        # work. httpx.Client's connection pool is safe to share across
        # threads, which matters since .convert() runs via asyncio.to_thread.
        self._client = httpx.Client(timeout=120.0)

    @property
    def _model_exported(self) -> bool:
        return bool(self.model_dir.exists() and any(self.model_dir.glob("*.pth"))
                   and any(self.model_dir.glob("*.index")))

    @property
    def available(self) -> bool:
        """Model exported AND worker healthy (cached for 30s)."""
        if not self._model_exported:
            return False
        if time.time() - self._worker_ok_at < 30:
            return True
        try:
            r = self._client.get(f"{self.worker_url}/health", timeout=2.0)
            if r.status_code == 200 and r.json().get("ok"):
                self._worker_ok_at = time.time()
                return True
        except httpx.HTTPError:
            pass
        return False

    def status_label(self) -> str:
        """Human-readable reason for logging - `available` collapses "model
        never trained" and "worker just isn't up yet" into one False, which
        misleadingly reads as a lost/corrupted model on every startup where
        the worker (a separate process) hasn't opened its port yet."""
        if not self._model_exported:
            return "not trained"
        if self.available:
            return "ready"
        return "trained, worker not reachable yet (start_rvc_worker.bat?)"

    def convert(self, samples: np.ndarray, sample_rate: int,
                f0_up_key: int = 0) -> tuple[np.ndarray, float]:
        """Convert audio into her voice. Returns (samples, latency_ms).

        Blocking (HTTP round-trip): call via asyncio.to_thread in async code.
        """
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, samples.astype(np.float32), sample_rate, format="WAV")
        t0 = time.perf_counter()
        # Raw bytes, not multipart (multipart parsing + FileResponse's chunked
        # disk read cost real overhead too), over the persistent client above.
        r = self._client.post(
            f"{self.worker_url}/convert",
            params={"pitch": f0_up_key},
            content=buf.getvalue(),
            headers={"Content-Type": "audio/wav"},
        )
        r.raise_for_status()
        out, out_sr = sf.read(io.BytesIO(r.content))
        if out.ndim > 1:
            out = out.mean(axis=1)
        if out_sr != sample_rate:
            # The worker (tools/rvc_server.py) resamples back to the input
            # rate itself before responding - this is a defensive fallback
            # only, not the normal path. Avoid a torch/torchaudio dependency
            # in this lean venv for what should never actually trigger.
            logger.warning("rvc worker returned %dHz, expected %dHz - resampling crudely",
                          out_sr, sample_rate)
            duration = len(out) / out_sr
            target_len = int(round(duration * sample_rate))
            out = np.interp(
                np.linspace(0, len(out) - 1, target_len),
                np.arange(len(out)), out,
            ).astype(np.float32)
        ms = (time.perf_counter() - t0) * 1000.0
        self.last_latency_ms = ms
        level = logging.INFO if ms < 300 else logging.WARNING
        logger.log(level, "rvc chunk: %.0fms %s", ms,
                   "[ok]" if ms < 300 else "[over 300ms target]")
        return out.astype(np.float32), ms

    def unload(self) -> None:  # worker owns the model; nothing to do here
        pass

    def close(self) -> None:
        self._client.close()
