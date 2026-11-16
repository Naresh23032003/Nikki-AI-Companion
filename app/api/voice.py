"""Voice bench + status endpoints (RVC/studio consistency check, training tail).

Extracted from app/main.py (N2 phase 2, round 2). Bodies unchanged.
"""
from __future__ import annotations

import asyncio
import re

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import ROOT
from app.deps import state

router = APIRouter(tags=["voice"])

MEDIA_DIR = ROOT / "media"


class BenchRequest(BaseModel):
    emotions: list[str] = Field(default=["neutral", "happy", "sad"])


@router.post("/voice/bench")
async def voice_bench(body: BenchRequest):
    """Consistency bench: same 3 sentences via studio clone AND kokoro(+rvc),
    per emotion, so you can tune until it's one person everywhere."""
    sentences = [
        "hey, i was just thinking about you.",
        "no way, tell me everything right now!",
        "okay fine, you win this one... this time.",
    ]
    bench_dir = MEDIA_DIR / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    out: dict = {"studio_available": bool(state.studio and state.studio.available),
                 "rvc_available": bool(state.rvc and state.rvc.available),
                 "renders": []}
    vcfg = (state.settings.raw or {}).get("voice", {})
    for emo in body.emotions:
        for i, line in enumerate(sentences):
            entry = {"emotion": emo, "line": i}
            # Kokoro (+ optional RVC) - the call voice.
            res = await asyncio.to_thread(state.tts.synth, line)
            samples, sr = res.samples, res.sample_rate
            if vcfg.get("call_voice") == "kokoro_rvc" and state.rvc.available:
                samples, _ = await asyncio.to_thread(state.rvc.convert, samples, sr)
            p = bench_dir / f"call_{emo}_{i}.wav"
            sf.write(p, samples, sr)
            entry["call_url"] = f"/media/bench/{p.name}"
            # Studio clone - the voice-note voice.
            if out["studio_available"]:
                try:
                    from app.gpu_queue import PRIORITY_BENCH
                    fut = await state.gpu_queue.submit(
                        f"bench_{emo}_{i}",
                        lambda l=line, e=emo: asyncio.to_thread(
                            state.studio.render, l, e),
                        priority=PRIORITY_BENCH)
                    s2, sr2 = await fut
                    p2 = bench_dir / f"studio_{emo}_{i}.wav"
                    sf.write(p2, s2, sr2)
                    entry["studio_url"] = f"/media/bench/{p2.name}"
                except Exception as e:  # noqa: BLE001
                    entry["studio_error"] = str(e)
            out["renders"].append(entry)
    return out


def _training_progress() -> dict | None:
    """Tail rvc_training.log (UTF-16 from Tee-Object) for stage/epoch lines."""
    log = ROOT / "rvc_training.log"
    if not log.exists():
        return None
    try:
        text = log.read_text(encoding="utf-16", errors="ignore")
    except (UnicodeError, OSError):
        try:
            text = log.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    stages = [l for l in lines if l.startswith("STAGE") or "COMPLETE" in l
              or "EXPORTED" in l]
    epochs = [l for l in lines if re.search(r"epoch[ =:]+\d+", l, re.I)]
    return {
        "last_stage": stages[-1] if stages else None,
        "last_epoch_line": epochs[-1][:120] if epochs else None,
        "done": any("TRAINING COMPLETE" in l for l in lines),
        "failed": any("FAILED" in l for l in stages[-1:]) if stages else False,
    }


@router.get("/voice/status")
async def voice_status():
    vcfg = (state.settings.raw or {}).get("voice", {})
    return {
        "training": _training_progress(),
        "call_voice": vcfg.get("call_voice", "kokoro_raw"),
        "rvc_ready": bool(state.rvc and state.rvc.available),
        "rvc_status": state.rvc.status_label() if state.rvc else "not trained",
        "rvc_last_latency_ms": state.rvc.last_latency_ms if state.rvc else None,
        "studio_engine": state.studio.engine_name if state.studio else None,
        "studio_installed": bool(state.studio and state.studio.available),
        "gpu_queue": state.gpu_queue.status() if state.gpu_queue else {},
        "song_library": [s.get("title") for s in state.covers.library()]
        if state.covers else [],
        "inbox_pending": [p.name for p in state.covers.pending_inbox()]
        if state.covers else [],
    }
