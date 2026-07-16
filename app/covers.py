"""PART 5 - Singing covers: songs/inbox/*.mp3 -> her voice -> songs/library/.

Pipeline (one GPU-queue job per song, priority below voice notes):
  Demucs two-stem split -> RVC-convert the vocal (auto pitch shift into her
  range, configurable semitones) -> remix with the instrumental + light
  reverb -> songs/library/<title>.mp3 + <title>.json metadata (editable mood
  tags: romantic, fun, sad, lullaby, birthday).

Delivery rules live in the sing tool (app/tools.py): library hits are instant,
misses queue a job + an in-character "gimme 10 mins 🤭" via the deferred flow.
She NEVER claims to sing live on calls - always "i'll send it after".
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from app.config import ROOT

logger = logging.getLogger("companion.covers")

INBOX = ROOT / "songs" / "inbox"
LIBRARY = ROOT / "songs" / "library"
WORK = ROOT / "songs" / "_work"
MOOD_TAGS = ["romantic", "fun", "sad", "lullaby", "birthday"]
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def ensure_dirs() -> None:
    for d in (INBOX, LIBRARY, WORK):
        d.mkdir(parents=True, exist_ok=True)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "song"


def _light_reverb(x: np.ndarray, sr: int) -> np.ndarray:
    """Subtle feedback-delay 'room' so the converted vocal sits in the mix."""
    out = x.copy()
    for delay_ms, gain in ((37, 0.18), (61, 0.12), (93, 0.07)):
        d = int(sr * delay_ms / 1000)
        out[d:] += x[:-d] * gain
    peak = float(np.max(np.abs(out)) or 1.0)
    return out * min(1.0, 0.95 / peak)


class CoverPipeline:
    def __init__(self, rvc, settings, gpu_queue):
        self.rvc = rvc
        self.gpu_queue = gpu_queue
        vcfg = (settings.raw or {}).get("voice", {})
        self.pitch_shift = int(vcfg.get("cover_pitch_semitones", 0))
        ensure_dirs()

    # -- library ---------------------------------------------------------------
    def library(self) -> list[dict]:
        out = []
        for meta in sorted(LIBRARY.glob("*.json")):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                if (LIBRARY / data.get("file", "")).exists():
                    out.append(data)
            except (json.JSONDecodeError, OSError):
                continue
        return out

    def find(self, query: str | None = None, mood: str | None = None,
             exclude_file: str | None = None) -> dict | None:
        """Find a song. `exclude_file` avoids repeating the last one sent when
        picking at random (a specific-title query still returns its exact
        match) - so "send another song" doesn't keep handing over the same
        track. Falls back to the excluded song only if it's the sole option."""
        import random
        lib = self.library()
        if not lib:
            return None
        if query:
            q = _slug(query)
            for s in lib:
                if q in _slug(s.get("title", "")) or _slug(s.get("title", "")) in q:
                    return s
            return None

        def pick(pool: list[dict]) -> dict:
            fresh = [s for s in pool if s.get("file") != exclude_file]
            return random.choice(fresh or pool)

        if mood:
            hits = [s for s in lib if mood in (s.get("moods") or [])]
            if hits:
                return pick(hits)
        return pick(lib)

    def pending_inbox(self) -> list[Path]:
        done = {json.loads(p.read_text(encoding="utf-8")).get("source")
                for p in LIBRARY.glob("*.json") if p.is_file()}
        return [p for p in INBOX.iterdir()
                if p.suffix.lower() in AUDIO_EXTS and p.name not in done]

    # -- job ---------------------------------------------------------------------
    async def enqueue(self, song_path: Path):
        from app.gpu_queue import PRIORITY_COVER
        return await self.gpu_queue.submit(
            f"cover:{song_path.stem}",
            lambda: asyncio.to_thread(self._process, song_path),
            priority=PRIORITY_COVER)

    def _process(self, song_path: Path) -> dict:
        """Blocking cover render (runs inside the GPU queue)."""
        import soundfile as sf

        if not self.rvc.available:
            raise RuntimeError("RVC model not trained yet (docs/RVC_TRAINING.md)")
        title = song_path.stem
        logger.info("cover: processing %r", title)

        # 1) split
        try:
            subprocess.run([sys.executable, "-m", "demucs", "--two-stems", "vocals",
                            "-n", "htdemucs", "-o", str(WORK), str(song_path)],
                           check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            # capture_output swallows stderr into the exception object - the
            # bare CalledProcessError repr (just "exit status 1") is useless
            # for diagnosing WHY it failed, so surface it in the log instead.
            logger.error("cover: demucs failed for %r:\n%s", title,
                        e.stderr.decode(errors="replace") if e.stderr else "(no stderr)")
            raise
        stem_dir = WORK / "htdemucs" / title
        vocals, sr = sf.read(stem_dir / "vocals.wav")
        instr, _ = sf.read(stem_dir / "no_vocals.wav")
        if vocals.ndim > 1:
            vocals = vocals.mean(axis=1)

        # 2) convert into her voice (+ pitch shift into her range)
        converted, ms = self.rvc.convert(vocals.astype(np.float32), sr,
                                         f0_up_key=self.pitch_shift)
        logger.info("cover: vocal converted in %.1fs", ms / 1000)

        # 3) remix: her vocal (light reverb) over the instrumental
        voc = _light_reverb(converted, sr)
        if instr.ndim > 1:
            instr = instr.mean(axis=1)
        n = min(len(voc), len(instr))
        mix = voc[:n] * 0.9 + instr[:n] * 0.75
        peak = float(np.max(np.abs(mix)) or 1.0)
        mix = mix * min(1.0, 0.95 / peak)

        slug = _slug(title)
        wav_tmp = WORK / f"{slug}.wav"
        sf.write(wav_tmp, mix.astype(np.float32), sr)
        mp3 = LIBRARY / f"{slug}.mp3"
        try:
            subprocess.run(["ffmpeg", "-y", "-i", str(wav_tmp), "-b:a", "192k",
                            str(mp3)], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error("cover: ffmpeg encode failed for %r:\n%s", title,
                        e.stderr.decode(errors="replace") if e.stderr else "(no stderr)")
            raise

        meta = {"title": title, "file": mp3.name, "source": song_path.name,
                "moods": [], "pitch_shift": self.pitch_shift,
                "url": f"/songs/{mp3.name}"}
        (LIBRARY / f"{slug}.json").write_text(json.dumps(meta, indent=2),
                                              encoding="utf-8")
        logger.info("cover: %r ready -> %s (edit moods in the .json)", title, mp3.name)
        return meta
