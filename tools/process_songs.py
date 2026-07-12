"""PART 1 — Song processing: Demucs two-stem separation of her singing.

    python tools/process_songs.py [--device cuda|cpu]

voice_tracks/song/ (singing + keyboard) -> htdemucs vocals stem -> normalize
-> voice_tracks/song_vocals/ (48k mono WAV). Each output gets a confidence
rating from the vocal/accompaniment energy ratio; LOW-confidence files are
flagged for a listen check.

Requires: pip install torchaudio demucs  (see requirements-voice.txt)
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SONG_DIR = ROOT / "voice_tracks" / "song"
OUT_DIR = ROOT / "voice_tracks" / "song_vocals"
SEP_DIR = ROOT / "voice_tracks" / "_demucs_out"

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    try:
        import demucs  # noqa: F401
    except ImportError:
        print("demucs not installed:  pip install torchaudio demucs")
        return 1
    import soundfile as sf

    songs = [p for p in SONG_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTS]
    if not songs:
        print(f"no audio in {SONG_DIR}")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"separating {len(songs)} file(s) with htdemucs on {args.device}…")
    # two-stems=vocals -> vocals.wav + no_vocals.wav per input
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals",
           "-n", "htdemucs", "-d", args.device, "-o", str(SEP_DIR),
           *[str(s) for s in songs]]
    subprocess.run(cmd, check=True)

    flagged = []
    for song in songs:
        stem_dir = SEP_DIR / "htdemucs" / song.stem
        vpath = stem_dir / "vocals.wav"
        apath = stem_dir / "no_vocals.wav"
        if not vpath.exists():
            print(f"  !! missing stem for {song.name}")
            continue
        v, sr = sf.read(vpath)
        if v.ndim > 1:
            v = v.mean(axis=1)
        # normalize to -1dBFS-ish
        peak = float(np.max(np.abs(v)) or 1.0)
        v = v * (0.89 / peak)
        # resample to 48k mono
        if sr != 48000:
            import torch
            import torchaudio.functional as AF
            v = AF.resample(torch.from_numpy(v.astype(np.float32)), sr, 48000).numpy()
        out = OUT_DIR / f"{song.stem}_vocals.wav"
        sf.write(out, v.astype(np.float32), 48000, subtype="PCM_16")

        # Confidence: vocal energy vs leftover accompaniment energy.
        ratio = float("inf")
        if apath.exists():
            a, _ = sf.read(apath)
            if a.ndim > 1:
                a = a.mean(axis=1)
            ve = float(np.sqrt(np.mean(v ** 2)))
            ae = float(np.sqrt(np.mean(a ** 2))) or 1e-9
            ratio = ve / ae
        conf = "OK" if ratio > 0.5 else "LOW"
        if conf == "LOW":
            flagged.append(out.name)
        print(f"  {song.name} -> {out.name}  [vocal/accomp ratio {ratio:.2f} -> {conf}]")

    print("\n" + "=" * 62)
    print("LISTEN CHECK before training: quiet instrumental residue in gaps")
    print("is fine; audible keyboard UNDER her voice is not — drop those files.")
    if flagged:
        print("LOW-confidence outputs (listen to these first):")
        for f in flagged:
            print(f"  - {f}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
