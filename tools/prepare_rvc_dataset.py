"""PART 2 - RVC dataset prep: gold speech + separated vocals -> rvc_dataset/.

    python tools/prepare_rvc_dataset.py            # voice/ + song_vocals/
    python tools/prepare_rvc_dataset.py --trim-silver path/  # silver helper

Slices long files into 5-15s pieces on silence boundaries, resamples to 48k
mono WAV (Applio resamples internally to the model rate you pick; 48k input
keeps every option open). --trim-silver runs an energy-based head/tail trim
for the "needs trimming" tier before inclusion.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "voice_tracks" / "voice"
SONG_VOCALS = ROOT / "voice_tracks" / "song_vocals"
OUT = ROOT / "rvc_dataset"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
SR = 48000
MIN_S, MAX_S = 5.0, 15.0


def load_mono_48k(path: Path) -> np.ndarray:
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != SR:
        try:
            import torch
            import torchaudio.functional as AF
            data = AF.resample(torch.from_numpy(data), sr, SR).numpy()
        except ImportError:  # crude linear fallback
            idx = np.linspace(0, len(data) - 1, int(len(data) * SR / sr))
            data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
    return data


def trim_silence(x: np.ndarray, thresh_db: float = -40.0) -> np.ndarray:
    """Energy-based head/tail trim (the 'silver' helper)."""
    if len(x) == 0:
        return x
    win = int(SR * 0.03)
    frames = [np.sqrt(np.mean(x[i:i + win] ** 2)) for i in range(0, len(x) - win, win)]
    frames = np.array(frames) + 1e-9
    db = 20 * np.log10(frames / (np.max(frames) + 1e-9))
    keep = np.where(db > thresh_db)[0]
    if len(keep) == 0:
        return x
    return x[keep[0] * win:(keep[-1] + 2) * win]


def slice_file(x: np.ndarray) -> list[np.ndarray]:
    """Split into 5-15s pieces, preferring quiet points as boundaries."""
    total = len(x) / SR
    if total <= MAX_S:
        return [x] if total >= 2.0 else []
    pieces = []
    start = 0
    while start < len(x):
        end = min(start + int(MAX_S * SR), len(x))
        if (end - start) / SR > MIN_S and end < len(x):
            # look for the quietest 30ms window in the last 3s to cut at
            zone = x[end - int(3 * SR):end]
            win = int(SR * 0.03)
            rms = [np.sqrt(np.mean(zone[i:i + win] ** 2))
                   for i in range(0, len(zone) - win, win)]
            if rms:
                end = end - int(3 * SR) + int(np.argmin(rms)) * win
        piece = x[start:end]
        if (len(piece) / SR) >= 2.0:
            pieces.append(piece)
        start = end
    return pieces


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trim-silver", metavar="DIR",
                    help="also include this folder, silence-trimming each clip")
    args = ap.parse_args()

    sources: list[tuple[Path, bool]] = [(GOLD, False)]
    if SONG_VOCALS.exists():
        sources.append((SONG_VOCALS, False))
    if args.trim_silver:
        sources.append((Path(args.trim_silver), True))

    OUT.mkdir(exist_ok=True)
    n = 0
    total_s = 0.0
    for folder, do_trim in sources:
        files = [p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXTS] \
            if folder.exists() else []
        print(f"{folder.name}: {len(files)} file(s)" + (" [trimming]" if do_trim else ""))
        for f in files:
            x = load_mono_48k(f)
            if do_trim:
                x = trim_silence(x)
            for piece in slice_file(x):
                peak = float(np.max(np.abs(piece)) or 1.0)
                piece = piece * (0.89 / peak)
                out = OUT / f"{f.stem}_{n:04d}.wav"
                sf.write(out, piece, SR, subtype="PCM_16")
                total_s += len(piece) / SR
                n += 1
    print(f"\nwrote {n} clips ({total_s / 60:.1f} min) to {OUT}")
    print("next: docs/RVC_TRAINING.md - train in Applio, export to voices/rvc/nikki")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
