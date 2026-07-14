"""Validate the trained RVC model: 3 test conversions, before/after files.

    python tools/validate_rvc.py

Picks 2 speech clips (Kokoro-synthesized, so you hear the actual call
pipeline) + 1 vocal snippet from song_vocals, converts through
voices/rvc/nikki, writes pairs to voices/rvc/validation/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "voices" / "rvc" / "nikki"
OUT = ROOT / "voices" / "rvc" / "validation"

TEST_LINES = [
    "hey you! i was just thinking about you, how was your day?",
    "okay wait, that is actually the funniest thing i've heard all week.",
]


def main() -> int:
    import soundfile as sf

    from app.rvc_layer import RVCConverter
    from app.tts import TTSEngine

    rvc = RVCConverter(MODEL_DIR)
    if not rvc.available:
        print(f"no model at {MODEL_DIR} (need nikki.pth + nikki.index) "
              f"or rvc-python missing:  pip install rvc-python")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    tts = TTSEngine()

    for i, line in enumerate(TEST_LINES):
        res = tts.synth(line)
        before = OUT / f"speech{i}_before.wav"
        sf.write(before, res.samples, res.sample_rate)
        converted, ms = rvc.convert(res.samples, res.sample_rate)
        sf.write(OUT / f"speech{i}_after.wav", converted, res.sample_rate)
        print(f"speech{i}: converted in {ms:.0f}ms -> {OUT}")

    vocals = sorted((ROOT / "voice_tracks" / "song_vocals").glob("*.wav"))
    if vocals:
        x, sr = sf.read(vocals[0])
        snippet = x[: sr * 10]
        sf.write(OUT / "vocal_before.wav", snippet, sr)
        converted, ms = rvc.convert(snippet, sr)
        sf.write(OUT / "vocal_after.wav", converted, sr)
        print(f"vocal snippet: converted in {ms:.0f}ms")
    else:
        print("no song_vocals yet - run tools/process_songs.py for the vocal test")

    print(f"\nlisten to the pairs in {OUT} - 'after' should be unmistakably her.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
