"""CONSENT WITHDRAWAL - delete EVERYTHING derived from her recordings.

    python cleanup_voice.py           # lists what will be deleted, asks once
    python cleanup_voice.py --yes     # no prompt

Removes: raw recordings, separated vocals, the RVC dataset and trained model,
emotion reference clips, all rendered studio audio, benches, and song covers.
Irreversible by design. If the person whose voice this is ever withdraws
consent, run this - no partial keeping.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TARGETS = [
    ROOT / "voice_tracks",        # raw recordings + refs + separated vocals
    ROOT / "rvc_dataset",         # training slices
    ROOT / "voices" / "rvc",      # trained model + validation renders
    ROOT / "songs",               # covers: inbox, work, library
    ROOT / "media" / "tts",       # rendered voice notes
    ROOT / "media" / "bench",     # consistency bench renders
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="skip confirmation")
    args = ap.parse_args()

    existing = [t for t in TARGETS if t.exists()]
    if not existing:
        print("nothing to delete - already clean.")
        return 0

    print("This will PERMANENTLY delete everything derived from her voice:")
    for t in existing:
        n = sum(1 for _ in t.rglob("*") if _.is_file())
        print(f"  {t}   ({n} files)")
    if not args.yes:
        answer = input("\ntype DELETE to confirm: ").strip()
        if answer != "DELETE":
            print("aborted - nothing deleted.")
            return 1

    for t in existing:
        shutil.rmtree(t, ignore_errors=True)
        print(f"deleted {t}")
    print("\ndone. also remember to delete any copies inside Applio's logs/ "
          "folder and set voice.call_voice back to kokoro_raw in config.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
