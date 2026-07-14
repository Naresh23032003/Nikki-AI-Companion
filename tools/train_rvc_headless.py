"""Headless RVC training driver: Applio CLI end-to-end, no GUI.

    <applio-env-python> tools/train_rvc_headless.py [--epochs 300] [--batch 4]

prerequisites -> preprocess -> extract(rmvpe) -> train(40k) -> index ->
export best weights + index to voices/rvc/nikki. Logs every stage; designed
to run unattended in the background for hours.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APPLIO = ROOT / "applio"
DATASET = ROOT / "rvc_dataset"
EXPORT = ROOT / "voices" / "rvc" / "nikki"
MODEL = "nikki"


def run(stage: str, *args: str, postcondition=None) -> None:
    """Run a stage. Applio sometimes soft-fails (prints an error but exits 0),
    so each stage also asserts a postcondition on the filesystem."""
    cmd = [sys.executable, str(APPLIO / "core.py"), *args]
    print(f"\n[{time.strftime('%H:%M:%S')}] STAGE {stage}: {' '.join(args[:6])}…",
          flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(APPLIO))
    mins = (time.time() - t0) / 60
    if proc.returncode != 0:
        print(f"STAGE {stage} FAILED (exit {proc.returncode}) after {mins:.1f}min",
              flush=True)
        raise SystemExit(proc.returncode)
    if postcondition is not None and not postcondition():
        print(f"STAGE {stage} FAILED (postcondition not met - Applio soft-fail) "
              f"after {mins:.1f}min", flush=True)
        raise SystemExit(3)
    print(f"STAGE {stage} OK ({mins:.1f}min)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--sample-rate", type=int, default=40000)
    args = ap.parse_args()

    if not DATASET.exists() or not any(DATASET.glob("*.wav")):
        print("rvc_dataset/ is empty - run tools/prepare_rvc_dataset.py first")
        return 1

    logs = APPLIO / "logs" / MODEL

    run("prerequisites", "prerequisites", "--models", "True",
        "--pretraineds_hifigan", "True", "--exe", "False")
    run("preprocess", "preprocess", "--model_name", MODEL,
        "--dataset_path", str(DATASET), "--sample_rate", str(args.sample_rate),
        "--cpu_cores", "4", "--cut_preprocess", "Skip",
        "--normalization_mode", "none",
        postcondition=lambda: any((logs / "sliced_audios").glob("*.wav")))
    run("extract", "extract", "--model_name", MODEL,
        "--f0_method", "rmvpe", "--sample_rate", str(args.sample_rate),
        "--include_mutes", "2", "--cpu_cores", "4", "--gpu", "0",
        postcondition=lambda: (logs / "config.json").exists())
    run("train", "train", "--model_name", MODEL,
        "--sample_rate", str(args.sample_rate),
        "--batch_size", str(args.batch), "--total_epoch", str(args.epochs),
        "--save_every_epoch", "50", "--save_only_latest", "True",
        "--save_every_weights", "True", "--gpu", "0", "--pretrained", "True",
        "--vocoder", "HiFi-GAN", "--overtraining_detector", "True",
        "--overtraining_threshold", "50",
        postcondition=lambda: any(logs.glob("*.pth")))
    run("index", "index", "--model_name", MODEL, "--index_algorithm", "Auto",
        postcondition=lambda: any(logs.glob("*.index")))

    # ---- export ----
    logs = APPLIO / "logs" / MODEL
    weights = sorted(logs.glob(f"{MODEL}_*e_*s.pth"),
                     key=lambda p: p.stat().st_mtime)
    if not weights:  # fallback naming across Applio versions
        weights = sorted(logs.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    indexes = sorted(logs.glob("*.index"), key=lambda p: p.stat().st_mtime)
    if not weights or not indexes:
        print(f"EXPORT FAILED: no weights/index in {logs}")
        return 1
    EXPORT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights[-1], EXPORT / "nikki.pth")
    shutil.copy2(indexes[-1], EXPORT / "nikki.index")
    print(f"\nEXPORTED {weights[-1].name} + {indexes[-1].name} -> {EXPORT}")
    print("TRAINING COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
