# Training her RVC voice model (Applio, RTX 3060 6GB)

The RVC model is her **timbre** — it powers real-time calls (Kokoro→RVC) and
singing covers.

## Fully headless (recommended)

Applio is cloned to `applio/` with its own isolated venv. One command does
prerequisites → preprocess → extract → train → index → export:

```bash
applio/env/Scripts/python.exe tools/train_rvc_headless.py --epochs 300 --batch 4
```

Inference runs as a warm worker (keep it running whenever
`voice.call_voice: kokoro_rvc`):

```bash
applio/env/Scripts/python.exe tools/rvc_server.py     # port 3002
```

The GUI walkthrough below is the manual alternative / for retraining by hand.

## 0. Prepare the dataset (this repo)

```bash
python tools/process_songs.py           # separate singing from keyboard
# LISTEN to the flagged files, delete any with audible keyboard under her voice
python tools/prepare_rvc_dataset.py     # gold speech + song vocals -> rvc_dataset/
```
Aim for **20+ minutes** total; 45–60 min is the sweet spot.

## 1. Install Applio

Grab the latest release: https://github.com/IAHispano/Applio (Windows
one-click zip). Run `run-applio.bat` → opens the web UI.

## 2. Train (exact settings for the 3060 6GB)

**Preprocess tab**
- Model name: `nikki`
- Dataset path: `<repo>\rvc_dataset`
- Sample rate: **40k** (best quality/VRAM balance on 6GB)
- Preprocess → wait for "completed"

**Extract tab**
- F0 method: **rmvpe** (best pitch tracker; handles singing)
- Embedder: contentvec (default)
- Extract → wait

**Train tab**
- Sample rate 40k, batch size: **4** (raise to 6–8 only if VRAM allows —
  watch Task Manager; OOM = lower it)
- Epochs: **300** (sweet spot is usually 200–400 for ~30 min of data;
  save-every 50 and audition checkpoints — stop when test conversions stop
  improving; overtraining sounds metallic)
- Pretrained: keep defaults; "Save only latest" ON to spare disk
- Train, then **Generate index** when done

**Export**
- Copy from Applio's `logs/nikki/`:
  - `nikki.pth`            → `<repo>/voices/rvc/nikki/nikki.pth`
  - `added_*_nikki.index`  → `<repo>/voices/rvc/nikki/nikki.index`

## 3. Validate

```bash
pip install rvc-python          # inference lib used by the app
python tools/validate_rvc.py    # writes before/after pairs to voices/rvc/validation/
```
Listen: it should be unmistakably HER on speech and on the vocal snippet.

## 4. Turn it on

`config.yaml` → `voice.call_voice: kokoro_rvc` and restart. Instant rollback:
set back to `kokoro_raw`.

## Growth path (optional, later)

- **Silver clips** (usable but untrimmed):
  `python tools/prepare_rvc_dataset.py --trim-silver path/to/silver/`
- **Bronze clips** (speech over music/noise): run them through
  `tools/process_songs.py` first (same Demucs flow), listen-check, then
  include the survivors the same way. Retrain with the bigger dataset for a
  quality bump — same settings, fresh model name (`nikki_v2`), and update
  `voice.rvc_model_dir` when you're happy.
