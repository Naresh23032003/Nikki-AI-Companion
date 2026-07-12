# Animated avatar layer spec

The Call-mode avatar is a **layered 2D sprite** (VTuber-style) — not video. The
renderer stacks transparent PNG layers on a canvas and swaps the eye/mouth layers
in real time for blinking and lip-sync. This is the ONLY place the avatar
animates; the chat screen always shows the static profile photo.

Each persona has its own folder named after its `avatar_id` (from the persona
YAML), e.g. `luna_default` → `frontend/public/avatars/luna_default/`.

The files here are **programmatically generated placeholders** so the system
works before real art exists. Replace them with real art using the exact same
names, dimensions, and alignment and everything keeps working.

## Files (all required)

| File                     | Purpose                                  |
|--------------------------|------------------------------------------|
| `base.png`               | Everything except eyes & mouth: head, hair, body, nose, brows, blush. Leaves the eye and mouth areas empty (transparent/skin) so the layers on top show through. |
| `eyes_open.png`          | Both eyes, open. Overlaid on base.       |
| `eyes_closed.png`        | Both eyes, closed (blink frame).         |
| `mouth_closed.png`       | Viseme: mouth closed / resting.          |
| `mouth_slightly_open.png`| Viseme: small opening (l, t, d, n, s…).  |
| `mouth_open.png`         | Viseme: open vowel (ah).                 |
| `mouth_wide.png`         | Viseme: wide smile / "aa".               |
| `mouth_rounded.png`      | Viseme: rounded "O/U/W".                 |
| `mouth_teeth.png`        | Viseme: "E" / teeth showing.             |

## Canvas & alignment

- **Dimensions:** every layer is exactly **1024 × 1024 px**, PNG with alpha.
- **Registration:** all layers share the same 1024×1024 frame and are pre-aligned
  — a layer is just composited at (0,0). Don't crop or offset individual layers.
- **Face anchor (where features sit in the placeholder):**
  - Head center ≈ `(512, 430)`; face ≈ 500 wide × 600 tall.
  - Eyes centered at `y ≈ 440`, left/right at `x ≈ 394 / 630`.
  - Mouth centered at `(512, 624)`, ≈ 120–200 px wide depending on viseme.
- Keep a transparent margin around the character so head-tilt/bounce animation
  doesn't clip at the edges.

## Layer stacking order (bottom → top)

```
base  →  eyes_(open|closed)  →  mouth_<viseme>
```

The renderer picks exactly one eye layer and one mouth layer each frame.

## How the renderer uses them

- **Blink:** swaps `eyes_open` → `eyes_closed` for ~120 ms every 2–6 s (randomized).
- **Breathing:** the whole sprite drifts vertically on a slow sine (idle life).
- **Lip-sync:** Phase 4 TTS `timings` are mapped to the 6 visemes and the mouth
  layer is swapped in sync with playback using `AudioContext.currentTime`.
- **Emotions:** the reply's emotion tag (happy, laughing, shy, sad, surprised,
  neutral, love) drives eye choice, head tilt/bounce, and particle effects
  (hearts/sparkles). If you add emotion-specific eye art later, name them
  `eyes_<emotion>.png` and the renderer will prefer them (falls back to
  `eyes_open`).

## Video mode (best fidelity — recommended)

If the avatar folder contains **`idle.mp4`** and **`talking.mp4`**, the renderer
uses them instead of the sprite layers: the idle loop plays while she's quiet
and cross-fades to the talking loop while she speaks. This is real-time and
cheap (plain `<video>` decode), so it works on modest hardware — and it can look
photoreal because the hard work happened offline.

**Clip specs**
- 512×512 (or any square), **6–10 s**, seamless loop, no audio track.
- H.264 MP4, ~500 kbps is plenty (`ffmpeg -i in.mp4 -vf scale=512:512 -an -b:v 500k out.mp4`).
- `idle.mp4`: subtle motion — breathing, occasional blink, tiny head drift.
- `talking.mp4`: same framing, mouth moving naturally in conversation.

**Ways to make the clips**
1. Generate one portrait image of the persona, then use an offline talking-head
   tool (e.g. SadTalker, LivePortrait, EchoMimic) to render a short "speaking"
   clip and a calm clip — one-time render, any speed is fine.
2. Or record a consenting real person: 8 s sitting still + 8 s chatting.

Mode priority: `idle.mp4`+`talking.mp4` → sprite layers → static photo.

## Photo-mode fallback

If a persona has no avatar folder (only the static profile photo), Call mode
shows the photo large with gentle motion and an **audio-reactive glow** while she
speaks. So a persona works with zero art — art just makes it richer.
