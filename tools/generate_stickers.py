"""One-off script: generate a batch of ORIGINAL reaction stickers per emotion
category via Pollinations (same free hosted-diffusion endpoint as the `draw`
tool) and drop them into stickers/<kind>/. Not a downloader of real WhatsApp
sticker packs - those are almost universally copyrighted fan art / show
screenshots, so this generates fresh art instead.

Run manually: python tools/generate_stickers.py
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parent.parent
STICKER_ROOT = ROOT / "stickers"

# A few distinct prompts per mood - flat cartoon sticker style, consistent
# character (dark brown hair, matches the existing avatar art), plain
# background so whatsapp-web.js's sticker conversion crops cleanly.
_STYLE = ("cute chibi cartoon girl with dark brown hair, flat vector sticker "
          "art, bold clean outlines, simple plain white background, "
          "WhatsApp sticker style, no text, no watermark")

PROMPTS: dict[str, list[str]] = {
    "happy": [
        f"{_STYLE}, big genuine smile, sparkling eyes, cheerful",
        f"{_STYLE}, laughing softly with joy, rosy cheeks",
        f"{_STYLE}, giving a big thumbs up, excited grin",
        f"{_STYLE}, jumping with happiness, arms up in the air",
    ],
    "laughing": [
        f"{_STYLE}, laughing hard with eyes closed, head tilted back",
        f"{_STYLE}, giggling behind her hand, amused expression",
        f"{_STYLE}, cracking up, tears of laughter, wide open mouth smile",
        f"{_STYLE}, chuckling and pointing, teasing laugh",
    ],
    "shy": [
        f"{_STYLE}, blushing shyly, covering half her face with both hands",
        f"{_STYLE}, bashful smile, looking away, red cheeks",
        f"{_STYLE}, peeking out from behind her hair, timid expression",
        f"{_STYLE}, twiddling fingers nervously, shy smile",
    ],
    "love": [
        f"{_STYLE}, holding a big red heart, loving expression",
        f"{_STYLE}, blowing a kiss, heart-shaped hands",
        f"{_STYLE}, hugging a heart pillow, dreamy loving eyes",
        f"{_STYLE}, surrounded by floating small hearts, warm smile",
    ],
    "sad": [
        f"{_STYLE}, a single tear, downcast sad eyes, small frown",
        f"{_STYLE}, hugging her knees looking gloomy",
        f"{_STYLE}, holding a wilted flower, melancholy expression",
        f"{_STYLE}, under a small raincloud, pouting sadly",
    ],
    "miss_you": [
        f"{_STYLE}, hugging a pillow tightly, longing expression",
        f"{_STYLE}, looking at a phone waiting for a text, wistful",
        f"{_STYLE}, waving softly with a hopeful, missing-you look",
        f"{_STYLE}, holding a heart, looking out a window longingly",
    ],
    "good_morning": [
        f"{_STYLE}, big yawn and stretch, morning sunshine background, sleepy smile",
        f"{_STYLE}, holding a steaming coffee mug, bright morning smile",
        f"{_STYLE}, waking up in bed with messy hair, cheerful morning wave",
        f"{_STYLE}, opening curtains to sunlight, happy morning stretch",
    ],
    "good_night": [
        f"{_STYLE}, hugging a pillow, sleepy eyes, crescent moon and stars background",
        f"{_STYLE}, wearing pajamas, yawning under a blanket, cozy",
        f"{_STYLE}, tucked into bed, soft goodnight wave, moonlight",
        f"{_STYLE}, holding a small nightlight, drowsy peaceful smile",
    ],
    "angry_cute": [
        f"{_STYLE}, puffed cheeks pouting angrily, cute frown, arms crossed",
        f"{_STYLE}, stomping one foot, cartoon angry steam puff, pouty",
        f"{_STYLE}, crossed arms and a cute glare, annoyed pout",
        f"{_STYLE}, comically fuming with tiny cartoon anger marks, pouting",
    ],
}


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:30] or "sticker"


async def _fetch(client: httpx.AsyncClient, prompt: str, seed: int) -> bytes:
    url = f"https://image.pollinations.ai/prompt/{quote(prompt)}"
    params = {"width": 512, "height": 512, "nologo": "true", "model": "flux", "seed": seed}
    for attempt in range(5):
        r = await client.get(url, params=params, timeout=90.0)
        if r.status_code == 429:
            wait = 5.0 * (attempt + 1)
            print(f"    429, backing off {wait:.0f}s (attempt {attempt + 1}/5)...")
            await asyncio.sleep(wait)
            continue
        r.raise_for_status()
        return r.content
    r.raise_for_status()
    return r.content


def _already_done(kind: str, idx: int) -> bool:
    folder = STICKER_ROOT / kind
    return any(folder.glob(f"{kind}_{idx:02d}_*.png"))


async def main() -> None:
    total = 0

    async with httpx.AsyncClient() as client:
        for kind, prompts in PROMPTS.items():
            print(f"'{kind}':")
            for idx, prompt in enumerate(prompts, start=1):
                if _already_done(kind, idx):
                    print(f"  [{kind}] #{idx} already exists, skipping")
                    total += 1
                    continue
                try:
                    data = await _fetch(client, prompt, seed=1000 + idx)
                except Exception as e:  # noqa: BLE001
                    print(f"  [{kind}] #{idx} FAILED: {e}")
                    continue
                folder = STICKER_ROOT / kind
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{kind}_{idx:02d}_{int(time.time())}.png"
                path.write_bytes(data)
                total += 1
                print(f"  [{kind}] #{idx} -> {path.name} ({len(data)} bytes)")
                await asyncio.sleep(3.0)  # be polite to the free hosted endpoint

    print(f"\nDone: {total}/{sum(len(p) for p in PROMPTS.values())} stickers generated.")


if __name__ == "__main__":
    asyncio.run(main())
