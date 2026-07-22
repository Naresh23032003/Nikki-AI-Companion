"""Draw/picture tool: generates a real image via Pollinations (hosted
diffusion, no API key) and sends it. Covers two cases:
  - "draw/paint me a <thing>"  -> an illustration of that subject
  - "send me a pic of you / a selfie" -> a photo of HER (persona appearance),
    otherwise an image model asked for "you" invents a random stranger.
She never claims she hand-painted it - it's a genuine generated image, same
honesty rule the sing tool applies to songs."""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from urllib.parse import quote

import httpx

from app.config import ROOT
from app.tools import Tool, ToolContext

logger = logging.getLogger("companion.tools.draw")

IMAGES_DIR = ROOT / "media" / "images"

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# A request that means "a picture of HER", not an illustration of some subject:
# "send me your pic", "a selfie", "pic of you", "show me yourself", "ur pic".
_SELF_REF = re.compile(
    r"\b(your ?self|yourself|urself|a selfie|your selfie|"
    r"(pic|picture|photo|photograph|selfie|image|snap)s? of (you|urself|yourself)|"
    r"you look like|what you look|see you|of you\b|\bof u\b|your pic|ur pic|"
    r"your (face|photo|picture|pics))\b", re.I)
# The bare pronoun case ("you"/"u"/"urself") once the subject is stripped to
# essentially nothing but a self-reference.
_BARE_SELF = re.compile(r"^(you|u|yourself|urself|me|a selfie|selfie)$", re.I)


def _slug(text: str) -> str:
    return _SLUG_RE.sub("_", text.lower()).strip("_")[:40] or "drawing"


def _is_self_portrait(subject: str) -> bool:
    s = subject.strip().lower()
    return bool(_BARE_SELF.match(s) or _SELF_REF.search(s))


def _build_prompt(subject: str, ctx: ToolContext) -> tuple[str, bool]:
    """Return (image_prompt, is_selfie). Self-portrait requests are rebuilt
    around the persona's appearance so 'send me your pic' looks like HER."""
    if _is_self_portrait(subject):
        appearance = getattr(ctx.persona, "appearance", "") or ""
        if not appearance:
            appearance = "a young woman, warm friendly smile, casual style"
        prompt = (f"A natural casual selfie photo of {appearance} "
                  f"Realistic photograph, phone selfie, soft lighting, looking "
                  f"at the camera, cozy everyday setting.")
        return prompt, True
    return subject, False


async def _generate(prompt: str) -> Path:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://image.pollinations.ai/prompt/{quote(prompt)}"
    params = {"width": 1024, "height": 1024, "nologo": "true", "model": "flux"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.content
    path = IMAGES_DIR / f"{_slug(prompt)}_{int(time.time())}.jpg"
    path.write_bytes(data)
    return path


async def execute(args: dict, ctx: ToolContext) -> dict:
    subject = (args.get("subject") or "").strip(" ?!.,\"'")
    if not subject:
        return {"ok": False, "result": "no subject given - ask what to draw before trying again"}
    prompt, is_selfie = _build_prompt(subject, ctx)
    try:
        path = await _generate(prompt)
    except Exception as e:  # noqa: BLE001
        logger.warning("draw: generation failed for %r: %s", prompt, e)
        return {"ok": False, "result": f"the picture didn't come through ({e}). "
                                       f"Be honest it didn't work, stay casual."}
    if is_selfie:
        result = ("you just took a cute selfie and it's already sending - add a short "
                  "playful/teasing line if you want, but NEVER describe or caption the "
                  "photo (no brackets like '[a selfie of...]') and NEVER ask if they "
                  "want it - it's already gone, this isn't a question")
    else:
        result = (f"you just drew '{subject}' and it's already sending - a short reaction "
                  f"is fine, but NEVER describe or caption the picture in words (no "
                  f"brackets like '[a drawing of...]') and NEVER ask if they want it - "
                  f"it's already gone, this isn't a question")
    return {"ok": True,
            "image": {"url": f"/media/images/{path.name}", "path": str(path)},
            "result": result}


TOOL = Tool(
    name="draw",
    description=("Make and send a picture: either draw/paint/sketch a subject the "
                "user names, OR send a photo/selfie of yourself when they ask for "
                "your pic. Pass what they want pictured as the subject (use 'you' "
                "for a selfie of yourself)."),
    parameters={"subject": {"type": "string",
                            "description": "what to picture - a subject to draw, or "
                                           "'you'/'a selfie' for a photo of yourself"}},
    required=["subject"],
    execute=execute,
)
