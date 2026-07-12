"""zomato_suggest: SUGGESTION only — there is no public Zomato ordering API.
Returns a dish/restaurant idea plus a Zomato search deep link. Never claims an
order was placed. Config-gated: tools.zomato.enabled (default true; no auth,
no API key — it's just a search URL)."""
from __future__ import annotations

from urllib.parse import quote

from app.tools import Tool, ToolContext


async def execute(args: dict, ctx: ToolContext) -> dict:
    dish = (args.get("dish") or "").strip()
    cfg = (ctx.settings.raw or {}).get("tools", {}).get("zomato", {})
    city = cfg.get("city", "chennai").lower()
    query = dish or "food near me"
    url = f"https://www.zomato.com/{quote(city)}/search?q={quote(query)}"
    suggestion = dish or "something good"
    return {"ok": True, "result": (
        f"suggestion only (you can't actually order): mention craving "
        f"'{suggestion}' and share this Zomato search link: {url}. "
        f"Never say you ordered or that it's on the way.")}


def _enabled(settings) -> bool:
    return bool((settings.raw or {}).get("tools", {}).get("zomato", {}).get("enabled", True))


TOOL = Tool(
    name="zomato_suggest",
    description=("Use when the user asks what to eat, wants a food/restaurant "
                "suggestion, or asks you to recommend a dish. Suggests a dish or "
                "restaurant with a Zomato search link — suggestion only, cannot "
                "place orders."),
    parameters={"dish": {"type": "string", "description": "craving or dish name, if mentioned"}},
    required=[],
    execute=execute,
    enabled=_enabled,
)
