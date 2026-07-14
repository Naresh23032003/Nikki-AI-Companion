"""Weather tool: Open-Meteo (free, no API key). Default city from config.yaml
`tools.weather.default_city` (falls back to Chennai). Morning proactive texts
can weave the forecast in naturally (see app/proactive.py)."""
from __future__ import annotations

import httpx

from app.tools import Tool, ToolContext


async def execute(args: dict, ctx: ToolContext) -> dict:
    city = args.get("city") or (
        (ctx.settings.raw or {}).get("tools", {}).get("weather", {}).get("default_city", "Chennai")
    )
    # A small model with nowhere to put "tomorrow" will otherwise stuff it
    # into `city` (city="tomorrow" -> geocoding fails). Giving it a real slot
    # for the day fixes that at the source.
    which_day = str(args.get("day") or "today").strip().lower()
    idx = 1 if which_day == "tomorrow" else 0
    async with httpx.AsyncClient(timeout=10.0) as c:
        g = await c.get("https://geocoding-api.open-meteo.com/v1/search",
                        params={"name": city, "count": 1})
        g.raise_for_status()
        hits = g.json().get("results") or []
        if not hits:
            return {"ok": False, "result": f"could not find city '{city}'"}
        lat, lon = hits[0]["latitude"], hits[0]["longitude"]
        place = hits[0]["name"]
        w = await c.get("https://api.open-meteo.com/v1/forecast",
                        params={"latitude": lat, "longitude": lon,
                                "current": "temperature_2m,apparent_temperature,"
                                           "precipitation,weather_code",
                                "daily": "temperature_2m_max,temperature_2m_min,"
                                         "precipitation_probability_max",
                                "timezone": "auto", "forecast_days": 2})
        w.raise_for_status()
        d = w.json()
    cur = d.get("current", {})
    daily = d.get("daily", {})
    day_max = (daily.get("temperature_2m_max") or [None, None])
    day_min = (daily.get("temperature_2m_min") or [None, None])
    rain = (daily.get("precipitation_probability_max") or [None, None])
    day_max = day_max[idx] if idx < len(day_max) else None
    day_min = day_min[idx] if idx < len(day_min) else None
    rain_p = rain[idx] if idx < len(rain) else None
    label = "tomorrow" if idx == 1 else "today"
    if idx == 0:
        return {"ok": True, "result": (
            f"weather in {place}: {cur.get('temperature_2m')}°C now "
            f"(feels {cur.get('apparent_temperature')}°C), {label} "
            f"{day_min}-{day_max}°C, rain chance {rain_p}%")}
    return {"ok": True, "result": (
        f"forecast for {place} {label}: {day_min}-{day_max}°C, rain chance {rain_p}%")}


TOOL = Tool(
    name="weather",
    description="Get the current weather, or tomorrow's forecast, for a city.",
    parameters={
        "city": {"type": "string", "description": "City name (optional, defaults to their home city)"},
        "day": {"type": "string", "enum": ["today", "tomorrow"],
                "description": "Which day's forecast - defaults to today"},
    },
    required=[],
    execute=execute,
)
