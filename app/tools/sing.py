"""Sing tool: library hit -> instant song; miss -> queue a cover + deferred
in-character line. She never claims to sing live on calls (see persona.py)."""
from __future__ import annotations

import re

from app.tools import Tool, ToolContext


# A library entry whose title is still just its raw export filename (never
# manually retitled - see app/covers.py's "(edit moods in the .json)" note)
# reads as broken if repeated verbatim ("wanna give sing_05_39s_vocals a
# listen?"). Detect that and let her tease it without naming it instead.
_AUTO_TITLE = re.compile(r"^[\w\-]*_vocals$", re.I)


async def execute(args: dict, ctx: ToolContext) -> dict:
    covers = ctx.covers
    if covers is None:
        return {"ok": False, "result": "singing isn't set up yet"}

    query = (args.get("song") or "").strip(" ?!.,\"'")
    mood = None
    if re.search(r"\bcan'?t sleep|sleepy|lullaby\b", query, re.I):
        mood = "lullaby"
    elif re.search(r"\bbirthday\b", query, re.I):
        mood = "birthday"

    # For a random pick (no specific title asked), don't hand over the same
    # track we just sent - "send another song" should actually be another one.
    last_file = ctx.db.get_setting("last_song_file") if ctx.db and not query else None
    song = covers.find(query or None, mood, exclude_file=last_file)
    if song:
        title = song.get("title", "")
        if ctx.db and song.get("file"):
            ctx.db.set_setting("last_song_file", song["file"])
        if _AUTO_TITLE.match(title):
            return {"ok": True, "song": song,
                    "result": "your song is already sending - it hasn't been given a "
                              "proper title yet, so tease that you found one without "
                              "naming it. NEVER describe/caption the audio in brackets "
                              "and NEVER ask if they want it - it's already gone"}
        return {"ok": True, "song": song,
                "result": f"'{title}' is already sending to them now - a short reaction "
                          f"is fine, but NEVER describe/caption the audio in brackets and "
                          f"NEVER ask if they want it - it's already gone, not a question"}
    if query:
        return {"ok": False, "queue_query": query,
                "result": f"'{query}' isn't in your library - tell them you'll send it "
                          f"in a bit (never claim you can sing live), it's being prepared"}
    return {"ok": False, "result": "no songs in your library yet - be honest and "
                                   "playful about owing them one"}


TOOL = Tool(
    name="sing",
    description="Sing or send a song for the user, optionally a specific title or mood (lullaby, birthday).",
    parameters={"song": {"type": "string", "description": "song title or mood requested, if any"}},
    required=[],
    execute=execute,
)
