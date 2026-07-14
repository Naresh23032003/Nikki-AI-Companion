"""Passive daily mood journal: she infers the USER's moods from the day's
conversations and logs them - never by asking, never with surveys.

Three jobs:
  - Nightly extraction (config: journal.nightly_time, default 23:45): pulls
    every message from every source (web chat/call/WhatsApp/tablet/iot) for
    the local calendar day just ending, and asks the LOCAL model to pull out
    mood entries grounded in concrete evidence (this data is intimate - never
    sent to the cloud brain). Quiet/flat days correctly yield 0-1 entries -
    the evidence rule below forbids filling rows. This is the ONLY step that
    needs a model: turning raw conversation into structured mood rows.
  - Nightly streak check (run_recent_streak_check, chained after extraction):
    flags a CURRENT 2-3 day rough stretch as a one-shot "Streak:" memory.
  - Weekly pattern-awareness (config: journal.weekly_pattern_day/time,
    default Sunday 23:55): recurring cross-week patterns stored as
    category="relationship" "Pattern:" memories, referenced naturally later
    (never as a report/stats-speak - see app/main.py's _pattern_note).

The streak + pattern layers are PURE CODE (no model): they run over rows that
are already structured (date, mood_label, intensity, why), so it's counting
and grouping - deterministic, unit-testable, zero hallucination risk. The
stored sentences are plain and factual on purpose: she rephrases them in her
own voice at reference time anyway (the _pattern_note/_streak_note prompt
notes), so model-written prose here bought nothing but failure modes.

See app/db.py for the mood_journal table + CRUD.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("companion.journal")

_SOURCE_LABELS = {
    "webapp_chat": "web chat", "webapp_call": "phone call",
    "whatsapp": "whatsapp", "tablet": "tablet", "iot": "device",
}

_EXTRACTION_SYSTEM = """\
You are inferring the USER's moods for one day from their conversations with \
their companion, for a private mood journal. Be STRICT and conservative - it \
is much better to log nothing than to invent a feeling that wasn't there.

EVIDENCE RULE (hard requirement):
- Every entry MUST be grounded in something the user actually said or a
  concrete thing that happened in the conversation that day. `why` must
  reference that concrete evidence in one short sentence.
- NEVER fabricate or guess a mood just to fill in the day. A quiet, flat, or
  purely logistical day means 0 or 1 entries - that is the CORRECT output,
  not a failure.
- Only log entries about the USER's own mood/feelings - never the
  companion's, never a mood merely mentioned about a third person.
- If the user's mood clearly shifted during the day (e.g. stressed in the
  afternoon, relieved by night), log multiple entries - one per genuine
  shift, each with its own time and evidence.
- `time` is your best estimate of when that mood showed, in 24h HH:MM local
  time, based on the message timestamps given.
- `intensity` is 1-5 (1 = mild, 5 = very strong), judged from the user's own
  words/tone - do not default to a fixed number.
- `mood_label` is a short single word/phrase (e.g. "happy", "stressed",
  "anxious", "content", "lonely", "excited", "frustrated", "tired",
  "hopeful", "overwhelmed", "grateful", "hurt", "proud", "bored") - pick
  whatever plain word actually fits; do not force it into a fixed list if
  none fit well.
- `source_channel` is whichever channel that evidence came from (given per
  message below).

Respond with STRICT JSON only: {"entries": [{"time": "HH:MM", "mood_label": \
"...", "intensity": 1, "why": "...", "source_channel": "..."}]}
If there is no real evidence of any mood that day: {"entries": []}"""

# ---------------------------------------------------------------------------
# Programmatic mood classification (pattern/streak layer - no model involved)
# ---------------------------------------------------------------------------
# The extraction prompt suggests plain single-word labels but doesn't force a
# fixed list, so classify by lexicon + a stem fallback. Unknown labels count
# as neither (never guessed negative).
_NEGATIVE_MOODS = {
    "sad", "down", "low", "upset", "hurt", "angry", "mad", "frustrated",
    "stressed", "anxious", "worried", "nervous", "scared", "afraid",
    "lonely", "isolated", "tired", "exhausted", "drained", "burnt out",
    "burned out", "overwhelmed", "hopeless", "depressed", "miserable",
    "disappointed", "discouraged", "annoyed", "irritated", "gloomy",
    "heartbroken", "homesick", "insecure", "jealous", "guilty", "ashamed",
    "restless", "unmotivated", "numb", "empty", "defeated", "helpless",
}
_NEGATIVE_STEMS = ("sad", "stress", "anx", "worri", "lonel", "exhaust",
                   "overwhelm", "frustrat", "depress", "hopeless", "tired",
                   "drain", "upset", "hurt", "discourag", "disappoint")
_POSITIVE_MOODS = {
    "happy", "excited", "content", "grateful", "proud", "hopeful", "calm",
    "relieved", "joyful", "cheerful", "optimistic", "energized", "loved",
    "confident", "peaceful", "playful", "amused", "inspired", "motivated",
}


def _is_negative_mood(label: str) -> bool:
    lbl = (label or "").strip().lower()
    if not lbl:
        return False
    if lbl in _POSITIVE_MOODS:
        return False
    if lbl in _NEGATIVE_MOODS:
        return True
    return any(lbl.startswith(stem) for stem in _NEGATIVE_STEMS)


def _negative_entries_by_date(entries: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for e in entries:
        if _is_negative_mood(e.get("mood_label", "")):
            out.setdefault(e["date"], []).append(e)
    return out


def _dominant_labels(entries: list[dict], top: int = 2) -> list[str]:
    """Most frequent negative labels, ties broken by total intensity."""
    counts: dict[str, list[int]] = {}
    for e in entries:
        counts.setdefault(e["mood_label"], [0, 0])
        counts[e["mood_label"]][0] += 1
        counts[e["mood_label"]][1] += int(e.get("intensity") or 3)
    ranked = sorted(counts, key=lambda k: (counts[k][0], counts[k][1]), reverse=True)
    return ranked[:top]


def _local_day_bounds_utc(day_offset: int = 0) -> tuple[str, str, str]:
    """Local-calendar-day boundaries, expressed as UTC ISO timestamps for the
    messages table (which stores UTC). Returns (date_label, start_utc, end_utc)."""
    local_now = datetime.now().astimezone()
    target = (local_now + timedelta(days=day_offset)).date()
    start_local = datetime.combine(target, datetime.min.time()).astimezone()
    end_local = start_local + timedelta(days=1)
    return (
        target.isoformat(),
        start_local.astimezone(timezone.utc).isoformat(),
        end_local.astimezone(timezone.utc).isoformat(),
    )


def _transcript_for(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        try:
            local_ts = datetime.fromisoformat(m["timestamp"]).astimezone()
            hhmm = local_ts.strftime("%H:%M")
        except (ValueError, TypeError):
            hhmm = "??:??"
        who = "user" if m["role"] == "user" else "her"
        channel = _SOURCE_LABELS.get(m.get("source") or "", m.get("source") or "unknown")
        lines.append(f"[{hhmm} · {channel}] {who}: {m['content']}")
    return "\n".join(lines)


def _clean_entries(raw_entries) -> list[dict]:
    out = []
    if not isinstance(raw_entries, list):
        return out
    for e in raw_entries:
        if not isinstance(e, dict):
            continue
        mood = str(e.get("mood_label") or "").strip().lower()
        why = str(e.get("why") or "").strip()
        if not mood or len(why) < 8:
            continue  # no real evidence -> drop rather than guess
        try:
            intensity = max(1, min(5, int(float(e.get("intensity", 3)))))
        except (TypeError, ValueError):
            intensity = 3
        time_val = str(e.get("time") or "").strip()
        if not _looks_like_hhmm(time_val):
            time_val = None
        source = str(e.get("source_channel") or "").strip().lower() or None
        out.append({"mood_label": mood[:40], "intensity": intensity,
                    "why": why[:300], "time": time_val, "source_channel": source})
    return out


def _looks_like_hhmm(s: str) -> bool:
    if not s or ":" not in s:
        return False
    h, _, m = s.partition(":")
    return h.strip().isdigit() and m.strip()[:2].isdigit()


async def run_nightly_extraction(db, llm, settings, day_offset: int = -1) -> int:
    """Extract + store mood entries for one local calendar day (default:
    yesterday relative to now, i.e. the day that JUST ended at the configured
    nightly_time). Returns the number of entries stored."""
    jcfg = (settings.raw or {}).get("journal", {})
    date_label, start_utc, end_utc = _local_day_bounds_utc(day_offset)

    messages = db.messages_between(start_utc, end_utc)
    user_said_something = any(m["role"] == "user" for m in messages)
    if not user_said_something:
        logger.info("journal: nightly %s - no user messages, nothing to extract", date_label)
        return 0

    transcript = _transcript_for(messages)
    model = jcfg.get("extract_model") or settings.ollama_extract_model
    try:
        raw = await llm.chat(
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM},
                {"role": "user", "content": f"Today's date: {date_label}\n\n{transcript}"},
            ],
            format="json", model=model, options={"temperature": 0.2},
            # Batch job, not a live reply: a full day's transcript through a
            # bigger model (plus the model swap) blows the 120s client
            # default - a busy day timed out and simply never got extracted.
            timeout=600.0,
            # Unload the 8B immediately after this one-off job. With the
            # global 2h keep_alive it sat in VRAM past midnight and starved
            # the studio TTS (voice notes silently fell back to the stock
            # Kokoro voice for 2 hours every night).
            keep_alive=0,
        )
        data = json.loads(raw)
    except Exception as e:  # noqa: BLE001 - a failed nightly run must never crash the scheduler
        # %r, not %s - httpx.ReadTimeout stringifies to "", which logged an
        # empty reason and made timeouts undiagnosable.
        logger.warning("journal: nightly extraction failed for %s: %r", date_label, e)
        return 0

    entries = _clean_entries(data.get("entries"))
    for entry in entries:
        db.add_mood_entry(date_label, entry["time"], entry["mood_label"],
                          entry["intensity"], entry["why"], entry["source_channel"])
    logger.info("journal: nightly %s - stored %d entr%s", date_label, len(entries),
               "y" if len(entries) == 1 else "ies")
    return len(entries)


def _weekday_pattern(neg_by_date: dict[str, list[dict]]) -> str | None:
    """Recurring low weekday: the same weekday negative on 2+ distinct dates.
    Two occurrences of the same weekday are, by definition, in different
    calendar weeks - so this can never fire on a single bad Mon-Tue-Wed
    stretch (the original complaint about the model-based version)."""
    by_weekday: dict[int, list[tuple[str, dict]]] = {}
    for d, day_entries in neg_by_date.items():
        wd = datetime.fromisoformat(d).weekday()
        for e in day_entries:
            by_weekday.setdefault(wd, []).append((d, e))
    best = None
    for wd, hits in by_weekday.items():
        dates = {d for d, _ in hits}
        if len(dates) < 2:
            continue
        if best is None or len(dates) > len(best[1]):
            best = (wd, dates, [e for _, e in hits])
    if best is None:
        return None
    wd, dates, hit_entries = best
    weekday_name = ["Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday"][wd]
    label = _dominant_labels(hit_entries, top=1)[0]
    evening = sum(1 for e in hit_entries
                  if (e.get("time") or "").strip()[:2].isdigit()
                  and int((e.get("time") or "0")[:2]) >= 18)
    when = f"{weekday_name} evenings" if evening >= max(1, round(len(hit_entries) * 0.7)) \
        else f"{weekday_name}s"
    return f"They tend to feel {label} on {when}"


def _persistent_mood_pattern(neg_by_date: dict[str, list[dict]],
                             skip_label: str | None = None) -> str | None:
    """One negative label recurring on 3+ distinct dates spanning 2+ weeks."""
    date_sets: dict[str, set[str]] = {}
    for d, day_entries in neg_by_date.items():
        for e in day_entries:
            date_sets.setdefault(e["mood_label"], set()).add(d)
    best = None
    for label, dates in date_sets.items():
        if label == skip_label or len(dates) < 3:
            continue
        weeks = {datetime.fromisoformat(d).isocalendar()[:2] for d in dates}
        if len(weeks) < 2:
            continue  # one hard week is not a pattern
        if best is None or len(dates) > len(best[1]):
            best = (label, dates)
    if best is None:
        return None
    return f"They've been feeling {best[0]} a lot lately - it's come up on several different days"


def _trend_pattern(neg_by_date: dict[str, list[dict]], today) -> str | None:
    """Clearly harder (or clearly lighter) recent fortnight vs the one before.

    The "harder lately" branch additionally requires the recent negative days
    to span 2+ distinct calendar weeks - one rough Mon-Tue-Wed is the streak
    check's job (flagged within days), not a weekly trend; without this it
    re-reported exactly the single-bad-week evidence every detector here is
    supposed to reject."""
    recent_cut = today - timedelta(days=14)
    recent = {d for d in neg_by_date if datetime.fromisoformat(d).date() > recent_cut}
    earlier = set(neg_by_date) - recent
    recent_weeks = {datetime.fromisoformat(d).isocalendar()[:2] for d in recent}
    if (len(recent) >= 3 and len(recent_weeks) >= 2
            and len(recent) >= 2 * max(1, len(earlier))):
        return "The last couple of weeks have seemed noticeably harder for them than before"
    if len(earlier) >= 3 and len(recent) <= len(earlier) // 2:
        return "They've seemed noticeably lighter lately compared to a rougher patch a few weeks ago"
    return None


async def run_weekly_patterns(db, memory, settings=None, lookback_days: int = 30) -> int:
    """Look for gentle recurring patterns in the recent journal and store them
    as category="relationship" memories. Returns the number stored.

    Pure code over structured rows (see module docstring): weekday recurrence,
    persistent mood, and fortnight trend. Every detector requires evidence
    spanning 2+ distinct calendar weeks, so a single bad day - or one bad
    Mon-Tue-Wed week - can never register as a "pattern". The stored text is
    deliberately plain: she phrases it warmly in-voice at reference time."""
    since = (datetime.now().date() - timedelta(days=lookback_days)).isoformat()
    entries = db.list_mood_entries(since_date=since)
    if len(entries) < 5:
        logger.info("journal: weekly pass - only %d entries in %dd, skipping", len(entries), lookback_days)
        return 0

    neg_by_date = _negative_entries_by_date(entries)
    texts: list[str] = []
    weekday = _weekday_pattern(neg_by_date)
    if weekday:
        texts.append(weekday)
    # Don't double-report the same label as both "on Mondays" and "a lot lately".
    weekday_label = weekday.split("feel ", 1)[-1].split(" on ")[0] if weekday else None
    persistent = _persistent_mood_pattern(neg_by_date, skip_label=weekday_label)
    if persistent:
        texts.append(persistent)
    trend = _trend_pattern(neg_by_date, datetime.now().date())
    if trend:
        texts.append(trend)

    stored = 0
    for text in texts[:3]:
        # add_fact's embedding dedup (>=0.92 cosine) updates the existing
        # Pattern: row on weekly reruns instead of stacking near-duplicates.
        memory_id = await memory.add_fact(f"Pattern: {text}", "relationship", source="mood_journal")
        if memory_id is not None:
            stored += 1
    logger.info("journal: weekly pass - stored %d pattern(s): %s", stored, texts[:3])
    return stored


async def run_recent_streak_check(db, memory, settings=None, lookback_days: int = 3) -> bool:
    """Short-window check (default: last 3 days) for a CURRENT rough streak -
    distinct from run_weekly_patterns' long-term, cross-week pattern check.
    Runs nightly (chained right after run_nightly_extraction - see main.py's
    _run_nightly_journal) so a Mon-Tue-Wed rough stretch is available to
    reference starting Thursday, instead of sitting unmentioned until it
    happens to recur weeks later on the Sunday weekly pass.

    Pure code (see module docstring): a streak = negative-mood entries on 2+
    distinct days within the window. The stored sentence is plain and factual
    - she rephrases it warmly in her own voice at reference time.

    Stored as a one-shot 'Streak: ...' memory: consumed (deleted) the first
    time she brings it up (app/main.py's _streak_note), unlike a genuine
    Pattern: which stays referenceable indefinitely - a rough few days is a
    point-in-time thing to check in on once, not an enduring trait.
    """
    # Don't stack a second streak note while one's still waiting to be
    # brought up - the overlapping 3-day windows would otherwise re-flag the
    # same rough days again every single night until it's referenced.
    existing = [m for m in db.list_memories_by_category("relationship")
               if (m.get("fact") or "").startswith("Streak:")]
    if existing:
        return False

    since = (datetime.now().date() - timedelta(days=lookback_days - 1)).isoformat()
    entries = db.list_mood_entries(since_date=since)
    neg_by_date = _negative_entries_by_date(entries)
    if len(neg_by_date) < 2:
        return False  # a single bad day alone is not a streak

    neg_entries = [e for day in neg_by_date.values() for e in day]
    labels = _dominant_labels(neg_entries, top=2)
    label_str = labels[0] if len(labels) == 1 or labels[1] == labels[0] \
        else f"{labels[0]} and {labels[1]}"
    strongest = max(neg_entries, key=lambda e: int(e.get("intensity") or 0))
    why = (strongest.get("why") or "").strip().rstrip(".")[:140]
    text = (f"They've seemed {label_str} on {len(neg_by_date)} of the last "
            f"{lookback_days} days" + (f" - most recently: {why}" if why else ""))

    memory_id = await memory.add_fact(f"Streak: {text}", "relationship", source="mood_journal")
    logger.info("journal: streak check - flagged %r (memory #%s)", text, memory_id)
    return memory_id is not None
