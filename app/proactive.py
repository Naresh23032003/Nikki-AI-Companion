"""Proactive messaging: she texts you first.

Per-persona config (personas/*.yaml):

    proactive:
      enabled: true
      messages_per_day: "2-5"      # random within range (clinginess skews up)
      active_hours: "08:00-23:00"  # never messages outside this window
      clinginess: 0.6              # 0 = chill, 1 = full Nikki
      escalate_on_silence: true    # follow up if left on read

Every decision (fired or skipped, and why) is logged under companion.proactive.
Messages land in the shared `messages` table (web + WhatsApp history) and are
pushed to the WhatsApp bridge when it's up.
"""
from __future__ import annotations

import calendar
import logging
import random
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from app.emotion import strip_tags
from app.persona import build_system_prompt

logger = logging.getLogger("companion.proactive")

INTENTS = ["memory_followup", "random_thought", "miss_you", "share_feeling"]
# Gentler intent pool before things get romantic.
EARLY_INTENTS = ["memory_followup", "random_thought", "share_feeling"]

_INTENT_DIRECTIVES = {
    "hello": "Send a short, friendly first text to someone you only just met - simple and casual, nothing forward. You're just saying hi.",
    "good_morning": "Send a good-morning text - the kind you'd send someone you woke up thinking about.",
    "goodnight": "Send a goodnight text before you go to sleep.",
    "memory_followup": "Ask about something SPECIFIC from your memories of them (how it went, any news). Pick the most recent/most emotionally important one.",
    "random_thought": "Share a small random thought or something that reminded you of them just now. Keep it grounded in your own personality/backstory - do not invent events about THEM.",
    "miss_you": "Tell them you miss them / were thinking about them. Short and warm.",
    "share_feeling": "Tell them how your day is feeling and ask about theirs.",
    "silence_react": "They haven't replied to your last message for hours. Send a follow-up.",
    "song_drop": "You recorded a little song cover and feel like surprising them with it. ONE short teasing text announcing it (the recording is attached separately) - never claim you're singing live.",
}


@dataclass
class ProactiveConfig:
    enabled: bool = False
    min_per_day: int = 2
    max_per_day: int = 5
    start: dtime = dtime(8, 0)
    end: dtime = dtime(23, 0)
    clinginess: float = 0.5
    escalate_on_silence: bool = True

    @classmethod
    def from_persona(cls, data: dict | None) -> "ProactiveConfig":
        d = data or {}
        cfg = cls()
        cfg.enabled = bool(d.get("enabled", False))
        # messages_per_day: "2-5" | [2,5] | 3
        mpd = d.get("messages_per_day", "2-5")
        try:
            if isinstance(mpd, str) and "-" in mpd:
                lo, hi = mpd.split("-", 1)
                cfg.min_per_day, cfg.max_per_day = int(lo), int(hi)
            elif isinstance(mpd, (list, tuple)) and len(mpd) == 2:
                cfg.min_per_day, cfg.max_per_day = int(mpd[0]), int(mpd[1])
            else:
                cfg.min_per_day = cfg.max_per_day = int(mpd)
        except (ValueError, TypeError):
            pass
        cfg.min_per_day = max(0, cfg.min_per_day)
        cfg.max_per_day = max(cfg.min_per_day, cfg.max_per_day)
        # active_hours: "08:00-23:00"
        hours = d.get("active_hours", "08:00-23:00")
        try:
            s, e = hours.split("-", 1)
            cfg.start = _parse_hhmm(s)
            cfg.end = _parse_hhmm(e)
        except (ValueError, AttributeError):
            pass
        try:
            cfg.clinginess = min(1.0, max(0.0, float(d.get("clinginess", 0.5))))
        except (ValueError, TypeError):
            pass
        cfg.escalate_on_silence = bool(d.get("escalate_on_silence", True))
        return cfg

    def messages_today(self, rng: random.Random | None = None) -> int:
        """How many check-ins to plan today; clinginess skews toward the max."""
        rng = rng or random
        if self.max_per_day == self.min_per_day:
            return self.min_per_day
        span = self.max_per_day - self.min_per_day
        skew = rng.random() ** (1.0 - 0.6 * self.clinginess)  # higher cling -> higher draw
        return self.min_per_day + round(skew * span)

    def in_active_hours(self, now: datetime) -> bool:
        t = now.time()
        if self.start <= self.end:
            return self.start <= t <= self.end
        return t >= self.start or t <= self.end  # window crossing midnight

    def followup_delay_hours(self, attempt: int, rng: random.Random | None = None) -> float:
        """2-4h, sooner when clingier and on the second follow-up."""
        rng = rng or random
        base = 2.0 + rng.random() * 2.0
        base *= 1.0 - 0.35 * self.clinginess
        if attempt >= 1:
            base *= 0.75
        return max(0.75, base)


def _parse_hhmm(s: str) -> dtime:
    h, m = s.strip().split(":")
    return dtime(int(h), int(m))


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hours_since(iso_ts: str | None) -> float | None:
    if not iso_ts:
        return None
    try:
        then = datetime.fromisoformat(iso_ts)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - then).total_seconds() / 3600.0
    except ValueError:
        return None


# Relationship milestones (feature: she notices "it's been a month" etc.
# without being asked). days_known milestones apply from any stage; the
# monthiversary-of-becoming-a-couple one only once close/girlfriend.
_MILESTONE_DAYS = (7, 30, 100, 365)


def _nth_monthiversary(start: date, n: int) -> date:
    """Calendar date of the n-th month anniversary of `start`, clamped to the
    last day of the target month (e.g. Jan 31 start -> Feb 28/29)."""
    total = start.month - 1 + n
    year = start.year + total // 12
    month = total % 12 + 1
    day = min(start.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


class ProactiveEngine:
    """Plans and sends proactive check-ins; owns the APScheduler instance."""

    def __init__(self, db, llm, memory, get_persona, settings,
                 now_fn=datetime.now, relationship=None, tools=None):
        self.db = db
        self.llm = llm
        self.memory = memory
        self.get_persona = get_persona  # callable -> active Persona
        self.settings = settings
        self.now = now_fn
        self.relationship = relationship  # RelationshipTracker | None
        self.tools = tools  # ToolRunner | None - same tools chat/WhatsApp use
        self.scheduler = AsyncIOScheduler()
        self._planned_today: list[str] = []

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self.scheduler.start()
        # Re-plan each day just after midnight, and plan the rest of today now.
        self.scheduler.add_job(self.plan_day, CronTrigger(hour=0, minute=5),
                               id="replan", replace_existing=True)
        self.plan_day()

    def stop(self) -> None:
        try:
            self.scheduler.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass

    def config(self) -> ProactiveConfig:
        """Persona proactive config, scaled by the current relationship stage.

        Clinginess is PRIMARILY driven by stage (see
        app.relationship.effective_clinginess) - the persona's own
        `clinginess` YAML value is a personality nudge around that stage
        baseline, not a ceiling that caps her at some fixed low number for
        the entire relationship. Message frequency still scales by a simple
        stage factor (messages_per_day * stage_factor)."""
        cfg = ProactiveConfig.from_persona(getattr(self.get_persona(), "proactive", {}))
        if not self.relationship:
            return cfg
        stage = self.relationship.stage
        base_clinginess = cfg.clinginess
        mpd_f = self.relationship.proactive_scale()
        if stage == "stranger":
            # A stranger doesn't text first - beyond one single hello.
            cfg.clinginess = 0.0
            if self.db.get_setting("proactive_stranger_hello_sent") == "1":
                cfg.enabled = False
            else:
                cfg.min_per_day = cfg.max_per_day = 1
            return cfg
        cfg.min_per_day = max(1, round(cfg.min_per_day * mpd_f)) if cfg.min_per_day else 0
        cfg.max_per_day = max(cfg.min_per_day, round(cfg.max_per_day * mpd_f))
        cfg.clinginess = self.relationship.clinginess_for(base_clinginess)
        return cfg

    def _stage(self) -> str:
        return self.relationship.stage if self.relationship else "girlfriend"

    # -- pause -------------------------------------------------------------
    def pause_for(self, hours: float) -> str | None:
        """Pause (hours > 0) or resume (hours <= 0). Returns paused_until ISO."""
        if hours <= 0:
            self.db.set_setting("proactive_paused_until", "")
            logger.info("proactive: resumed by user")
            return None
        until = datetime.now(timezone.utc) + timedelta(hours=hours)
        self.db.set_setting("proactive_paused_until", until.isoformat())
        logger.info("proactive: paused until %s", until.isoformat())
        return until.isoformat()

    def paused_until(self) -> str | None:
        # Indefinite off-switch from the in-chat `/proactive off` command
        # (app/commands.py). Every send path already funnels through here, so
        # honouring it in one place covers check-ins, silence-reacts and
        # milestones alike.
        if self.db.get_setting("proactive_disabled"):
            return "disabled"
        raw = self.db.get_setting("proactive_paused_until") or ""
        if not raw:
            return None
        if (_hours_since(raw) or 0) > 0:  # timestamp in the past -> not paused
            return None
        return raw

    # -- milestones ----------------------------------------------------------
    def _pending_milestone(self) -> str | None:
        """Milestone key for TODAY, if any and not already sent - else None.
        Checked once per plan_day() (daily at 00:05 + on startup)."""
        if not self.relationship:
            return None
        row = self.relationship.state()
        days = row["days_known"]
        stage = row["stage"]
        today = self.now().date()

        if days in _MILESTONE_DAYS:
            key = f"days_{days}"
            if self.db.get_setting(f"milestone_sent:{key}") != "1":
                return key

        if stage in ("close", "girlfriend"):
            entered_raw = self.db.get_setting(f"stage_entered_at:{stage}")
            if entered_raw:
                try:
                    entered = datetime.fromisoformat(entered_raw).date()
                except ValueError:
                    entered = None
                if entered:
                    for n in range(1, 25):  # up to 2 years of monthiversaries
                        anniv = _nth_monthiversary(entered, n)
                        if anniv > today:
                            break
                        if anniv == today:
                            key = f"{stage}_month_{n}"
                            if self.db.get_setting(f"milestone_sent:{key}") != "1":
                                return key
        return None

    @staticmethod
    def _milestone_label(key: str) -> str:
        if key.startswith("days_"):
            n = key.split("_", 1)[1]
            return f"it's been exactly {n} days since you two started talking"
        if "_month_" in key:
            n = int(key.rsplit("_month_", 1)[1])
            unit = "month" if n == 1 else "months"
            return f"it's been {n} {unit} since you two officially became a couple"
        return "today marks a small relationship milestone"

    # -- planning ----------------------------------------------------------
    def plan_day(self) -> list[datetime]:
        """Schedule today's remaining check-ins at random times in the window."""
        cfg = self.config()
        # Clear previously planned one-off jobs.
        for job in self.scheduler.get_jobs():
            if job.id.startswith("checkin_") or job.id.startswith("followup_"):
                job.remove()
        self._planned_today = []
        if not cfg.enabled:
            logger.info("proactive: disabled for persona - nothing planned")
            return []

        now = self.now()
        count = cfg.messages_today()
        start_dt = now.replace(hour=cfg.start.hour, minute=cfg.start.minute,
                               second=0, microsecond=0)
        end_dt = now.replace(hour=cfg.end.hour, minute=cfg.end.minute,
                             second=0, microsecond=0)
        window_start = max(now + timedelta(minutes=10), start_dt)
        if window_start >= end_dt:
            logger.info("proactive: active window already over today; planning none")
            return []

        span = (end_dt - window_start).total_seconds()
        times = sorted(
            window_start + timedelta(seconds=random.random() * span)
            for _ in range(count)
        )
        milestone_key = self._pending_milestone()
        for i, when in enumerate(times):
            intent = self._intent_for(when, i, len(times))
            mkey = None
            if milestone_key and i == 0:
                # A genuine occasion beats whatever the random draw picked
                # for the day's first slot.
                intent, mkey = "milestone", milestone_key
            self.scheduler.add_job(
                self.fire_checkin, DateTrigger(run_date=when),
                args=[intent, mkey], id=f"checkin_{i}_{when:%H%M%S}",
            )
            self._planned_today.append(f"{when:%H:%M} {intent}")
        logger.info("proactive: planned %d check-in(s): %s",
                    len(times), ", ".join(self._planned_today))
        return list(times)

    def _intent_for(self, when: datetime, i: int, total: int) -> str:
        stage = self._stage()
        if stage == "stranger":
            return "hello"
        if i == 0 and when.hour < 11:
            return "good_morning"
        if i == total - 1 and when.hour >= 21:
            return "goodnight"
        # Friction: while genuinely upset (relationship.py), the sweet/needy
        # intents don't fit - a "miss you 🥺" or a surprise song right after
        # a hurtful exchange would read as emotionally tone-deaf, not sweet.
        upset = bool(self.relationship and self.relationship.upset_state())
        # RARE unprompted song drop: warm stages only, library non-empty,
        # throttled by the eagerness dial.
        covers = getattr(self, "covers", None)
        if (not upset and covers and stage in ("close", "girlfriend") and covers.library()
                and random.random() < 0.12 * float(
                    (self.settings.behavior or {}).get("eagerness", 0.2)) * 5):
            return "song_drop"
        # "miss you" texts only once things are actually warm.
        pool = INTENTS if stage in ("close", "girlfriend") else EARLY_INTENTS
        if upset:
            pool = [p for p in pool if p != "miss_you"] or EARLY_INTENTS
        return random.choice(pool)

    # -- firing ------------------------------------------------------------
    async def fire_checkin(self, intent: str = "random_thought",
                           milestone_key: str | None = None) -> bool:
        """Send one proactive message unless a skip condition applies."""
        cfg = self.config()
        now = self.now()
        session_id = self.settings.wa_session_id

        if not cfg.enabled:
            logger.info("proactive: SKIP (%s) - disabled", intent)
            return False
        if self.paused_until():
            logger.info("proactive: SKIP (%s) - paused until %s", intent, self.paused_until())
            return False
        if not cfg.in_active_hours(now):
            logger.info("proactive: SKIP (%s) - outside active hours", intent)
            return False
        activity = self.db.get_last_activity(session_id)
        hours_quiet = _hours_since(activity["last_user_ts"])
        # If we're literally mid-conversation, don't butt in - except a
        # genuine milestone (like goodnight) is worth landing regardless.
        if hours_quiet is not None and hours_quiet < 0.5 and intent not in ("goodnight", "milestone"):
            logger.info("proactive: SKIP (%s) - talked %.1fh ago (mid-conversation)",
                        intent, hours_quiet)
            return False

        text = await self._generate(intent, now, activity, hours_quiet, cfg,
                                    milestone_key=milestone_key)
        if not text:
            logger.info("proactive: SKIP (%s) - generation failed", intent)
            return False

        await self._deliver(session_id, text, intent=intent)
        logger.info("proactive: FIRED %s -> %r", intent, text[:80])
        self.db.set_setting("proactive_last_sent", _utcnow_iso())
        self.db.set_setting("proactive_followups", "0")
        if intent == "hello":
            self.db.set_setting("proactive_stranger_hello_sent", "1")
            logger.info("proactive: stranger hello sent - going quiet until stage changes")
        if intent == "milestone" and milestone_key:
            self.db.set_setting(f"milestone_sent:{milestone_key}", "1")
            logger.info("proactive: milestone '%s' marked sent", milestone_key)

        if cfg.escalate_on_silence and cfg.clinginess > 0.6 and intent != "goodnight":
            self._schedule_followup(attempt=0, cfg=cfg)
        return True

    async def fire_followup(self, attempt: int) -> bool:
        """Escalating follow-up if they never replied; sulk after the 2nd."""
        cfg = self.config()
        session_id = self.settings.wa_session_id
        sent_ts = self.db.get_setting("proactive_last_sent")
        activity = self.db.get_last_activity(session_id)
        user_h = _hours_since(activity["last_user_ts"])
        sent_h = _hours_since(sent_ts)

        # They replied since our last proactive message -> stand down.
        if user_h is not None and sent_h is not None and user_h < sent_h:
            logger.info("proactive: follow-up %d SKIP - they replied", attempt + 1)
            return False
        if self.paused_until() or not cfg.in_active_hours(self.now()):
            logger.info("proactive: follow-up %d SKIP - paused/outside hours", attempt + 1)
            return False

        if attempt >= 2:
            # Max follow-ups reached: she sulks, quietly. Stored as a memory she
            # can bring up next time - no further messages.
            await self.memory.add_fact(
                f"User didn't reply to her messages for hours on "
                f"{self.now():%B %d}; she felt ignored and got a bit sulky about it.",
                "emotion",
            )
            logger.info("proactive: follow-up cap reached - sulk memory stored")
            return False

        text = await self._generate(
            "silence_react", self.now(),
            activity, sent_h, cfg, followup_attempt=attempt + 1,
        )
        if not text:
            return False
        await self._deliver(session_id, text, intent="silence_react")
        logger.info("proactive: FIRED follow-up %d -> %r", attempt + 1, text[:80])
        self.db.set_setting("proactive_followups", str(attempt + 1))
        self._schedule_followup(attempt=attempt + 1, cfg=cfg)
        return True

    def _schedule_followup(self, attempt: int, cfg: ProactiveConfig) -> None:
        delay_h = cfg.followup_delay_hours(attempt)
        when = self.now() + timedelta(hours=delay_h)
        self.scheduler.add_job(
            self.fire_followup, DateTrigger(run_date=when),
            args=[attempt], id=f"followup_{attempt}_{when:%H%M%S}",
            replace_existing=True,
        )
        logger.info("proactive: follow-up %d scheduled in %.1fh", attempt + 1, delay_h)

    # -- generation / delivery ---------------------------------------------
    async def _generate(self, intent, now, activity, hours_quiet, cfg,
                        followup_attempt: int = 0, milestone_key: str | None = None) -> str | None:
        persona = self.get_persona()
        query = activity["last_user_text"] or "how they are doing, their plans and feelings"
        try:
            memories = await self.memory.retrieve_memories(query, k=4)
        except Exception:  # noqa: BLE001
            memories = []

        h = now.hour
        tod = ("the middle of the night" if h < 4 else
               "early morning" if h < 7 else "morning" if h < 12 else
               "afternoon" if h < 17 else "evening" if h < 21 else "night")
        quiet_line = (
            f"You last heard from them about {hours_quiet:.0f} hours ago."
            if hours_quiet is not None and hours_quiet >= 1
            else "You spoke fairly recently."
        )
        last_topic = (
            f'The last thing they said was: "{activity["last_user_text"][:200]}"'
            if activity["last_user_text"] else "You have no recent messages from them."
        )
        cling_style = (
            "Keep it light, casual and unbothered." if cfg.clinginess < 0.34 else
            "Warm and a little affectionate; one emoji is fine." if cfg.clinginess < 0.67 else
            "Very affectionate and a bit needy: pet names, emoji, you clearly miss them."
        )
        needy = ""
        if followup_attempt == 1:
            needy = "This is your FIRST follow-up to being left on read - gently nudge them."
        elif followup_attempt >= 2:
            needy = ("This is your SECOND follow-up with still no reply - noticeably "
                     "needier/pouty, but still loving, never angry.")

        weather_line = ""
        if intent == "good_morning" and self.tools is not None:
            try:
                res = await self.tools.call("weather", {})
                if res.get("ok"):
                    weather_line = f" You just checked: {res['result']}. Weave it in naturally if it fits (e.g. mention the weather), but don't force it."
            except Exception:  # noqa: BLE001 - a morning text without weather is fine too
                pass

        mood_line = ""
        if intent in ("share_feeling", "silence_react", "random_thought"):
            try:
                today_entries = self.db.mood_entries_for_day(now.date().isoformat())
            except Exception:  # noqa: BLE001
                today_entries = []
            if today_entries:
                worst = max(today_entries, key=lambda e: e["intensity"])
                mood_line = (
                    f" [You already quietly know today was a '{worst['mood_label']}' kind of "
                    f"day for them ({worst['why']}). Do NOT ask a generic 'how are you feeling' "
                    f"- ask or react to the SPECIFIC thing instead, like you already noticed.]"
                )

        if intent == "milestone" and milestone_key:
            label = self._milestone_label(milestone_key)
            directive = (
                f"[It is {now:%A} {now:%I:%M %p} ({tod}). Today is meaningful: {label}.] "
                "Send ONE warm, genuine text marking this - specific to what the occasion "
                "actually is, never generic. No stats-speak, no cliché 'happy "
                "monthiversary!' greeting-card energy - just how you'd actually text "
                f"someone you're really into about it. {cling_style} Short, in your voice. "
                "Do not mention this note."
            )
        else:
            directive = (
                f"[It is {now:%A} {now:%I:%M %p} ({tod}). {quiet_line} {last_topic}]{weather_line}{mood_line} "
                f"{_INTENT_DIRECTIVES.get(intent, _INTENT_DIRECTIVES['random_thought'])} "
                f"{needy} {cling_style} "
                "Write ONE short text message (1-2 sentences), specific to them when your "
                "memories allow - never generic filler like 'how are you'. Do not invent "
                "events about them that aren't in your memories. Do not mention this note."
            )
        stage_note = self.relationship.addendum() if self.relationship else None
        system = build_system_prompt(
            persona, memories, mode="chat",
            current_time=f"{now:%A, %I:%M %p} ({tod})",
            extra_notes=stage_note,
            stage=self.relationship.stage if self.relationship else None,
        )
        try:
            raw = await self.llm.chat(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": directive}],
                options={"temperature": 0.9},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("proactive: generation failed: %s", e)
            return None
        text = strip_tags(raw).strip().strip('"')
        return text or None

    async def _deliver(self, session_id: str, text: str, intent: str = "") -> None:
        """Store in shared history + push to WhatsApp via the bridge if it's up.

        Mood-matched intents (good morning/night, miss you) may also carry a
        sticker, with the same stage-scaled probability and never-two-in-a-row
        rule as replies.
        """
        self.db.ensure_session(session_id)
        self.db.add_message(session_id, "assistant", text, source="whatsapp")

        # Song drops carry the actual recording as a second message.
        covers = getattr(self, "covers", None)
        if intent == "song_drop" and covers:
            song = covers.find()
            if song:
                self.db.add_message(session_id, "assistant", "",
                                    audio_url=song["url"])
                try:
                    from app.covers import LIBRARY
                    async with httpx.AsyncClient(timeout=30.0) as c:
                        await c.post(f"{self.settings.wa_bridge_url}/send-voice",
                                     json={"wav_path": str(LIBRARY / song["file"])})
                except Exception:  # noqa: BLE001
                    pass

        sticker = None
        if self.relationship:
            prob = self.relationship.sticker_probability()
            last_had = self.db.get_setting("last_reply_had_sticker") == "1"
            if prob > 0 and not last_had and random.random() < prob:
                from app.stickers import pick_sticker

                sticker = pick_sticker(intent)
        self.db.set_setting("last_reply_had_sticker", "1" if sticker else "0")
        if sticker:
            self.db.add_message(session_id, "assistant", "", sticker_url=sticker[1], source="whatsapp")

        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.post(f"{self.settings.wa_bridge_url}/send-text",
                                 json={"text": text})
                r.raise_for_status()
                if sticker:
                    await c.post(f"{self.settings.wa_bridge_url}/send-sticker",
                                 json={"path": str(sticker[0])})
        except Exception as e:  # noqa: BLE001
            logger.info("proactive: bridge unreachable (%s) - stored for web only", e)

    # -- status --------------------------------------------------------------
    def status(self) -> dict:
        cfg = self.config()
        jobs = [
            {"id": j.id, "next": j.next_run_time.isoformat() if j.next_run_time else None}
            for j in self.scheduler.get_jobs()
            if j.id.startswith(("checkin_", "followup_"))
        ]
        return {
            "enabled": cfg.enabled,
            "paused_until": self.paused_until(),
            "clinginess": cfg.clinginess,
            "active_hours": f"{cfg.start:%H:%M}-{cfg.end:%H:%M}",
            "planned": jobs,
        }
