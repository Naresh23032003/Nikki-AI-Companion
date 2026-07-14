"""Cloud "big brain": OpenAI-compatible providers with fallback + budgets.

Chain (config `brain.providers`, keys in .env): Groq -> Gemini -> Cerebras ->
local-only. Reads Groq rate-limit headers, persists daily usage (requests +
tokens), and appends every outbound payload to a local JSONL audit log so you
can see exactly what left the laptop.

The big brain NEVER speaks to the user: it returns terse structured facts that
the local persona model rephrases in-character (see the wrapper in main.py).
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone

import httpx

from app.config import ROOT

logger = logging.getLogger("companion.brain")

BIG_BRAIN_SYSTEM = """\
You are a silent reasoning engine backing a companion app. Answer the question
directly and factually.
- Terse bullet points or 2-4 compact sentences. Max ~200 words.
- NO personality, NO greetings, NO "great question", NO markdown headers.
- If you are uncertain or the info may be stale, say so explicitly in one
  clause. Never fabricate specifics (prices, dates, news)."""


class BrainUnavailable(Exception):
    def __init__(self, reason: str, retry_after: float | None = None):
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after  # seconds, if a provider told us


class CloudBrain:
    def __init__(self, db, settings):
        self.db = db
        cfg = settings.brain or {}
        self.enabled = bool(cfg.get("cloud_enabled", True))
        self.providers = cfg.get("providers", [])
        self.req_budget = int(cfg.get("requests_per_day", 950))
        self.tok_budget = int(cfg.get("tokens_per_day", 190000))
        self.degrade_at = float(cfg.get("degrade_at", 0.8))
        self.audit_path = ROOT / cfg.get("audit_log", "cloud_audit.jsonl")
        self._last_headers: dict = {}

    # -- usage persistence (app_settings, keyed by day) ----------------------
    def _ukey(self) -> str:
        return f"brain_usage:{date.today().isoformat()}"

    def usage_today(self) -> dict:
        raw = self.db.get_setting(self._ukey())
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        return {"requests": 0, "tokens": 0, "by_provider": {}}

    def _record_usage(self, provider: str, tokens: int) -> None:
        u = self.usage_today()
        u["requests"] += 1
        u["tokens"] += max(0, tokens)
        per = u["by_provider"].setdefault(provider, {"requests": 0, "tokens": 0})
        per["requests"] += 1
        per["tokens"] += max(0, tokens)
        self.db.set_setting(self._ukey(), json.dumps(u))

    def budget_fraction(self) -> float:
        u = self.usage_today()
        return max(u["requests"] / self.req_budget if self.req_budget else 0,
                   u["tokens"] / self.tok_budget if self.tok_budget else 0)

    def should_degrade(self) -> bool:
        """Above N% of daily budget: keep casual stuff local."""
        return self.budget_fraction() >= self.degrade_at

    def over_budget(self) -> bool:
        return self.budget_fraction() >= 1.0

    # -- audit ----------------------------------------------------------------
    def _audit(self, provider: str, messages: list, status: str) -> None:
        try:
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "provider": provider,
                    "status": status,
                    "messages": messages,  # exactly what left the laptop
                }) + "\n")
        except OSError as e:
            logger.warning("audit log write failed: %s", e)

    # -- main entry -------------------------------------------------------------
    async def ask(self, question: str, context: str | None = None,
                  max_tokens: int = 700) -> tuple[str, str]:
        """Ask the big brain. Returns (answer, provider_name).

        Raises BrainUnavailable when every cloud provider fails (caller then
        falls back to local-only or queues a deferred task).
        """
        if not self.enabled:
            raise BrainUnavailable("cloud disabled")
        if self.over_budget():
            raise BrainUnavailable("daily budget exhausted")

        user_content = question if not context else f"{context}\n\nQuestion: {question}"
        messages = [
            {"role": "system", "content": BIG_BRAIN_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        import os
        retry_after: float | None = None
        for p in self.providers:
            if not p.get("enabled"):
                continue
            key = os.environ.get(p.get("key_env", ""), "")
            if not key:
                logger.info("brain: provider %s skipped (no key)", p["name"])
                continue
            try:
                text, tokens, headers = await self._call(p, key, messages, max_tokens)
                self._last_headers = {p["name"]: headers}
                self._record_usage(p["name"], tokens)
                self._audit(p["name"], messages, "ok")
                logger.info("brain: %s answered (%d tok)", p["name"], tokens)
                return text, p["name"]
            except httpx.HTTPStatusError as e:
                self._audit(p["name"], messages, f"http {e.response.status_code}")
                if e.response.status_code == 429:
                    ra = e.response.headers.get("retry-after")
                    try:
                        retry_after = float(ra) if ra else retry_after
                    except ValueError:
                        pass
                    logger.warning("brain: %s rate-limited (retry-after=%s)",
                                   p["name"], ra)
                else:
                    logger.warning("brain: %s HTTP %s", p["name"],
                                   e.response.status_code)
            except Exception as e:  # noqa: BLE001 - walk the chain silently
                self._audit(p["name"], messages, f"error {type(e).__name__}")
                logger.warning("brain: %s failed: %s", p["name"], e)

        raise BrainUnavailable("all cloud providers failed", retry_after=retry_after)

    async def _call(self, provider: dict, key: str, messages: list,
                    max_tokens: int) -> tuple[str, int, dict]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{provider['base_url'].rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": provider["model"], "messages": messages,
                      "max_tokens": max_tokens, "temperature": 0.3},
            )
            r.raise_for_status()
            data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        tokens = int((data.get("usage") or {}).get("total_tokens", 0))
        # Groq-style rate-limit headers (persisted for the Brain status page).
        headers = {k: v for k, v in r.headers.items()
                   if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"}
        return text, tokens, headers

    # -- status for the settings page ------------------------------------------
    def status(self) -> dict:
        import os
        u = self.usage_today()
        return {
            "cloud_enabled": self.enabled,
            "usage": {"requests": u["requests"], "requests_budget": self.req_budget,
                      "tokens": u["tokens"], "tokens_budget": self.tok_budget,
                      "fraction": round(self.budget_fraction(), 3),
                      "degraded": self.should_degrade()},
            "by_provider": u.get("by_provider", {}),
            "providers": [
                {"name": p["name"], "enabled": bool(p.get("enabled")),
                 "model": p.get("model"),
                 "has_key": bool(os.environ.get(p.get("key_env", ""), ""))}
                for p in self.providers
            ],
            "rate_limit_headers": self._last_headers,
        }
