"""Three-layer router: deterministic rules -> native function calling -> dispatch.

CRITICAL RULE - MENTION vs REQUEST: a tool/DEEP route may only fire on a
REQUEST (an imperative or direct ask aimed at her). A MENTION ("i'm hungry",
"it's so hot today") ALWAYS routes to CHAT - she responds like a person, never
like an assistant. She may later OFFER an action (offer throttling in
behavior.py); the flow proceeds only if the user accepts.

Layer 1 (deterministic, no model call): the is_request() gate + DEEP
heuristics (code/math/length/verbs). This is the safety-critical layer and
never touches a model - mentions can never reach a tool no matter how the
model behaves.

Layer 2 (Ollama NATIVE function calling): for requests that passed layer 1
without an obvious DEEP/chat verdict, ONE Ollama chat call with `tools=` lets
the model pick a tool AND extract its arguments in the same round trip - no
separate arg-extraction call, no regex tool-name matching. If the model
doesn't call anything, that's CHAT.

Model choice: llama3.2:3b, empirically tested against qwen2.5:3b-instruct for
tool SELECTION (tests/tool_calling_eval.py) - see app/tools/__init__.py
docstring for the full result. llama3.2:3b scored 100% with zero
mention->tool false positives; qwen2.5:3b-instruct deterministically
hallucinated a tool call on a mention, which is disqualifying here.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from app.tools import all_tools

logger = logging.getLogger("companion.router")


@dataclass
class Route:
    kind: str            # chat | tool | deep
    tool: str | None = None
    args: dict = field(default_factory=dict)
    layer: str = "rules"  # rules | tools
    reason: str = ""


# --- REQUEST detection ------------------------------------------------------
# Direct-ask markers aimed at her ("can you...", "please...", imperative verbs).
_ASK_PREFIX = re.compile(
    r"^(please\s+|can you|could you|will you|would you|pls\s+|hey[, ]+)", re.I)
_IMPERATIVE_VERBS = re.compile(
    r"^(remind|check|tell|show|set|add|log|order|book|get|find|look up|search|"
    r"send|text|ping|message|list|mark|calculate|explain|research|compare|"
    r"summari[sz]e|translate|help me|sing|play|suggest|recommend)\b", re.I)
# A short throwaway interjection before the real ask ("Nice, what's the
# weather like") shouldn't defeat the anchored checks below - these are safe
# to strip unconditionally since what follows still has to pass the strict
# question-word anchor.
_LEADING_FILLER = re.compile(
    r"^(nice|cool|ok|okay|alright|sure|great|omg|lol|haha|hey|so|well|"
    r"anyway)[,!.]?\s+", re.I)
# Temporal adverbs ("tomorrow will it rain") are riskier to strip
# unconditionally - "tomorrow"/"today"/"tonight" routinely lead plain
# STATEMENTS too ("tomorrow is my exam", "today was rough"), which must stay
# mentions (regression-tested in routing_eval.py). Only strip one when it's
# immediately followed by an auxiliary+pronoun that looks like a real
# question ("tomorrow WILL IT rain", not "tomorrow IS MY exam").
_LEADING_TEMPORAL_Q = re.compile(
    r"^(today|tomorrow|tonight|later|now)[,!.]?\s+"
    r"(?=(will|is|are|does|do|did|should)\s+(it|i|you|we|they|this|that|there)\b)",
    re.I)


def _strip_leading_filler(t: str) -> str:
    for _ in range(2):
        stripped = _LEADING_FILLER.sub("", t, count=1)
        stripped = _LEADING_TEMPORAL_Q.sub("", stripped, count=1)
        if stripped == t:
            break
        t = stripped
    return t


def is_request(message: str) -> bool:
    """True only for imperatives/direct asks - the gate in front of every
    tool/DEEP route. Questions to HER as a person ("are you hungry?") and
    statements about my life ("i'm hungry") are NOT requests."""
    t = _strip_leading_filler(message.strip())
    if _ASK_PREFIX.match(t):
        return True
    if _IMPERATIVE_VERBS.match(t):
        return True
    # Interrogative information asks ("what's the weather...", "will it rain
    # tomorrow", "why does X... explain"): question-word-initial phrasing.
    # NOTE: do NOT require a trailing '?' - real chat routinely omits it
    # ("whats the weather like today", "will it rain tomorrow") and requiring
    # one silently demoted these to mentions, so they never even reached
    # layer 2's tool-calling model. Genuine mentions phrased as questions
    # ("what am i gonna do about this mess") still safely fall to chat at
    # layer 2 (no matching tool) or via _ABOUT_HER/_EXPERIENCE_Q below.
    if re.match(r"^(what|how|when|where|who|why|which|will|is|are|does|do|did|"
                r"should)('s|s)?\b", t, re.I):
        return True
    return False


# Questions about HER (her day, her feelings, her activities) are conversation,
# not information requests - even though they're phrased interrogatively.
# NOTE: this must be a curated phrase list, not a bare \b(you|your)\b match -
# that broad version silently ate every "can/could/would you <verb>" request
# ("can you check the weather?", "can you sing me something") because they
# also contain the word "you", routing them to chat before layer 2 ever saw
# them. Regression cases for this live in tests/routing_eval.py.
_ABOUT_HER = re.compile(
    r"\b(are you|do you (feel|miss|love|like|think)|did you (sleep|eat|have|do)|"
    r"what are you (doing|up to)|tell me (about )?your|you look(ed)?\b|"
    r"how('s| is) (your|ur) (day|night|morning|week))", re.I)
_EXPERIENCE_Q = re.compile(r"^how (was|is|are|were)\b", re.I)


# --- DEEP heuristics ----------------------------------------------------------
_CODE_BLOCK = re.compile(r"```|(\bdef |\bclass |\bfunction\b|\bimport )")
_MATH_EXPR = re.compile(r"\d+\s*[-+*/^%]\s*\d+|\b(sqrt|integral|derivative|equation)\b", re.I)

_TOOL_SYSTEM = ("You are a helpful assistant that can call tools when the user "
               "directly asks for an action. If the message is just conversation, "
               "casual talk, or a mention of a feeling, do NOT call any tool.")


class Router:
    def __init__(self, llm, settings, brain=None):
        self.llm = llm
        self.settings = settings
        self.brain = brain  # for degrade-mode threshold shifting
        cfg = settings.router or {}
        self.deep_verbs = [v.lower() for v in cfg.get("deep_verbs", [])]
        self.length_threshold = int(cfg.get("length_threshold", 350))
        self.db = None  # set by main for stats

    # -- stats ----------------------------------------------------------------
    def _bump(self, key: str) -> None:
        if not self.db:
            return
        raw = self.db.get_setting("routing_stats") or "{}"
        try:
            stats = json.loads(raw)
        except json.JSONDecodeError:
            stats = {}
        stats[key] = stats.get(key, 0) + 1
        self.db.set_setting("routing_stats", json.dumps(stats))

    def stats(self) -> dict:
        if not self.db:
            return {}
        try:
            return json.loads(self.db.get_setting("routing_stats") or "{}")
        except json.JSONDecodeError:
            return {}

    # -- layer 1: deterministic ------------------------------------------------
    def pre_route(self, message: str) -> Route | None:
        t = message.strip()
        request = is_request(t)

        # Self-logging statements ("i spent 250 on lunch") are implicit tool
        # requests even without an imperative - but only the concrete form;
        # "i spent way too much lol" stays a mention (falls through to chat).
        if re.match(r"^i spent \d+(\.\d+)?\b", t, re.I):
            return None  # let layer 2 extract the amount/note via native tool-calling

        # DEEP signals. Degrade mode raises the bar (casual stays local).
        degraded = bool(self.brain and self.brain.should_degrade())
        length_threshold = self.length_threshold * (2 if degraded else 1)

        if _CODE_BLOCK.search(t):
            return Route("deep", reason="code")
        if _MATH_EXPR.search(t) and request:
            return Route("deep", reason="math")
        if len(t) > length_threshold:
            return Route("deep", reason="length")
        if t.count("?") >= 2 and len(t) > 80:
            return Route("deep", reason="multi-question")
        if request and not degraded:
            lowered = t.lower()
            for verb in self.deep_verbs:
                if re.search(rf"\b{re.escape(verb)}\b", lowered):
                    return Route("deep", reason=f"verb:{verb}")

        # Obvious small talk: short, no request → chat without any model call.
        # This is the safety-critical short-circuit - mentions never reach
        # layer 2's tool-calling model, regardless of how reliable it is.
        if not request and len(t) < 120:
            return Route("chat", reason="mention-or-smalltalk")
        if not request:
            return Route("chat", reason="not-a-request")
        # Requests aimed at HER as a person ("what are you doing rn?", "how
        # was pilates?") are conversation, not tool candidates.
        if _ABOUT_HER.search(t) or _EXPERIENCE_Q.match(t):
            return Route("chat", reason="question-about-her")
        return None  # a real request, not obviously deep/chat → layer 2

    # -- layer 2: Ollama native function calling -----------------------------------
    async def classify(self, message: str) -> Route:
        """Offer every enabled tool via Ollama's native function-calling. The
        model both picks the tool and extracts its arguments in one call -
        no regex tool-matching, no separate JSON-extraction round trip."""
        tools = all_tools(self.settings)
        schemas = [t.to_ollama_schema() for t in tools]
        try:
            msg = await self.llm.chat_with_tools(
                messages=[{"role": "system", "content": _TOOL_SYSTEM},
                          {"role": "user", "content": message}],
                tools=schemas,
                options={"temperature": 0.0},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("tool-calling classify failed (%s) - defaulting CHAT", e)
            return Route("chat", layer="tools", reason="error")

        calls = msg.get("tool_calls") or []
        if not calls:
            return Route("chat", layer="tools", reason="no-tool-call")

        fn = calls[0].get("function", {})
        name = fn.get("name")
        args = fn.get("arguments") or {}
        if isinstance(args, str):  # some models emit a JSON string, not a dict
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name not in {t.name for t in tools}:
            logger.warning("model called unknown/disabled tool %r - treating as chat", name)
            return Route("chat", layer="tools", reason="unknown-tool")
        return Route("tool", tool=name, args=args, layer="tools", reason="native-fn-call")

    # -- entry ---------------------------------------------------------------
    async def route(self, message: str) -> Route:
        r = self.pre_route(message)
        if r is None:
            r = await self.classify(message)
        self._bump(f"{r.kind}:{r.tool or r.reason.split(':')[0]}")
        logger.info("route: %s%s [%s: %s] %r",
                    r.kind, f"/{r.tool}" if r.tool else "", r.layer, r.reason,
                    message[:60])
        return r
