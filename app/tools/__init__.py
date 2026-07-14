"""Tool-calling framework: she can take real actions via Ollama NATIVE
function calling (the model itself picks the tool + extracts args in one
round trip - no regex tool-matching, no separate arg-extraction call).

Each tool is one file in this package exporting a single `TOOL = Tool(...)`.
Flow: model requests tool -> execute() -> result fed back -> she phrases the
result in-character (main.py). Raw tool JSON never reaches the user; every
call is logged (tool_call_log table).

Model choice: llama3.2:3b was empirically tested against qwen2.5:3b-instruct
for tool SELECTION specifically (tests/tool_calling_eval.py) - llama3.2:3b
scored 100% with zero false-positive tool calls on mentions ("i'm hungry"),
while qwen2.5:3b-instruct deterministically hallucinated a weather() call on
a pure mention ("it's so hot today i'm melting" -> weather(city="AnyCity")).
That's disqualifying regardless of qwen's edge on raw JSON formatting
elsewhere, so llama3.2:3b (the main chat model) also handles tool selection.
See README "Tool-calling model selection" for the full test transcript.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("companion.tools")


@dataclass
class ToolContext:
    """Everything a tool's execute() might need. Built once in main.py."""
    db: Any
    llm: Any
    settings: Any
    covers: Any = None       # CoverPipeline, set after construction (sing tool)
    relationship: Any = None  # RelationshipTracker (stage-gated tools)
    deliver: Any = None      # async fn(session_id, text) -> None (delivery routing)


@dataclass
class Tool:
    name: str
    description: str
    # JSON-schema "properties" dict, Ollama/OpenAI function-calling style.
    # Keep flat + 1-2 args max - the local 3B model is unreliable with nesting.
    parameters: Dict[str, Dict[str, Any]]
    required: List[str]
    execute: Callable[[Dict[str, Any], ToolContext], Awaitable[Dict[str, Any]]]
    enabled: Callable[[Any], bool] = field(default=lambda settings: True)

    def to_ollama_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


def _load_registry() -> Dict[str, Tool]:
    """Import each tool module and collect its TOOL export."""
    from app.tools import events, reminder, sing, weather, zomato

    modules = [weather, reminder, events, sing, zomato]
    return {m.TOOL.name: m.TOOL for m in modules}


_REGISTRY: Dict[str, Tool] = _load_registry()


def all_tools(settings) -> List[Tool]:
    """Tools enabled under the current config (e.g. zomato_suggest can be off)."""
    return [t for t in _REGISTRY.values() if t.enabled(settings)]


def get_tool(name: str) -> Optional[Tool]:
    return _REGISTRY.get(name)


def _coerce(value: Any, schema: Dict[str, Any]) -> Any:
    """Small models sometimes emit '250' for a number param - tolerate it."""
    want = schema.get("type")
    if want == "number" and not isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if want == "integer" and not isinstance(value, int):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return value
    return value


class ToolRunner:
    """Executes a tool call and logs it. Tool SELECTION happens in the router
    (native function calling); this class just runs whatever was decided -
    used identically by chat, WhatsApp, and the proactive scheduler."""

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        # back-compat attributes some call sites set directly after construction
        self.covers = None

    @property
    def db(self):
        return self.ctx.db

    def _sync_ctx(self) -> None:
        # main.py sets state.tools.covers = state.covers after construction;
        # keep ctx in sync so tool execute() functions see it via ctx.covers.
        self.ctx.covers = self.covers

    async def call(self, name: str, args: Dict[str, Any] | None) -> Dict[str, Any]:
        """Execute `name` with `args` (already extracted by native function
        calling). Returns {"ok": bool, "result": <factual text for the persona
        wrapper>, ...extra machine-readable fields}."""
        self._sync_ctx()
        tool = get_tool(name)
        if tool is None:
            logger.warning("unknown tool requested: %r", name)
            return {"ok": False, "result": f"unknown tool {name}"}

        args = dict(args or {})
        for key, schema in tool.parameters.items():
            if key in args and args[key] is not None:
                args[key] = _coerce(args[key], schema)

        t0 = time.perf_counter()
        try:
            result = await tool.execute(args, self.ctx)
        except Exception as e:  # noqa: BLE001 - tools must never crash the turn
            logger.exception("tool %s failed", name)
            result = {"ok": False, "result": f"{name} failed: {e}"}
        ms = (time.perf_counter() - t0) * 1000
        self._log_call(name, args, result, ms)
        return result

    def _log_call(self, name: str, args: dict, result: dict, ms: float) -> None:
        logger.info("tool_call name=%s args=%s ok=%s ms=%.0f",
                   name, json.dumps(args, default=str), result.get("ok"), ms)
        try:
            self.ctx.db.log_tool_call(name, json.dumps(args, default=str),
                                      bool(result.get("ok")),
                                      str(result.get("result", ""))[:500])
        except Exception:  # noqa: BLE001 - logging must never break a tool call
            pass


__all__ = ["Tool", "ToolContext", "ToolRunner", "all_tools", "get_tool"]
