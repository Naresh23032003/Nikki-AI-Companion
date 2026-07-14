"""Ollama client: streaming chat, one-shot chat, and embeddings.

- stream_chat: token-by-token generation for the /chat endpoint (/api/chat).
- chat:        one-shot completion, optionally JSON-constrained - used by the
               memory fact-extractor.
- embed:       vector for a piece of text via the embedding model
               (/api/embeddings), used by the long-term memory store.

The long-term memory retrieval hook now lives in app/memory.py
(MemoryStore.retrieve_memories); see app/main.py for where it's wired in.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        embed_model: str = "nomic-embed-text",
        options: Dict[str, Any] | None = None,
        timeout: float = 120.0,
        keep_alive: str = "2h",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.options = options or {}
        # Ollama evicts models after 5 minutes by default - the first message
        # after any idle gap then pays a full cold reload (measured >1s for
        # nomic-embed alone, dwarfing the actual work). Pin both chat and
        # embed models resident instead; they fit the VRAM budget together.
        self.keep_alive = keep_alive
        # httpx drops idle keep-alive connections after 5s by default, so any
        # pause in conversation forced a fresh TCP connect on the next message
        # - measured at ~900ms on Windows loopback, most of the "SLOW"
        # retrieve_memories time. Keep pooled connections for an hour instead.
        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=10,
                                keepalive_expiry=3600.0),
        )

    async def stream_chat(
        self, messages: List[Dict[str, str]]
    ) -> AsyncIterator[str]:
        """Stream a chat completion token-by-token.

        `messages` is a list of {"role": ..., "content": ...} dicts, including
        the leading system message. Yields content chunks as they arrive.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": self.options,
            "keep_alive": self.keep_alive,
        }

        url = f"{self.base_url}/api/chat"
        async with self._client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # Ollama streams newline-delimited JSON; skip anything odd.
                    continue

                # Each chunk looks like:
                # {"message": {"role": "assistant", "content": "..."}, "done": false}
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk

                if data.get("done"):
                    break

    async def chat(
        self,
        messages: List[Dict[str, str]],
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        keep_alive: Optional[str | int] = None,
    ) -> str:
        """One-shot (non-streaming) completion; returns the full text.

        Pass format="json" to constrain the model to emit valid JSON, and
        `model` to override the default (e.g. the dedicated extraction model).
        `timeout` overrides the client default (120s) - offline batch jobs
        like the nightly journal push a whole day's transcript through a
        bigger model (with a model swap first) and can legitimately run
        longer than any live reply ever should.
        `keep_alive` overrides the client default residency - pass 0 for a
        one-off model (nightly journal's 8B) so it unloads immediately
        instead of squatting in VRAM for hours and starving TTS/RVC.
        """
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": options if options is not None else self.options,
            "keep_alive": keep_alive if keep_alive is not None else self.keep_alive,
        }
        if format:
            payload["format"] = format

        # NB: request timeout=None means "no timeout" in httpx - the client
        # default only applies when the argument is left as the sentinel.
        resp = await self._client.post(
            f"{self.base_url}/api/chat", json=payload,
            timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    async def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """One-shot completion using Ollama's NATIVE function calling.

        `tools` is a list of OpenAI-style tool schemas:
          {"type": "function", "function": {"name", "description", "parameters"}}

        Returns the raw message dict, e.g.
          {"role": "assistant", "content": "...", "tool_calls": [
              {"function": {"name": "weather", "arguments": {"city": "Chennai"}}}
          ]}
        `tool_calls` is absent/empty when the model chose not to call anything.
        """
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": options if options is not None else self.options,
            "keep_alive": self.keep_alive,
        }
        resp = await self._client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}) or {}

    async def embed(self, text: str) -> List[float]:
        """Return the embedding vector for `text` using the embedding model."""
        resp = await self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.embed_model, "prompt": text,
                  "keep_alive": self.keep_alive},
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding")
        if not embedding:
            raise ValueError(f"Ollama returned no embedding for text: {text!r}")
        return embedding

    async def close(self) -> None:
        await self._client.aclose()
