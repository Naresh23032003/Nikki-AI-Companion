"""Fake Ollama HTTP server for local demos and offline testing.

Emulates the subset of Ollama's HTTP API used by the companion app:
  - GET  /api/tags        -> List loaded/available models
  - POST /api/pull        -> Mock model pull status
  - POST /api/chat        -> Chat completions (stream: true newline-delimited JSON,
                             stream: false text, format: "json" structured memory extraction)
  - POST /api/embeddings  -> Deterministic 768-dimensional float embedding derived from prompt hash

Usage:
  python tools/fake_ollama.py --port 11434
  python tools/fake_ollama.py --host 0.0.0.0 --port 11434
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import random
import sys
from typing import Any, Dict, List

CANNED_REPLIES = [
    "heyy! i was just thinking about that. how's everything going with you?",
    "haha no way! that's honestly so cool. tell me more!",
    "aww, thank you! i'm always right here if you need anything 💕",
    "hmm, let me think about that for a second... okay yeah, i totally agree with you!",
    "okay but hear me out: that actually sounds like so much fun!",
]


def generate_embedding(text: str, dim: int = 768) -> List[float]:
    """Generate a deterministic float vector in [-1, 1] derived from text hash."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    # Generate floats rounded to 6 decimal places in [-1.0, 1.0]
    return [round(rng.uniform(-1.0, 1.0), 6) for _ in range(dim)]


def pick_canned_reply(messages: List[Dict[str, Any]]) -> str:
    """Pick a deterministic in-character reply based on the last user message."""
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                last_user_msg = content
            elif isinstance(content, list):
                last_user_msg = " ".join(str(item) for item in content)
            break
    if not last_user_msg:
        last_user_msg = "hello"
    idx = int.from_bytes(hashlib.md5(last_user_msg.encode("utf-8")).digest()[:4], "big") % len(CANNED_REPLIES)
    return CANNED_REPLIES[idx]


class FakeOllamaHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy standard HTTP logs during tests unless desired
        pass

    def _send_json(self, status: int, data: Any) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/api/tags":
            models = [
                {"name": "llama3.2:3b", "model": "llama3.2:3b", "details": {"family": "llama"}},
                {"name": "nomic-embed-text", "model": "nomic-embed-text", "details": {"family": "nomic"}},
                {"name": "llama3.1:8b", "model": "llama3.1:8b", "details": {"family": "llama"}},
                {"name": "moondream", "model": "moondream", "details": {"family": "moondream"}},
            ]
            self._send_json(200, {"models": models})
        elif path == "/":
            self._send_json(200, {"status": "Ollama is running (fake)"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        if path == "/api/pull":
            model_name = payload.get("name", "unknown")
            self._send_json(200, {"status": "success", "model": model_name})
        elif path == "/api/embeddings":
            prompt = payload.get("prompt", "")
            vector = generate_embedding(prompt, dim=768)
            self._send_json(200, {"embedding": vector})
        elif path == "/api/chat":
            self._handle_chat(payload)
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_chat(self, payload: Dict[str, Any]) -> None:
        messages = payload.get("messages", [])
        stream = payload.get("stream", False)
        fmt = payload.get("format")
        model = payload.get("model", "llama3.2:3b")

        if fmt == "json":
            reply_text = json.dumps({"memories": [], "affection_delta": 0, "entities": [], "relations": []})
        else:
            reply_text = pick_canned_reply(messages)

        if not stream:
            response_data = {
                "model": model,
                "message": {"role": "assistant", "content": reply_text},
                "done": True,
            }
            self._send_json(200, response_data)
            return

        # Streamed response: send newline-delimited JSON chunks.
        # Ensure at least 2 JSON lines carry message.content, and the last line has "done": true.
        tokens = reply_text.split(" ")
        if len(tokens) >= 2:
            mid = len(tokens) // 2
            chunk1 = " ".join(tokens[:mid]) + " "
            chunk2 = " ".join(tokens[mid:])
            chunks = [chunk1, chunk2]
        else:
            chunks = [reply_text, ""]

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        self.end_headers()

        for chunk in chunks:
            if chunk:
                line = json.dumps({
                    "model": model,
                    "message": {"role": "assistant", "content": chunk},
                    "done": False,
                }) + "\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()

        done_line = json.dumps({
            "model": model,
            "message": {"role": "assistant", "content": ""},
            "done": True,
        }) + "\n"
        self.wfile.write(done_line.encode("utf-8"))
        self.wfile.flush()


def create_server(host: str = "127.0.0.1", port: int = 11434) -> http.server.ThreadingHTTPServer:
    return http.server.ThreadingHTTPServer((host, port), FakeOllamaHandler)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a fake Ollama server for testing and demos.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=11434, help="Port to listen on (default: 11434)")
    args = parser.parse_args()

    server = create_server(args.host, args.port)
    print(f"Fake Ollama server listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down fake Ollama server...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
