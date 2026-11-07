"""Unit tests for tools/fake_ollama.py."""
from __future__ import annotations

import json
import threading
import unittest
import urllib.request

from tools.fake_ollama import create_server


class TestFakeOllama(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = create_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=5.0)

    def test_tags(self) -> None:
        req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("models", data)
            self.assertIsInstance(data["models"], list)
            self.assertTrue(len(data["models"]) > 0)
            model_names = [m["name"] for m in data["models"]]
            self.assertIn("llama3.2:3b", model_names)

    def test_pull(self) -> None:
        body = json.dumps({"name": "llama3.2:3b"}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/pull",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "success")

    def test_chat_stream(self) -> None:
        body = json.dumps({
            "model": "llama3.2:3b",
            "messages": [{"role": "user", "content": "hello there"}],
            "stream": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        lines = []
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            for raw_line in resp:
                text = raw_line.decode("utf-8").strip()
                if text:
                    lines.append(json.loads(text))

        self.assertGreaterEqual(len(lines), 2)
        content_lines = [line for line in lines if line.get("message", {}).get("content")]
        self.assertGreaterEqual(len(content_lines), 2)
        self.assertTrue(lines[-1].get("done"))

    def test_chat_non_stream_json(self) -> None:
        body = json.dumps({
            "model": "llama3.2:3b",
            "messages": [{"role": "user", "content": "Extract memories"}],
            "stream": False,
            "format": "json",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data.get("done"))
            content_str = data.get("message", {}).get("content", "")
            parsed = json.loads(content_str)
            self.assertIn("memories", parsed)
            self.assertIn("affection_delta", parsed)

    def test_embeddings_deterministic(self) -> None:
        body = json.dumps({
            "model": "nomic-embed-text",
            "prompt": "Test embedding vector for companion memory",
        }).encode("utf-8")

        def _get_embedding() -> list[float]:
            req = urllib.request.Request(
                f"{self.base_url}/api/embeddings",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("embedding", [])

        vec1 = _get_embedding()
        vec2 = _get_embedding()
        self.assertEqual(len(vec1), 768)
        self.assertEqual(len(vec2), 768)
        self.assertEqual(vec1, vec2)
        for val in vec1:
            self.assertGreaterEqual(val, -1.0)
            self.assertLessEqual(val, 1.0)


if __name__ == "__main__":
    unittest.main()
