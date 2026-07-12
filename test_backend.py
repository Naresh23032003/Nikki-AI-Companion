#!/usr/bin/env python3
"""Simple client to test the backend running on port 8000."""
import httpx
import json

BASE_URL = "http://localhost:8000"

async def test_health():
    """Test backend health."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{BASE_URL}/health", timeout=5)
            print(f"Health: {resp.status_code}")
            print(json.dumps(resp.json(), indent=2))
        except Exception as e:
            print(f"Health check failed: {e}")

async def test_chat():
    """Test chat endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            payload = {
                "session_id": "test_session",
                "message": "Hello, how are you?",
            }
            resp = await client.post(
                f"{BASE_URL}/chat",
                json=payload,
                timeout=10,
            )
            print(f"\nChat response: {resp.status_code}")
            # Stream the SSE response
            for line in resp.text.split("\n"):
                if line.strip():
                    print(line)
        except Exception as e:
            print(f"Chat failed: {e}")

async def test_history():
    """Fetch chat history."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{BASE_URL}/history/test_session",
                timeout=5,
            )
            print(f"\nHistory: {resp.status_code}")
            print(json.dumps(resp.json(), indent=2))
        except Exception as e:
            print(f"History fetch failed: {e}")

if __name__ == "__main__":
    import asyncio
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "health":
            asyncio.run(test_health())
        elif cmd == "chat":
            asyncio.run(test_chat())
        elif cmd == "history":
            asyncio.run(test_history())
        else:
            print("Usage: python test_backend.py [health|chat|history]")
    else:
        print("Usage: python test_backend.py [health|chat|history]")
        print("\nExamples:")
        print("  python test_backend.py health    # Check if backend is running")
        print("  python test_backend.py chat      # Send a test message")
        print("  python test_backend.py history   # Fetch message history")
