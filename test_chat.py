"""End-to-end test for the companion + long-term memory.

Scenario:
  1. In session A, have a short conversation that mentions a birthday and a
     favorite food.
  2. Wait for background fact-extraction to persist those memories.
  3. Start a fresh session B and ask about both - verify the companion recalls
     the birthday and the favorite food.

Requires the server running (see README) with Ollama up and both models pulled:
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text

Usage:
    python test_chat.py
"""
import sys
import time

import httpx

BASE_URL = "http://localhost:8000"

BIRTHDAY = "March 3rd"
FAVORITE_FOOD = "spicy ramen"


def send(message: str, session_id: str) -> str:
    """Send one message, print the streamed reply, and return the full text."""
    print(f"\n[{session_id}] You: {message}")
    print(f"[{session_id}] Her: ", end="", flush=True)

    reply_parts = []
    with httpx.Client(timeout=180.0) as client:
        with client.stream(
            "POST",
            f"{BASE_URL}/chat",
            json={"message": message, "session_id": session_id},
        ) as resp:
            resp.raise_for_status()
            event = None
            for line in resp.iter_lines():
                if not line:
                    event = None
                    continue
                if line.startswith("event: "):
                    event = line[len("event: "):].strip()
                elif line.startswith("data: "):
                    payload = _json(line[len("data: "):])
                    if event == "token":
                        tok = payload.get("token", "")
                        reply_parts.append(tok)
                        print(tok, end="", flush=True)
                    elif event == "error":
                        print(f"\n[error] {payload.get('error')}")
                    elif event == "done":
                        print()
    return "".join(reply_parts)


def _json(s: str):
    import json

    return json.loads(s)


def show_memories() -> None:
    r = httpx.get(f"{BASE_URL}/memories", timeout=30.0)
    r.raise_for_status()
    mems = r.json()["memories"]
    print(f"\n--- stored memories ({len(mems)}) ---")
    for m in mems:
        print(f"  #{m['id']} [{m['category']}] {m['fact']}")
    print("-" * 30)


def main() -> None:
    session_a = f"mem-test-a-{int(time.time())}"
    session_b = f"mem-test-b-{int(time.time())}"

    print("=" * 60)
    print("PHASE 1 - conversation that drops facts to remember")
    print("=" * 60)
    send(f"hey! just so you know, my birthday is {BIRTHDAY} 🎂", session_a)
    send(f"also i'm obsessed with {FAVORITE_FOOD}, could eat it every day", session_a)

    # Give the background fact-extraction + embedding time to finish.
    print("\n...waiting for background memory extraction...")
    time.sleep(8)
    show_memories()

    print("\n" + "=" * 60)
    print("PHASE 2 - brand new session; does she remember?")
    print("=" * 60)
    r1 = send("quick - do you remember when my birthday is?", session_b)
    r2 = send("and what's that food i can't get enough of?", session_b)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    combined = (r1 + " " + r2).lower()
    birthday_ok = "march" in combined or "3rd" in combined or "3" in combined
    food_ok = "ramen" in combined

    print(f"  birthday recalled ({BIRTHDAY}): {'PASS' if birthday_ok else 'FAIL'}")
    print(f"  favorite food recalled ({FAVORITE_FOOD}): {'PASS' if food_ok else 'FAIL'}")

    if birthday_ok and food_ok:
        print("\n✅ Long-term memory works across sessions.")
        sys.exit(0)
    else:
        print("\n❌ One or both facts were not recalled. Check server logs / memories.")
        sys.exit(1)


if __name__ == "__main__":
    main()
