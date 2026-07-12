"""Benchmark tools + providers: p50/p95 latency, written to config/latency.json.

    python tools/bench_tools.py [N]

The router reads config/latency.json to decide when react-then-deliver kicks
in (expected wait > router.react_then_deliver_after_s). Run after changing
models or providers.
"""
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT, load_settings  # noqa: E402
from app.db import Database  # noqa: E402
from app.llm import OllamaClient  # noqa: E402
from app.providers import BrainUnavailable, CloudBrain  # noqa: E402
from app.tools import ToolRunner  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3


async def timed(name, coro_factory, n=N):
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        try:
            await coro_factory()
            times.append(time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            print(f"  {name} run {i + 1}: FAILED ({e})")
    if not times:
        return None
    p50 = statistics.median(times)
    p95 = max(times) if len(times) < 20 else statistics.quantiles(times, n=20)[18]
    print(f"  {name}: p50={p50:.2f}s p95={p95:.2f}s (n={len(times)})")
    return {"p50": round(p50, 2), "p95": round(p95, 2)}


async def main():
    settings = load_settings()
    db = Database(settings.db_path)
    llm = OllamaClient(settings.ollama_base_url, settings.ollama_model,
                       settings.ollama_embed_model, settings.ollama_options)
    tools = ToolRunner(db, llm, settings)
    brain = CloudBrain(db, settings)

    results = {}
    print("== local model ==")
    results["local_chat"] = await timed(
        "local_chat", lambda: llm.chat(messages=[{"role": "user", "content": "say hi in 3 words"}]))
    results["arg_extraction"] = await timed(
        "arg_extraction", lambda: tools.extract_args("reminder", "remind me at 6pm to stretch"))
    print("== tools ==")
    results["weather"] = await timed("weather", lambda: tools.run("weather", "weather in chennai"))
    print("== big brain ==")

    async def brain_call():
        try:
            await brain.ask("what is 17*23? answer with the number only")
        except BrainUnavailable as e:
            raise RuntimeError(f"unavailable: {e.reason}") from e
    results["deep"] = await timed("deep", brain_call)

    out = ROOT / "config" / "latency.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({k: v for k, v in results.items() if v}, indent=2))
    print(f"\nwrote {out}")
    await llm.close()
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
