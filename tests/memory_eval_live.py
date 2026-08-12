"""Live retrieval-threshold verification (HOST_VERIFICATION.md §3), NOT part
of the headless pytest/run_evals.py gate.

`tests/memory_eval.py` stubs the ANN index with a deterministic bag-of-words
cosine because ChromaDB/Ollama were unavailable in the build sandbox. That
stub exercises app.memory_core's decision logic (consolidation, ranking) for
free, but the tuned constants in app/memory_core.py - SIM_DISTINCT,
SIM_SAME_TOPIC, RELEVANCE_FLOOR - were only ever checked against bag-of-words
similarity, not a real embedding model. This script re-runs the EXACT SAME
scenarios (imported, not copied, so there is one source of truth) through the
real app.memory.MemoryStore: real Ollama embeddings (nomic-embed-text), real
ChromaDB, real SQLite. Requires Ollama running locally with nomic-embed-text
pulled - this is why it is a standalone script, not a pytest test.

Run:
    python -m tests.memory_eval_live
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCRATCH = Path(tempfile.mkdtemp(prefix="nikki_live_memeval_"))
# Must be set BEFORE app.config/app.db/app.memory import anything that reads
# them at import time - mirrors tests/test_api_routes.py's pattern. Never
# touches the real companion.db / chroma_db.
os.environ["COMPANION_DB_PATH"] = str(_SCRATCH / "eval.db")
os.environ["CHROMA_PATH"] = str(_SCRATCH / "chroma")

from app.config import load_settings  # noqa: E402
from app.db import Database  # noqa: E402
from app.llm import OllamaClient  # noqa: E402
from app.memory import MemoryStore  # noqa: E402
from tests.memory_eval import SCENARIOS  # noqa: E402 - single source of truth

NOW = datetime.now(timezone.utc)


async def run_scenario_live(scenario, memory: MemoryStore) -> tuple[bool, list[str]]:
    for fact, cat, kind in scenario.facts:
        valid_until = None
        if kind == "transient":
            valid_until = (NOW - timedelta(hours=2)).isoformat()
        await memory.add_fact(fact, cat, kind=kind, valid_until=valid_until)

    recalled = await memory.retrieve_memories(scenario.query)
    blob = " | ".join(recalled)
    ok = all(e.lower() in blob.lower() for e in scenario.expect_contains)
    ok = ok and not any(e.lower() in blob.lower() for e in scenario.expect_absent)
    return ok, recalled


async def main() -> int:
    settings = load_settings()
    try:
        async with __import__("httpx").AsyncClient(timeout=3.0) as probe:
            r = await probe.get(f"{settings.ollama_base_url}/api/tags")
            r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"Ollama unreachable at {settings.ollama_base_url}: {e}")
        print("This script needs a live Ollama with nomic-embed-text pulled.")
        return 1

    print(f"Ollama reachable at {settings.ollama_base_url}, "
          f"embed model: {settings.ollama_embed_model}\n")

    results: dict[str, dict[str, int]] = {}
    failures: list[str] = []
    for i, scenario in enumerate(SCENARIOS):
        db = Database(_SCRATCH / f"live_{i}.db")
        llm = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            embed_model=settings.ollama_embed_model,
        )
        memory = MemoryStore(db, llm, settings, collection=f"eval_live_{i}")
        try:
            ok, recalled = await run_scenario_live(scenario, memory)
        finally:
            await llm.close()
            db.close()
        bucket = results.setdefault(scenario.category, {"pass": 0, "total": 0})
        bucket["total"] += 1
        bucket["pass"] += int(ok)
        if not ok:
            failures.append(
                f"  [{scenario.category}] {scenario.name}\n"
                f"      query:    {scenario.query}\n"
                f"      recalled: {recalled}\n"
                f"      expected: +{scenario.expect_contains} -{scenario.expect_absent}")

    print("Live memory evaluation (real Ollama embeddings + real ChromaDB)")
    print("=" * 64)
    total_pass = total_all = 0
    for category, r in sorted(results.items()):
        total_pass += r["pass"]
        total_all += r["total"]
        pct = 100.0 * r["pass"] / r["total"]
        flag = "" if r["pass"] == r["total"] else "   <-- FAIL"
        print(f"  {category:<18} {r['pass']}/{r['total']}  {pct:5.1f}%{flag}")
    print("-" * 64)
    print(f"  {'overall':<18} {total_pass}/{total_all}  "
          f"{100.0 * total_pass / total_all:5.1f}%\n")
    if failures:
        print("Failures:\n" + "\n".join(failures))

    shutil.rmtree(_SCRATCH, ignore_errors=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
