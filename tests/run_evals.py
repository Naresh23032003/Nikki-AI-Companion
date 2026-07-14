"""One command to run the eval + unit suites and print a summary table.

    python tests/run_evals.py            # deterministic suites only (no Ollama)
    python tests/run_evals.py --llm      # also run the Ollama-backed evals

The deterministic suites need no model and are what CI runs. The --llm suites
(memory extraction, tool-calling) require a local Ollama with the models pulled;
they print their own scores.
"""
from __future__ import annotations

import argparse
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _run_routing() -> tuple[str, bool]:
    from tests import routing_eval

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = routing_eval.main()
    out = buf.getvalue()
    # Surface the score lines only.
    score_lines = [ln for ln in out.splitlines() if "passed" in ln or "FAIL" in ln]
    return "\n".join(score_lines), rc == 0


def _run_unittests(module: str) -> tuple[str, bool]:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(module)
    buf = io.StringIO()
    result = unittest.TextTestRunner(stream=buf, verbosity=1).run(suite)
    n = result.testsRun
    ok = result.wasSuccessful()
    passed = n - len(result.failures) - len(result.errors)
    return f"{passed}/{n} tests passed", ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also run Ollama-backed evals")
    args = ap.parse_args()

    print("=" * 60)
    print("Deterministic suites (no model required)")
    print("=" * 60)

    results: list[tuple[str, str, bool]] = []

    summary, ok = _run_routing()
    print(f"\n[routing + tone guards]\n{summary}")
    results.append(("routing + tone guards", summary.replace("\n", " | "), ok))

    for label, mod in [
        ("temporal memory", "tests.test_temporal_memory"),
        ("journal patterns", "tests.test_journal_patterns"),
    ]:
        summary, ok = _run_unittests(mod)
        print(f"\n[{label}] {summary}")
        results.append((label, summary, ok))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for label, summary, ok in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {label:24} {summary}")

    if args.llm:
        print("\n" + "=" * 60)
        print("Ollama-backed evals (need a local Ollama + pulled models)")
        print("=" * 60)
        import asyncio

        from tests import extraction_eval, tool_calling_eval

        print("\n--- memory extraction ---")
        try:
            asyncio.run(extraction_eval.run(extraction_eval.load_settings().ollama_extract_model))
        except Exception as e:  # noqa: BLE001
            print(f"  skipped: {e}")
        print("\n--- tool calling ---")
        try:
            asyncio.run(tool_calling_eval.run("llama3.2:3b"))
        except Exception as e:  # noqa: BLE001
            print(f"  skipped: {e}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
