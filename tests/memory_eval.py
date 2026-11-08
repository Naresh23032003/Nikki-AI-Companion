"""Memory evaluation harness — scenario-level, headless.

Runs multi-turn scenarios through the real consolidation and ranking code
(app.memory_core) against a real SQLite database (app.db), and scores the
result. Only the ANN index is stubbed: ChromaDB is used in production purely as
an index over vectors Nikki computes herself, so substituting a deterministic
similarity function exercises the same thresholds without needing Ollama, a
GPU, or a 2GB embedding model in CI.

Scenario categories follow the LongMemEval taxonomy, which is the closest thing
to a standard for this problem:

  extraction        can a stated fact be recalled later?
  knowledge_update  when a fact changes, is the NEW value returned and the old
                    one retired rather than both being offered?
  temporal          are expired and aged-out memories excluded?
  multi_session     is a fact learned long ago still retrievable?
  abstention        when nothing relevant is known, is nothing returned?

Abstention is the one most memory systems quietly fail, and the one that
produces the "why is she bringing that up?" feeling. It is weighted here
accordingly.

Run standalone for a report:

    python -m tests.memory_eval
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Sequence

from app.db import Database
from app.memory_core import (
    Consolidation,
    adjust_confidence,
    build_record,
    decide_consolidation,
    is_temporally_current,
    rerank,
    rrf_fuse,
    score_importance,
)

NOW = datetime.now(timezone.utc)

_STOP = {"the", "a", "an", "is", "are", "was", "to", "of", "in", "on", "at",
         "for", "and", "or", "user", "users", "my", "i", "what", "whats",
         "where", "who", "when", "does", "do", "did", "s"}


def _tokens(text: str) -> set[str]:
    raw = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    return {t for t in raw if t not in _STOP and len(t) > 1}


def _cosine(a: set[str], b: set[str]) -> float:
    """Cosine over binary bag-of-words vectors — deterministic stand-in for the
    embedding model. Ordered like a real embedder for these inputs: same
    sentence ~1.0, paraphrase high, unrelated near 0."""
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


class FakeIndex:
    """Minimal stand-in for the Chroma collection."""

    def __init__(self) -> None:
        self._docs: Dict[int, set[str]] = {}

    def add(self, mid: int, text: str) -> None:
        self._docs[mid] = _tokens(text)

    def update(self, mid: int, text: str) -> None:
        self._docs[mid] = _tokens(text)

    def delete(self, mid: int) -> None:
        self._docs.pop(mid, None)

    def count(self) -> int:
        return len(self._docs)

    def query(self, text: str, n: int) -> List[tuple[int, float]]:
        q = _tokens(text)
        scored = [(mid, _cosine(q, toks)) for mid, toks in self._docs.items()]
        scored.sort(key=lambda p: p[1], reverse=True)
        return [(mid, s) for mid, s in scored[:n] if s > 0]


class MemoryHarness:
    """Thin wiring around the real core functions.

    Deliberately mirrors app/memory.py's calls rather than reimplementing the
    decisions, so what is measured here is the shipped logic.
    """

    def __init__(self, db: Database) -> None:
        self.db = db
        self.index = FakeIndex()

    def remember(self, fact: str, category: str = "personal_info",
                 kind: str = "permanent", *, user_asserted: bool = False,
                 event_datetime: str | None = None,
                 valid_until: str | None = None) -> int | None:
        candidates = []
        for mid, sim in self.index.query(fact, 5):
            row = self.db.get_memory(mid)
            if row:
                candidates.append((build_record(row), sim))

        result = decide_consolidation(fact, candidates, user_asserted=user_asserted)
        importance = score_importance(fact, category, kind)

        if result.decision is Consolidation.NOOP:
            return None

        if result.decision is Consolidation.REINFORCE and result.target_id:
            target = next((r for r, _ in candidates if r.id == result.target_id), None)
            self.db.reinforce_memory(
                result.target_id,
                adjust_confidence(target.confidence if target else 0.7,
                                  reinforced=True, user_asserted=user_asserted),
            )
            return result.target_id

        if result.decision is Consolidation.UPDATE and result.target_id:
            self.db.update_memory(result.target_id, result.merged_fact or fact,
                                  category, kind=kind)
            self.index.update(result.target_id, result.merged_fact or fact)
            return result.target_id

        mid = self.db.add_memory(fact, category, kind=kind,
                                 event_datetime=event_datetime,
                                 valid_until=valid_until)
        self.db.set_memory_scores(mid, importance=importance,
                                  confidence=0.95 if user_asserted else 0.7)
        self.index.add(mid, fact)

        if result.decision is Consolidation.SUPERSEDE and result.target_id:
            self.db.supersede_memory(result.target_id, mid, result.reason)
            self.index.delete(result.target_id)
        return mid

    def recall(self, query: str, k: int = 5) -> List[str]:
        pool = max(k * 3, 20)
        dense = [mid for mid, _ in self.index.query(query, pool)]
        lexical = self.db.search_memories_lexical(query, limit=pool)
        fused = rrf_fuse({"dense": dense, "lexical": lexical},
                         weights={"dense": 1.0, "lexical": 0.7})
        if not fused:
            return []
        records = [build_record(r) for r in self.db.get_memories_by_ids(list(fused))]
        records = [r for r in records if is_temporally_current(r)]
        return [r.fact for r, _ in rerank(records, fused, limit=k)]


# ---------------------------------------------------------------- scenarios


@dataclass
class Scenario:
    name: str
    category: str
    facts: List[tuple[str, str, str]]          # (fact, category, kind)
    query: str
    expect_contains: List[str] = field(default_factory=list)
    expect_absent: List[str] = field(default_factory=list)


SCENARIOS: Sequence[Scenario] = [
    Scenario(
        name="recalls a stated preference",
        category="extraction",
        facts=[("User dislikes coffee", "preference", "permanent"),
               ("User enjoys long walks", "preference", "permanent")],
        query="do I like coffee",
        expect_contains=["dislikes coffee"],
    ),
    Scenario(
        name="recalls a name by exact token",
        category="extraction",
        facts=[("User's sister is named Meera", "relationship", "permanent"),
               ("User enjoys long walks", "preference", "permanent")],
        query="what is my sister called",
        expect_contains=["Meera"],
    ),
    Scenario(
        name="employer change returns only the new employer",
        category="knowledge_update",
        facts=[("User works at Google", "personal_info", "permanent"),
               ("User works at Stripe", "personal_info", "permanent")],
        query="where do I work",
        expect_contains=["Stripe"],
        expect_absent=["Google"],
    ),
    Scenario(
        name="city change returns only the new city",
        category="knowledge_update",
        facts=[("User lives in Bangalore", "personal_info", "permanent"),
               ("User lives in Chennai", "personal_info", "permanent")],
        query="where do I live",
        expect_contains=["Chennai"],
        expect_absent=["Bangalore"],
    ),
    Scenario(
        name="expired transient state is not recalled",
        category="temporal",
        facts=[("User has a headache", "emotion", "transient")],
        query="how am I feeling",
        expect_absent=["headache"],
    ),
    Scenario(
        name="durable fact survives many later memories",
        category="multi_session",
        facts=([("User's birthday is March 3rd", "personal_info", "permanent")]
               + [(f"User watched film number {i}", "event", "event")
                  for i in range(20)]),
        query="when is my birthday",
        expect_contains=["March 3rd"],
    ),
    Scenario(
        name="unrelated question returns nothing",
        category="abstention",
        facts=[("User's sister is named Meera", "relationship", "permanent"),
               ("User dislikes coffee", "preference", "permanent")],
        query="what is the capital of Peru",
        expect_absent=["Meera", "coffee"],
    ),
    Scenario(
        name="recent but irrelevant memory is not injected",
        category="abstention",
        facts=[("User's exam is on Friday", "event", "event"),
               ("User said ok", "emotion", "transient")],
        query="what is the capital of Peru",
        expect_absent=["ok", "exam"],
    ),
]


def run_scenario(scenario: Scenario, db: Database) -> tuple[bool, List[str]]:
    harness = MemoryHarness(db)
    for fact, cat, kind in scenario.facts:
        valid_until = None
        if kind == "transient":
            # already expired
            valid_until = (NOW - timedelta(hours=2)).isoformat()
        harness.remember(fact, cat, kind, valid_until=valid_until)

    recalled = harness.recall(scenario.query)
    blob = " | ".join(recalled)
    ok = all(e.lower() in blob.lower() for e in scenario.expect_contains)
    ok = ok and not any(e.lower() in blob.lower() for e in scenario.expect_absent)
    return ok, recalled


def run_eval(tmp_dir) -> Dict[str, Dict[str, int]]:
    results: Dict[str, Dict[str, int]] = {}
    for i, scenario in enumerate(SCENARIOS):
        db = Database(tmp_dir / f"eval_{i}.db")
        try:
            ok, _ = run_scenario(scenario, db)
        finally:
            try:
                db.close()
            except Exception:
                pass
        bucket = results.setdefault(scenario.category, {"pass": 0, "total": 0})
        bucket["total"] += 1
        bucket["pass"] += int(ok)
    return results


# -------------------------------------------------------------------- tests


def test_memory_eval_suite(tmp_path):
    """Every scenario must pass. Failures print the recalled set for diagnosis."""
    failures = []
    for i, scenario in enumerate(SCENARIOS):
        db = Database(tmp_path / f"s{i}.db")
        try:
            ok, recalled = run_scenario(scenario, db)
        finally:
            try:
                db.close()
            except Exception:
                pass
        if not ok:
            failures.append(f"  [{scenario.category}] {scenario.name}\n"
                            f"      query:    {scenario.query}\n"
                            f"      recalled: {recalled}\n"
                            f"      expected: +{scenario.expect_contains} "
                            f"-{scenario.expect_absent}")
    assert not failures, "memory eval failures:\n" + "\n".join(failures)


def test_superseded_fact_remains_in_history(tmp_path):
    """Knowledge updates must retire the old value, not erase it."""
    db = Database(tmp_path / "hist.db")
    try:
        h = MemoryHarness(db)
        old = h.remember("User works at Google", "personal_info")
        h.remember("User works at Stripe", "personal_info")

        assert "Google" not in " ".join(h.recall("where do I work"))
        row = db.get_memory(old)
        assert row is not None and row["fact"] == "User works at Google"
        assert db.get_memory_history(old)[0]["operation"] == "supersede"
    finally:
        db.close()


if __name__ == "__main__":  # pragma: no cover
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        results = run_eval(Path(tmp))

    print("\nMemory evaluation\n" + "=" * 46)
    total_pass = total_all = 0
    for category, r in sorted(results.items()):
        total_pass += r["pass"]
        total_all += r["total"]
        pct = 100.0 * r["pass"] / r["total"]
        flag = "" if r["pass"] == r["total"] else "   <-- FAIL"
        print(f"  {category:<18} {r['pass']}/{r['total']}  {pct:5.1f}%{flag}")
    print("-" * 46)
    print(f"  {'overall':<18} {total_pass}/{total_all}  "
          f"{100.0 * total_pass / total_all:5.1f}%\n")
