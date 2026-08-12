# Nikki — Memory Architecture

Status: **N1 implemented.** 84 headless tests + an 8-scenario evaluation suite,
all green without Ollama, ChromaDB or a GPU.

---

## Why this was rebuilt

The previous pipeline was `extract → embed → store → top-K cosine → inject`,
with one destructive shortcut at its centre:

```python
existing_id = self._nearest_duplicate(embedding)   # nearest vector above threshold
if existing_id is not None:
    self.db.update_memory(existing_id, fact, ...)  # overwrite it, in place
```

"User works at Google" and "User works at Stripe" are near-identical vectors.
So **duplication and contradiction used the same code path**, and correcting a
fact silently destroyed the previous one with no history. That is data loss, not
a missing feature — and it is why the fix was prioritised over everything else.

Five further gaps followed from the same flat design:

| Gap | Consequence |
|---|---|
| No importance score | A birthday ranked equal to this morning's chai |
| No confidence | No way to represent uncertainty or strengthen on repetition |
| `access_count` written, never read | Bookkeeping with no effect on ranking |
| Unconditional "recent N" union | Irrelevant memories injected into *every* prompt |
| Vector-only retrieval | Exact-name questions depended entirely on embedding recall |

---

## The architecture

```
                        incoming fact
                              │
                    ┌─────────▼─────────┐
                    │  nearest 5 by     │
                    │  cosine (Chroma)  │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼──────────────────────────┐
                    │  CONSOLIDATION (memory_core.py)    │
                    │                                    │
                    │  structural contradiction scan     │
                    │  across ALL candidates ────────────┼──► SUPERSEDE
                    │  (not gated by similarity)         │
                    │            │                       │
                    │  sim < 0.72 ──────────────────────┼──► ADD
                    │  sim ≥ 0.93, identical ───────────┼──► REINFORCE
                    │  sim ≥ 0.93, adds detail ─────────┼──► UPDATE
                    │  otherwise ───────────────────────┼──► ADD
                    └────────────────────────────────────┘

     retrieval
         │
    ┌────┴────┬──────────────┐
    │ dense   │ lexical      │ graph
    │ Chroma  │ SQLite FTS5  │ entities/relations
    └────┬────┴──────┬───────┘
         └─────┬─────┘
          RRF fusion (position-based, no score calibration)
               │
          RERANK  0.55·relevance + 0.20·importance
                + 0.15·confidence + 0.10·recency-decay
               │
          relevance floor (0.15) — drops anything unrelated
               │
            top-k → prompt
```

### Scoring

**Importance** is heuristic, not a model call — it is computed for every stored
fact, and an LLM round-trip per fact would put a local model on the critical
path of the background extraction queue. Signals: category prior, kind prior,
stability cues (`always`, `birthday`, `sister`, `works at`…), entity presence,
and reinforcement with diminishing returns.

**Confidence** starts at 0.7, approaches 1.0 asymptotically on independent
restatement (repetition never manufactures certainty), halves on contradiction,
and is **pinned to 0.95 when the user asserts a fact directly** — a correction
must take effect on the next turn, not several turns later.

**Decay** is exponential with a per-kind half-life, stretched by importance and
by access count (crude spaced repetition — a memory you keep needing stays
available). `last_accessed` takes precedence over `created_at`, so a year-old
fact retrieved yesterday is still fresh.

| kind | base half-life |
|---|---|
| permanent | 3650 d |
| recurring | 365 d |
| event | 45 d |
| transient | 0.5 d |

### Supersession (bi-temporal)

A contradicted memory is **retired, never deleted**: `superseded_by` and
`t_invalid` are set, its vector is dropped from the ANN index, and a row is
appended to `memory_revisions`. The fact stays in SQLite, so:

- "Where do I work?" → Stripe
- "Where did I used to work?" → answerable from history
- A bad merge is recoverable

### Retrieval

Two channels fused by **Reciprocal Rank Fusion** — position-based, so the dense
cosine channel and the lexical FTS5 channel combine without calibrating two
incompatible score distributions. FTS5 ships inside SQLite, so the lexical
channel costs no new dependency and no new service, and it keeps working when
Ollama is busy and the embedding call times out.

The **relevance floor** is what replaced the unconditional recent-N union.
Recency is now a ranking signal, not a guaranteed ticket into the prompt.
Importance is a tie-breaker and explicitly *cannot* rescue an irrelevant
memory — there is a test asserting exactly that.

---

## Schema changes

Added to `memories` (non-destructive `ALTER TABLE`, backfilled, idempotent):
`importance`, `confidence`, `superseded_by`, `superseded_at`, `t_invalid`,
`reinforced_count`.

New: `memory_revisions` (append-only history), `memories_fts` (FTS5 virtual
table kept in sync by insert/update/delete triggers, with backfill for existing
databases).

A test upgrades a hand-built pre-migration database and asserts facts, access
counts and ids all survive.

---

## What the evaluation measures

Scenario categories follow the LongMemEval taxonomy:

| category | scenarios | result |
|---|---|---|
| extraction | 2 | 2/2 |
| knowledge_update | 2 | 2/2 |
| temporal | 1 | 1/1 |
| multi_session | 1 | 1/1 |
| abstention | 2 | 2/2 |

`python -m tests.memory_eval` prints the report.

**Abstention is weighted deliberately.** It is the category most memory systems
quietly fail, and it is the direct cause of the "why is she bringing that up?"
feeling.

### A bug the evaluation found

The eval initially scored **knowledge_update 0/2**. Contradiction detection was
gated behind the similarity threshold, so any contradictory pair an embedder
scores as merely "related" slipped through and *both* facts were stored — the
companion believing you work at two places at once.

The fix: structural predicate conflict is checked across **all** candidates and
is not gated by cosine distance. A functional-predicate conflict is stronger
evidence that a fact changed than vector distance is that it didn't. Locked in
by three regression tests.

This is the argument for building the eval harness alongside the feature rather
than after it — the unit tests were all green while this was broken.

---

## Deliberately not done

- **No LLM adjudication call is wired in yet.** `needs_llm_adjudication()`
  exists and is tested; the call site is not. Structural rules settle the common
  cases for free, and the escalation path should be added with a measured
  false-positive rate rather than on principle.
- **No consolidation/summarisation job.** Episodic→semantic rollup is real work
  and belongs in its own pass with its own evaluation.
- **Memory classes are not separate stores.** Episodic/semantic/relationship
  distinctions ride on the existing `category` + `kind` columns. Splitting them
  into separate stores would be architecture for its own sake at this data volume.

---

## Verification still required on the Windows host

Everything above is tested headlessly. These need the real stack:

1. `pip install -r requirements.txt && pytest tests/` with ChromaDB present —
   confirms `tests/test_temporal_memory.py` (chromadb-gated) still passes.
2. Start the app against the **real `companion.db`** and confirm the migration
   applies cleanly to production data. Back it up first; `backups/` already
   exists and `_nightly_backup` runs.
3. Confirm FTS5 is compiled into the host Python's SQLite
   (`sqlite3.connect(':memory:').execute("CREATE VIRTUAL TABLE t USING fts5(x)")`).
   If absent, lexical retrieval self-disables and the system degrades to
   vector-only — by design, not by accident.
4. Exercise a live correction: tell her you changed jobs, then ask where you
   work. Expect the new employer only, and the old one intact in
   `memory_revisions`.
