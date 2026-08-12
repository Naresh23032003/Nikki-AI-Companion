# Open-Source Component Survey — Memory (N1)

Every capability was classified **USE / ADAPT / BUILD / HYBRID** before code was
written. Nikki's binding constraints: single-user, local-first, Windows laptop,
already running Ollama + ChromaDB + SQLite, privacy as a product value, and **no
new background services**.

| Component | Project | License | Verdict | Reasoning |
|---|---|---|---|---|
| Consolidation loop | [Mem0](https://github.com/mem0ai/mem0) | Apache 2.0 | **ADAPT pattern, reject dependency** | See below |
| Contradiction / temporal model | [Graphiti / Zep](https://github.com/getzep/graphiti) | Apache 2.0 | **ADAPT model only** | Requires Neo4j, FalkorDB, Neptune or Kuzu — a graph database service. Non-starter for an app that currently needs only SQLite + Chroma. The **bi-temporal edge model** is the right idea and was adopted; Nikki already had half of it in `valid_from`/`valid_until`. |
| Agent memory runtime | [Letta / MemGPT](https://github.com/letta-ai/letta) | Apache 2.0 | **REJECT** | A full agent framework with its own server and runtime. Nikki already has a runtime; adopting Letta means rebuilding the app around its agent abstraction. Its self-editing memory blocks remain interesting for *procedural* memory later. |
| Evaluation methodology | [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | dataset | **USE (methodology)** | A benchmark, not a library. Its question taxonomy — extraction, multi-session, temporal, knowledge update, abstention — became the structure of `tests/memory_eval.py`. |
| Lexical retrieval | SQLite FTS5 | public domain | **USE** | Already compiled into SQLite. Zero new dependency, zero new service. Chosen over `rank_bm25` for exactly that reason. |
| Rank fusion | Reciprocal Rank Fusion | algorithm | **BUILD** | ~15 lines. A dependency would cost more than it saves. |
| Vector index | ChromaDB | Apache 2.0 | **KEEP** | Already integrated and working. No reason to churn it. |

## Why Mem0 was not taken as a dependency

Mem0 is the closest fit on paper — Apache 2.0, works with Ollama and Chroma,
and its extraction→consolidation→retrieval structure is the right shape.

The deciding argument is Nikki-specific. Her extraction prompt encodes something
Mem0's generic extractor cannot express:

> *"Possessives matter: 'your cat', 'your job' = the COMPANION's — never store
> the companion's pets/life as the user's."*

There is a **companion with her own simulated life** in every exchange, and
attributing her life to the user is the single worst failure mode in this
product. That prompt — with its possessive disambiguation, anti-inference rules
and companion/user attribution guards — is the best piece of memory engineering
in the repository. Adopting Mem0 would have replaced it.

Mem0's actual value here is its **ADD / UPDATE / DELETE / NOOP** consolidation
decision loop, which is a *pattern*, not a package. That pattern was adopted,
with one deliberate change: **DELETE became SUPERSEDE**. Destroying a
contradicted memory is what the old code already did wrong; Graphiti's
invalidate-don't-delete semantics are strictly better for a companion who
should be able to answer "where did I *used* to work?"

Secondary factors: Mem0's graph layer wants Neo4j; its self-hosted stack is
three Docker containers (API + pgvector + Neo4j); and it carries no importance
or confidence scoring, so the ranking work would have been needed anyway.

## Net dependency change

**Zero.** No packages added, no services added. `app/memory_core.py` imports
only the Python standard library, which is why the whole suite runs headless.

## License compliance

No third-party code was copied. Mem0 and Graphiti informed the design; both are
Apache 2.0, which permits this in any case. Attribution is recorded here and in
the `app/memory_core.py` module docstring.
