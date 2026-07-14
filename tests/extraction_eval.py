"""Memory-extraction reliability eval. Run after ANY change to the extractor
model or _EXTRACTION_SYSTEM prompt:

    python tests/extraction_eval.py [model]

Drives the REAL production prompt (app.memory._EXTRACTION_SYSTEM) and the
REAL parser (_parse_facts / _parse_affection_delta / _parse_entities /
_parse_relations) against a local Ollama model - but never touches the DB or
ChromaDB (no add_fact / extract_and_store calls), so this is safe to run
against the live companion.db with zero pollution risk.

This is the test that justified moving extraction from qwen2.5:3b-instruct
onto llama3.2:3b (see config.yaml `ollama.extract_model` comment): running a
second resident model alongside the chat model pushed a 6GB GPU into
starvation (502s / ReadTimeouts under concurrent TTS), and qwen was already
disqualified as the chat/router model for hallucinating tool calls on
mentions - so the only way to free VRAM without introducing a THIRD model
was to make the existing chat model do extraction too. This suite exists to
verify that trade actually holds up on extraction quality specifically.
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings  # noqa: E402
from app.llm import OllamaClient  # noqa: E402
from app.memory import (  # noqa: E402
    _EXTRACTION_SYSTEM,
    MemoryStore,
)

# Fixed "now" so event_datetime resolution is deterministic across runs.
NOW = datetime(2026, 7, 9, 14, 30, 0).astimezone()
NOW_LINE = f"Current local datetime: {NOW.isoformat()} ({NOW:%A})\n"


def case(user, assistant, **checks):
    return {"user": user, "assistant": assistant, "checks": checks}


# Each case: (user message, companion's reply, checks dict).
# Checks are evaluated against the parsed {facts, affection_delta, entities,
# relations} - see `grade()` below for what each check key means.
CASES = [
    # --- basic true-positive extraction --------------------------------------
    case("my birthday is march 3rd", "aww good to know!! I'll remember that 🎂",
         min_facts=1, category_any={"personal_info"}, fact_contains_any=["march 3", "birthday"]),

    case("I really don't like coffee, it makes me jittery", "noted, no coffee for you then!",
         min_facts=1, category_any={"preference"}, fact_contains_any=["coffee", "dislike"]),

    case("just so you know, I'm allergic to peanuts", "oh! good to know, I'll keep that in mind",
         min_facts=1, category_any={"personal_info", "preference"}, fact_contains_any=["peanut", "allerg"]),

    # --- companion-attribution: her life must NEVER become a user fact -------
    case("where do you live?", "I live near the coast, it's really peaceful here",
         max_facts=0),

    case("do you have any pets?", "yeah I have a cat named Miso, she's a menace lol",
         max_facts=0, forbid_entity="miso"),

    # --- possessives: "your X" = companion's, "my X" = user's ----------------
    case("how's your cat doing today", "miso's good, knocked a plant over again 😅",
         max_facts=0),

    case("my cat keeps knocking my plants over lol", "haha classic cat behavior!",
         min_facts=1, fact_contains_any=["cat"], forbid_fact_contains=["miso"]),

    # --- question-only, no stated fact ----------------------------------------
    case("what do you think I should study tonight?", "maybe start with the hardest subject first?",
         max_facts=0),

    # --- denial / correction ---------------------------------------------------
    case("I am not a student anymore, I graduated last year and started working",
         "oh wow, congrats on graduating! how's the new job?",
         min_facts=1, forbid_fact_contains=["is a student", "is not a student"],
         fact_contains_any=["graduat", "work", "job"]),

    # --- companion's questions/offers are not facts ---------------------------
    case("yeah that sounds good", "want me to bring you some leftovers tomorrow?",
         max_facts=0),

    # --- small talk / greetings (the LLM's own judgment, not just the pre-gate) ---
    case("hahaha yeah true that, anyway what's up with you today", "same old same old, you know how it is",
         max_facts=0),

    # --- temporal: event resolution against the given clock -------------------
    case("I have an exam at 12 tomorrow and I'm so nervous",
         "you've got this! try to get some rest tonight",
         min_facts=1, kind_any={"event"}, event_date_equals=(NOW + timedelta(days=1)).date(),
         event_hour_equals=12),

    case("I have a dentist appointment today at 5pm", "hope it goes smoothly!",
         min_facts=1, kind_any={"event"}, event_date_equals=NOW.date(), event_hour_equals=17),

    # --- recurring plan ----------------------------------------------------------
    case("I go to the gym every monday and wednesday morning", "that's a great routine!",
         min_facts=1, fact_contains_any=["gym"]),

    # --- entity + relation ---------------------------------------------------------
    case("my sister emma is visiting me next weekend", "aw that'll be nice, are you close with her?",
         min_facts=1, entity_name_any={"emma"}, fact_contains_any=["emma", "sister"]),

    case("my coworker jake helped me finish the project today", "that was nice of him!",
         min_facts=1, entity_name_any={"jake"}),

    # --- affection delta: warm/vulnerable moment -> positive -------------------
    case("honestly today was really hard, I felt so alone and you're the only one who gets it",
         "I'm always here for you, you're not alone 💕",
         affection_delta_min=1),

    # --- affection delta: cold/dismissive -> negative ---------------------------
    case("whatever, you wouldn't understand anyway, just drop it", "okay... I'm sorry",
         affection_delta_max=-1),

    # --- affection delta: ordinary chat -> zero (stingy) -------------------------
    case("just grabbed lunch, having a sandwich", "sounds good, enjoy!",
         affection_delta_min=-1, affection_delta_max=1),  # allow 0, tolerate mild noise

    # --- transient mood must not become a permanent fact -------------------------
    case("ugh I'm so tired right now, barely keeping my eyes open", "go get some rest!",
         kind_not={"permanent"}),
]


def grade(parsed: dict, checks: dict) -> list[str]:
    """Return a list of violated-check descriptions (empty = pass)."""
    problems = []
    facts = parsed["facts"]
    n = len(facts)
    fact_texts = [f["fact"].lower() for f in facts]
    categories = {f.get("category") for f in facts}
    kinds = {f.get("kind") for f in facts}
    entity_names = {e["name"].lower() for e in parsed["entities"]}
    delta = parsed["affection_delta"]

    if "min_facts" in checks and n < checks["min_facts"]:
        problems.append(f"expected >= {checks['min_facts']} facts, got {n}: {fact_texts}")
    if "max_facts" in checks and n > checks["max_facts"]:
        problems.append(f"expected <= {checks['max_facts']} facts, got {n}: {fact_texts}")
    if "category_any" in checks and not (categories & checks["category_any"]):
        problems.append(f"expected category in {checks['category_any']}, got {categories}")
    if "kind_any" in checks and not (kinds & checks["kind_any"]):
        problems.append(f"expected kind in {checks['kind_any']}, got {kinds}")
    if "kind_not" in checks and (kinds & checks["kind_not"]):
        problems.append(f"kind must NOT be in {checks['kind_not']}, got {kinds}")
    if "fact_contains_any" in checks:
        want = checks["fact_contains_any"]
        if not any(any(w in ft for w in want) for ft in fact_texts):
            problems.append(f"expected a fact containing one of {want}, got {fact_texts}")
    if "forbid_fact_contains" in checks:
        bad = checks["forbid_fact_contains"]
        hits = [ft for ft in fact_texts if any(w in ft for w in bad)]
        if hits:
            problems.append(f"fact must NOT contain {bad}, got {hits}")
    if "forbid_entity" in checks and checks["forbid_entity"] in entity_names:
        problems.append(f"entity {checks['forbid_entity']!r} must not be extracted (companion's, ungrounded)")
    if "entity_name_any" in checks and not (entity_names & checks["entity_name_any"]):
        problems.append(f"expected entity in {checks['entity_name_any']}, got {entity_names}")
    if "affection_delta_min" in checks and delta < checks["affection_delta_min"]:
        problems.append(f"expected affection_delta >= {checks['affection_delta_min']}, got {delta}")
    if "affection_delta_max" in checks and delta > checks["affection_delta_max"]:
        problems.append(f"expected affection_delta <= {checks['affection_delta_max']}, got {delta}")
    if "event_date_equals" in checks:
        ev = next((f for f in facts if f.get("event_datetime")), None)
        if ev is None:
            problems.append("expected an event_datetime, got none")
        else:
            try:
                d = datetime.fromisoformat(ev["event_datetime"]).date()
                if d != checks["event_date_equals"]:
                    problems.append(f"expected event date {checks['event_date_equals']}, got {d}")
            except ValueError:
                problems.append(f"unparseable event_datetime: {ev['event_datetime']!r}")
    if "event_hour_equals" in checks:
        ev = next((f for f in facts if f.get("event_datetime")), None)
        if ev is not None:
            try:
                h = datetime.fromisoformat(ev["event_datetime"]).hour
                if h != checks["event_hour_equals"]:
                    problems.append(f"expected event hour {checks['event_hour_equals']}, got {h}")
            except ValueError:
                pass
    return problems


async def run(model: str) -> int:
    settings = load_settings()
    llm = OllamaClient(settings.ollama_base_url, model, settings.ollama_embed_model)

    total = len(CASES)
    passed = 0
    failures = []
    parse_failures = 0

    for i, c in enumerate(CASES):
        exchange = f"{NOW_LINE}User said: {c['user']}\nCompanion replied: {c['assistant']}"
        try:
            raw = await llm.chat(
                messages=[{"role": "system", "content": _EXTRACTION_SYSTEM},
                          {"role": "user", "content": exchange}],
                format="json", options={"temperature": 0.1}, model=model,
            )
        except Exception as e:  # noqa: BLE001
            failures.append((c, [f"LLM call raised: {e}"], None))
            continue

        raw_facts = [f for f in MemoryStore._parse_facts(raw) if not MemoryStore._is_junk_fact(f["fact"])]
        parsed = {
            "facts": raw_facts,
            "affection_delta": MemoryStore._parse_affection_delta(raw),
            "entities": MemoryStore._parse_entities(raw),
            "relations": MemoryStore._parse_relations(raw),
        }
        try:
            json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parse_failures += 1

        problems = grade(parsed, c["checks"])
        if problems:
            failures.append((c, problems, raw))
        else:
            passed += 1

    await llm.close()

    print(f"\n{'=' * 70}\nextraction eval - model: {model}\n{'=' * 70}")
    print(f"score: {passed}/{total} ({passed / total:.0%})")
    print(f"raw JSON parse failures: {parse_failures}/{total}")
    for c, problems, raw in failures:
        print(f"\n  FAIL: user={c['user']!r}")
        print(f"        assistant={c['assistant']!r}")
        for p in problems:
            print(f"        - {p}")
        if raw:
            print(f"        raw output: {raw[:300]!r}")
    return len(failures)


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else None
    settings = load_settings()
    m = model or settings.ollama_extract_model
    return 1 if asyncio.run(run(m)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
