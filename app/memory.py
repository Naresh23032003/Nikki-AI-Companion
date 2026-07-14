"""Long-term memory: fact extraction, vector storage, and retrieval.

Pipeline:
  1. After each user+assistant exchange, `extract_and_store` asks the LLM to
     pull durable facts out of the exchange (strict JSON) and stores each one.
  2. Storing a fact writes a row to SQLite (`memories` table) and an embedding
     to a persistent ChromaDB collection. Near-duplicates (cosine > threshold)
     update the existing memory instead of inserting a new one.
  3. `retrieve_memories` embeds the incoming user message, pulls the top-k most
     relevant memories from ChromaDB, always unions in the N most recent
     memories, and returns them for injection into the system prompt.

Vectors come from Ollama's embedding model (nomic-embed-text); Chroma is used
purely as an ANN index over vectors we compute ourselves.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings
from app.db import Database
from app.llm import OllamaClient

logger = logging.getLogger("companion.memory")

VALID_CATEGORIES = {
    "personal_info",
    "preference",
    "event",
    "emotion",
    "plan",
    "relationship",
}

_EXTRACTION_SYSTEM = """\
You extract durable, long-term facts about the USER from one exchange between \
the user and their companion. Be STRICT and conservative - it is much better to \
record nothing than to record something wrong.

HARD RULES:
- Only record facts the USER explicitly stated about themselves or their life.
- NEVER infer, guess, assume, or extrapolate. If it isn't clearly stated, skip it.
- NEVER attribute the COMPANION's life to the user. If the companion said "I
  live near the coast", that is NOT a fact about the user. If the user only
  asked a question ("where do you live?"), there is no fact - return [].
- NEVER record that the user said yes/no/ok, agreed, greeted, or asked for
  something in the moment (stickers, photos). Those are not memories.
- Possessives matter: "your cat", "your job" = the COMPANION's - never store the
  companion's pets/life as the user's. Only "my cat", "my job" are the user's.
- For `event_datetime`: resolve relative times ("at 12", "tomorrow", "tonight")
  against the "Current local datetime" line provided, and output full ISO-8601
  WITH the same timezone offset. "exam at 12" = today 12:00 local unless another
  day is stated. Never invent a date that wasn't implied.
- If the user DENIES or corrects something ("I am not a student"), never store
  the denied thing as a fact. Store the correction instead, if it's durable.
- IGNORE everything the companion said - especially the companion's questions,
  guesses, suggestions, or offers (e.g. if the companion asks "want me to bring
  leftovers?", that is NOT a fact about the user).
- IGNORE small talk, greetings, acknowledgements ("hi", "good", "ok", "lol"),
  and anything transient (fleeting moods, one-off logistics for right now).
- Only keep things worth remembering for weeks/months: names, stable
  preferences, personal details, real events, meaningful plans, relationships.
- Write each fact as a short standalone third-person statement, grounded in the
  user's own words (e.g. "User's birthday is March 3rd", "User dislikes coffee").
- One category per fact from exactly: personal_info, preference, event, emotion,
  plan, relationship.
- When the user mentions a person, place, thing, event, pet, or org, include it
  in `entities` and any clear relationship in `relations`.

Also rate `affection_delta`: how much this single exchange should move her
affection for the user, as an integer from -2 to +2:
- 0  = ordinary small talk / logistics - THE DEFAULT for most exchanges.
- +1 = genuinely warm, personal, vulnerable, or supportive moment.
- +2 = exceptional: a deep conversation, real openness, something that matters.
- -1 = cold, dismissive, or hurtful toward her.
- -2 = genuinely mean.
Be stingy: casual pleasant chat is 0, not +1.

Respond with STRICT JSON only:
{"memories": [{"fact": "...", "category": "...", "kind": "permanent|event|recurring|transient", "event_datetime": "ISO-8601-or-null", "recurrence_rule": "... or null", "valid_from": "ISO-8601-or-null", "valid_until": "ISO-8601-or-null"}], "affection_delta": 0, "entities": [{"name": "...", "type": "person", "notes": "..."}], "relations": [{"source": "...", "relation": "works_at", "target": "...", "confidence": 0.9}]}
When in doubt, or for ordinary chit-chat: {"memories": [], "affection_delta": 0, "entities": [], "relations": []}"""


class MemoryStore:
    def __init__(self, db: Database, llm: OllamaClient, settings: Settings):
        self.db = db
        self.llm = llm
        self.settings = settings
        self.dedup_threshold = settings.memory_dedup_threshold
        # Set by main after construction; receives affection deltas per exchange.
        self.relationship = None
        # Set by main after construction; receives affection deltas so
        # conversation warmth nudges her intra-day mood (see app/dayseed.py).
        self.daylife = None
        # Entity names that must never enter the graph (persona's own name is
        # added by main at startup/persona switch - her cat isn't YOUR cat).
        self.blocked_names: set[str] = set()

        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Cosine space so distance = 1 - cosine_similarity.
        self._collection = self._client.get_or_create_collection(
            name=settings.memory_collection,
            metadata={"hnsw:space": "cosine"},
        )

    async def add_fact(
        self,
        fact: str,
        category: str,
        kind: str | None = None,
        event_datetime: str | None = None,
        recurrence_rule: str | None = None,
        valid_from: str | None = None,
        valid_until: str | None = None,
        source: str | None = None,
    ) -> int | None:
        """Store one fact (with dedup). Returns the memory id, or None on error.

        If an existing memory is more similar than the dedup threshold, that
        memory is updated in place and its id returned.
        """
        fact = fact.strip()
        if not fact:
            return None
        if category not in VALID_CATEGORIES:
            category = "personal_info"

        # Small models emit junk strings instead of JSON null - normalize them
        # (a literal "now"/"null" valid_until made facts silently unretrievable).
        def _clean_dt(v):
            if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "now", "n/a", "unknown"}:
                return None
            return v
        event_datetime = _clean_dt(event_datetime)
        valid_from = _clean_dt(valid_from)
        valid_until = _clean_dt(valid_until)
        recurrence_rule = _clean_dt(recurrence_rule)

        try:
            embedding = await self.llm.embed(fact)
        except Exception as e:  # noqa: BLE001 - never let memory break the app
            logger.warning("Embedding failed for fact %r: %s", fact, e)
            return None

        # --- Deduplicate against the nearest existing memory ---
        existing_id = self._nearest_duplicate(embedding)
        if existing_id is not None:
            self.db.update_memory(
                existing_id,
                fact,
                category,
                kind=kind,
                event_datetime=event_datetime,
                recurrence_rule=recurrence_rule,
                valid_from=valid_from,
                valid_until=valid_until,
                source=source,
            )
            self._collection.update(
                ids=[str(existing_id)],
                embeddings=[embedding],
                documents=[fact],
                metadatas=[{"category": category, "kind": kind or "permanent"}],
            )
            logger.info("Updated existing memory #%s (dedup): %r", existing_id, fact)
            return existing_id

        # --- Otherwise insert a new memory ---
        memory_id = self.db.add_memory(
            fact,
            category,
            kind=kind,
            event_datetime=event_datetime,
            recurrence_rule=recurrence_rule,
            valid_from=valid_from,
            valid_until=valid_until,
            source=source,
        )
        self._collection.add(
            ids=[str(memory_id)],
            embeddings=[embedding],
            documents=[fact],
            metadatas=[{"category": category}],
        )
        logger.info("Stored new memory #%s [%s]: %r", memory_id, category, fact)
        # A freshly-stored dated event gets a care check-in scheduled around
        # it (see _schedule_event_followup) - read the row back since kind/
        # event_datetime may have been normalized by db.add_memory.
        row = self.db.get_memory(memory_id)
        if row and (row.get("kind") or "").lower() == "event":
            self._schedule_event_followup(memory_id, fact, row.get("event_datetime"))
        return memory_id

    def _schedule_event_followup(self, memory_id: int, fact: str,
                                 event_datetime: str | None) -> None:
        """Schedule a before-event encouragement + after-event 'how did it go'
        check-in (app/db.py event_followups table; polled in main.py like
        reminders). Skipped for events with no resolved datetime, or whose
        follow-up window has already passed (e.g. a past event just logged)."""
        dt = self._parse_datetime(event_datetime)
        if dt is None:
            return
        now = datetime.now(timezone.utc)
        encouragement_at = dt - timedelta(hours=3)
        followup_at = dt + timedelta(hours=3)
        if followup_at <= now:
            return
        enc_iso = encouragement_at.isoformat() if encouragement_at > now else None
        try:
            self.db.add_event_followup(
                memory_id, fact, dt.isoformat(), enc_iso, followup_at.isoformat(),
                session_id=self.settings.wa_session_id,
            )
            logger.info("Scheduled event follow-up for memory #%s (%s)", memory_id, fact)
        except Exception as e:  # noqa: BLE001 - never let this break fact storage
            logger.warning("Event follow-up scheduling failed for memory #%s: %s",
                           memory_id, e)

    def _nearest_duplicate(self, embedding: List[float]) -> int | None:
        """Return the id of an existing memory within the dedup threshold, else None."""
        if self._collection.count() == 0:
            return None
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["distances"],
        )
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]
        if not ids:
            return None
        similarity = 1.0 - float(distances[0])  # cosine space
        if similarity >= self.dedup_threshold:
            return int(ids[0])
        return None

    def sync_from_row(self, memory_id: int) -> None:
        """(Re)embed and index an existing SQLite memory row.

        Used by the manual POST /memories endpoint, which writes the row first.
        """
        row = self.db.get_memory(memory_id)
        if not row:
            return
        try:
            embedding = self._embed_sync(row["fact"])
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to embed manual memory #%s: %s", memory_id, e)
            return
        self._collection.upsert(
            ids=[str(memory_id)],
            embeddings=[embedding],
            documents=[row["fact"]],
            metadatas=[{"category": row["category"]}],
        )

    def remove(self, memory_id: int) -> None:
        """Drop a memory's vector from Chroma (SQLite row deleted separately)."""
        try:
            self._collection.delete(ids=[str(memory_id)])
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to delete vector #%s: %s", memory_id, e)

    # ------------------------------------------------------------------
    # Retrieval  (this is the hook wired into the system prompt)
    # ------------------------------------------------------------------
    def _name_match_score(self, query: str, entity_name: str) -> float:
        q = re.sub(r"[^a-z0-9]+", "", query.lower())
        e = re.sub(r"[^a-z0-9]+", "", entity_name.lower())
        if not q or not e:
            return 0.0
        if e in q:
            return 1.0
        if q in e:
            return 0.95
        return 0.0

    def retrieve_graph_facts(self, query: str, limit: int = 8) -> List[str]:
        entities = self.db.list_entities()
        matched: List[tuple[float, int, str]] = []
        for entity in entities:
            norm = entity.get("normalized_name") or ""
            if norm in self._BLOCKED_ENTITY_NAMES or norm in self.blocked_names:
                continue
            score = self._name_match_score(query, entity["name"])
            if score > 0.0:
                matched.append((score, int(entity["id"]), entity["name"]))
        if not matched:
            return []
        matched.sort(reverse=True)
        facts: List[str] = []
        for _, entity_id, entity_name in matched[:3]:
            relations = self.db.get_relations_for_entity(entity_id, active_only=True)
            for rel in relations[: max(1, limit // 3)]:
                source = self.db.get_entity(int(rel["source_id"])) or {}
                target = self.db.get_entity(int(rel["target_id"])) or {}
                source_name = source.get("name", "unknown")
                target_name = target.get("name", "unknown")
                if int(rel["source_id"]) == entity_id:
                    facts.append(f"{source_name} {rel['relation']} {target_name}")
                else:
                    facts.append(f"{target_name} {rel['relation']} {source_name}")
                if len(facts) >= limit:
                    break
            if len(facts) >= limit:
                break
        return facts[:limit]

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                dt = datetime.fromisoformat(text)
                # Naive timestamps are assumed UTC so comparisons never crash.
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        return None

    @classmethod
    def _memory_is_current(cls, row: Dict[str, Any]) -> bool:
        kind = (row.get("kind") or "permanent").strip().lower()
        if kind == "permanent":
            return True
        if kind == "transient":
            valid_until = cls._parse_datetime(row.get("valid_until"))
            if valid_until is None:
                return False
            return datetime.now(timezone.utc) <= valid_until
        if kind == "event":
            # A completed/dismissed event (valid_until in the past) stops being
            # injected immediately - this is what "mark completed" sets.
            valid_until = cls._parse_datetime(row.get("valid_until"))
            event_dt = cls._parse_datetime(row.get("event_datetime"))
            if valid_until is not None and datetime.now(timezone.utc) > valid_until:
                if event_dt is None or valid_until < event_dt:
                    return False
            if event_dt is None:
                return True
            # Stay current for 7 days AFTER the event so she can reference it in
            # past tense ("how did the exam go") - then it ages out of injection.
            return datetime.now(timezone.utc) <= event_dt + timedelta(days=7)
        if kind == "recurring":
            valid_until = cls._parse_datetime(row.get("valid_until"))
            if valid_until is not None and datetime.now(timezone.utc) > valid_until:
                return False
            return True
        return True

    @classmethod
    def _format_memory_for_prompt(cls, row: Dict[str, Any]) -> str:
        kind = (row.get("kind") or "permanent").strip().lower()
        fact = str(row.get("fact", "")).strip()
        if kind == "event":
            event_dt = cls._parse_datetime(row.get("event_datetime"))
            if event_dt is not None:
                local = event_dt.astimezone()  # show HER the user's local time
                if datetime.now(timezone.utc) > event_dt:
                    return f"PAST event (already happened): {fact} (was on {local:%b %d})"
                return f"UPCOMING event: {fact} (on {local:%b %d at %I:%M %p})"
        if kind == "recurring":
            rule = (row.get("recurrence_rule") or "").strip()
            return f"Recurring: {fact}" + (f" ({rule})" if rule else "")
        if kind == "transient":
            return f"Right now (temporary): {fact}"
        return fact

    async def retrieve_memories(self, query: str, k: int | None = None) -> List[str]:
        """Return facts relevant to `query`, plus the most recent memories.

        Embeds the query, fetches the top-k nearest memories from ChromaDB, and
        always unions in the N most-recently-created memories. Returns a list of
        plain fact strings ready for the system prompt.
        """
        k = k or self.settings.memory_top_k
        start = time.perf_counter()

        ordered: List[tuple[int, str]] = []
        seen: set[int] = set()

        # --- Semantic top-k, gated by a relevance threshold ---
        # Only inject memories that are actually related to what was said, so
        # she doesn't bring up unrelated facts at random.
        threshold = self.settings.memory_retrieval_threshold
        try:
            if self._collection.count() > 0:
                # Hard cap on the embed call: when Ollama is busy with a
                # background extraction, this can otherwise queue for seconds
                # and stall the reply. Degrade to recent+graph facts instead.
                embedding = await asyncio.wait_for(self.llm.embed(query), timeout=2.5)
                res = self._collection.query(
                    query_embeddings=[embedding],
                    n_results=k,
                    include=["distances"],
                )
                ids = res.get("ids", [[]])[0]
                dists = res.get("distances", [[]])[0]
                hit_ids = [
                    int(mid)
                    for mid, dist in zip(ids, dists)
                    if (1.0 - float(dist)) >= threshold  # cosine space
                ]
                rows = {r["id"]: r for r in self.db.get_memories_by_ids(hit_ids)}
                for mid_int in hit_ids:  # preserve similarity order
                    row = rows.get(mid_int)
                    if row and self._memory_is_current(row) and mid_int not in seen:
                        seen.add(mid_int)
                        ordered.append((mid_int, self._format_memory_for_prompt(row)))
        except asyncio.TimeoutError:
            logger.warning("Memory retrieval: embed timed out - semantic search skipped")
        except Exception as e:  # noqa: BLE001 - retrieval must never break chat
            logger.warning("Memory retrieval failed: %s", e)

        # --- Include a few most-recent memories (freshly-learned context) ---
        for row in self.db.get_recent_memories(self.settings.memory_recent_count):
            if row["id"] not in seen and self._memory_is_current(row):
                seen.add(row["id"])
                ordered.append((row["id"], self._format_memory_for_prompt(row)))

        graph_facts = self.retrieve_graph_facts(query)
        if graph_facts:
            ordered.append((-1, "What you know about the people mentioned: " + "; ".join(graph_facts)))

        # --- Book-keeping: mark these as accessed ---
        self.db.touch_memories([mid for mid, _ in ordered if mid >= 0])

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        status = "ok" if elapsed_ms < 200 else "SLOW"
        logger.info(
            "retrieve_memories: %d memories in %.1fms [%s]",
            len(ordered),
            elapsed_ms,
            status,
        )

        return [fact for _, fact in ordered]

    # ------------------------------------------------------------------
    # Fact extraction (runs as a FastAPI BackgroundTask after each exchange)
    # ------------------------------------------------------------------
    # Generic/meta names that must never become graph entities.
    _BLOCKED_ENTITY_NAMES = {
        "user", "companion", "assistant", "friend", "person", "someone",
        "me", "you", "her", "him", "them", "cat", "dog", "pet",
        "exam", "test", "work", "home", "today", "unknown", "none", "n/a",
    }

    def _entity_ok(self, name: str | None, user_message: str) -> bool:
        """An entity is kept only if it's a real name the USER actually typed.

        Grounding in the user's own words is what stops the graph filling up
        with the companion's hallucinations (e.g. a cat name SHE invented) and
        her own persona (her name, her pet) becoming the user's entities.
        """
        n = (name or "").strip()
        if len(n) < 3:
            return False
        norm = re.sub(r"[^a-z0-9]+", "", n.lower())
        if not norm or norm in self._BLOCKED_ENTITY_NAMES or norm in self.blocked_names:
            return False
        return n.lower() in user_message.lower()

    async def _store_graph_triples(self, raw: str, user_message: str) -> None:
        entities = self._parse_entities(raw)
        relations = self._parse_relations(raw)
        for item in entities:
            name = item.get("name", "")
            if not self._entity_ok(name, user_message):
                logger.info("Rejected ungrounded entity: %r", name)
                continue
            self.db.upsert_entity(name, item.get("type", "thing"), item.get("notes"))
        for item in relations:
            source_name = item.get("source")
            target_name = item.get("target")
            # Both endpoints must be grounded, real names.
            if not (self._entity_ok(source_name, user_message)
                    and self._entity_ok(target_name, user_message)):
                continue
            source_id = self.db.upsert_entity(source_name, "thing")
            target_id = self.db.upsert_entity(target_name, "thing")
            self.db.upsert_relation(
                source_id,
                item.get("relation", "knows"),
                target_id,
                float(item.get("confidence", 0.5) or 0.5),
            )

    async def extract_and_store(
        self,
        user_message: str,
        assistant_message: str,
        source: str | None = None,
    ) -> None:
        """Extract durable facts from one exchange and store them."""
        # Pre-gate: don't even ask the LLM about trivial messages ("yup", "ok",
        # "hello") - small models hallucinate facts rather than return empty.
        if not self._worth_extracting(user_message):
            logger.debug("Extraction skipped (trivial message): %r", user_message)
            return

        # The model needs a clock to resolve relative times ("exam at 12",
        # "tomorrow") into real datetimes - without it, dates are hallucinated.
        now_local = datetime.now().astimezone()
        exchange = (
            f"Current local datetime: {now_local.isoformat()} ({now_local:%A})\n"
            f"User said: {user_message}\n"
            f"Companion replied: {assistant_message}"
        )
        try:
            raw = await self.llm.chat(
                messages=[
                    {"role": "system", "content": _EXTRACTION_SYSTEM},
                    {"role": "user", "content": exchange},
                ],
                format="json",
                # Low temperature for stable, faithful extraction.
                options={"temperature": 0.1},
                # Dedicated structured-output model (config: ollama.extract_model).
                model=self.settings.ollama_extract_model,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Fact extraction LLM call failed: %s", e)
            return

        facts = [f for f in self._parse_facts(raw) if not self._is_junk_fact(f["fact"])]
        for item in facts:
            await self.add_fact(
                item["fact"],
                item["category"],
                kind=item.get("kind"),
                event_datetime=item.get("event_datetime"),
                recurrence_rule=item.get("recurrence_rule"),
                valid_from=item.get("valid_from"),
                valid_until=item.get("valid_until"),
                source=source or "webapp_chat",
            )
        await self._store_graph_triples(raw, user_message)

        # Relationship progression: apply this exchange's affection delta.
        if self.relationship is not None:
            delta = self._parse_affection_delta(raw)
            try:
                await self.relationship.apply_exchange(delta, meaningful=bool(facts),
                                                        trigger_text=user_message)
            except Exception as e:  # noqa: BLE001 - never let this break extraction
                logger.warning("Affection update failed: %s", e)
            # Same signal nudges her intra-day mood (see DayLife.apply_drift) -
            # this was computed but never wired anywhere, so her mood was
            # frozen at the morning seed all day regardless of how the
            # conversation actually went.
            if self.daylife is not None and delta:
                try:
                    self.daylife.apply_drift(delta)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Day mood drift failed: %s", e)

    # Acknowledgements/greetings that carry no memorable information.
    _TRIVIAL = re.compile(
        r"^(yup|yes|yeah( sure)?|no|nope|ok(ay)?|ohk|sure|hi|hello|hey|bye|"
        r"bye-?bye|good|fine|nice|cool|lol|haha+|hmm+|thanks|thank you|"
        r"what|why|k|great|same|me too|love you|miss you)[\s.!?,]*$",
        re.IGNORECASE,
    )

    @classmethod
    def _worth_extracting(cls, user_message: str) -> bool:
        """Cheap gate: only run extraction on messages that could hold a fact."""
        text = user_message.strip()
        if len(text) < 15 or len(text.split()) < 4:
            return False
        # Short questions carry no durable facts - extracting from them is how
        # "what is my cat name?" became "User has a cat". Long mixed messages
        # (question + statements) still pass.
        if text.endswith("?") and len(text.split()) < 12:
            return False
        return not cls._TRIVIAL.match(text)

    # Patterns of "facts" that are really just conversation mechanics.
    _JUNK_FACT = re.compile(
        r"^user('s)?\s+(said|says|asked|asks|agree[sd]?|responded|replied|"
        r"confirmed|acknowledged|greets?|greeted|thinks the same|is good|"
        r"is fine|is okay|is here|is awake|is asleep|is back|came back|"
        r"recalls?|remembers?|acknowledges?|is unsure|is confused|"
        r"gets confused|is trying to|apologi[zs]e[sd]?|"
        r"expresse[sd]?|hopes?|wishes|worrie[sd]?|is concerned|"
        r"do(es)? not know (their|his|her) own name|"
        r"do(es)? not have a (stated|known)|"
        r"says? (goodbye|hello|hi|yes|no))\b"
        r"|(\bsticker\b)|(\bcompanion\b)|(\bassistant\b)|(\bis a person\b)"
        r"|^user (wants|would like) (a|to see|to get) ",
        re.IGNORECASE,
    )

    @classmethod
    def _is_junk_fact(cls, fact: str) -> bool:
        """Reject conversation-mechanics 'facts' and companion-referencing ones."""
        f = fact.strip()
        if len(f.split()) < 3:
            return True
        if cls._JUNK_FACT.search(f):
            logger.info("Rejected junk fact: %r", f)
            return True
        return False

    @staticmethod
    def _parse_affection_delta(raw: str) -> int:
        """Pull affection_delta out of the extractor JSON; 0 on any problem."""
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return max(-2, min(2, int(data.get("affection_delta", 0))))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return 0

    @staticmethod
    def _parse_entities(raw: str) -> List[Dict[str, str]]:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(data, dict):
            return []
        entities = data.get("entities", [])
        if not isinstance(entities, list):
            return []
        out: List[Dict[str, str]] = []
        for item in entities:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                if name:
                    out.append(
                        {
                            "name": name,
                            "type": str(item.get("type", "thing") or "thing"),
                            "notes": str(item.get("notes", "") or ""),
                        }
                    )
        return out

    @staticmethod
    def _parse_relations(raw: str) -> List[Dict[str, Any]]:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(data, dict):
            return []
        relations = data.get("relations", [])
        if not isinstance(relations, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in relations:
            if isinstance(item, dict):
                source = str(item.get("source", "")).strip()
                target = str(item.get("target", "")).strip()
                relation = str(item.get("relation", "knows") or "knows").strip()
                if source and target and relation:
                    try:
                        confidence = float(item.get("confidence", 0.5) or 0.5)
                    except (TypeError, ValueError):
                        confidence = 0.5
                    out.append(
                        {
                            "source": source,
                            "relation": relation,
                            "target": target,
                            "confidence": confidence,
                        }
                    )
        return out

    @staticmethod
    def _parse_facts(raw: str) -> List[Dict[str, Any]]:
        """Robustly parse the extractor's JSON into a list of {fact, category, ...}."""
        try:
            data: Any = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Extractor returned non-JSON: %r", raw[:200])
            return []

        # Accept either a bare list or an object wrapping one.
        if isinstance(data, dict):
            items = None
            for key in ("memories", "facts", "items"):
                if isinstance(data.get(key), list):
                    items = data[key]
                    break
            if items is None:
                # Fall back to the first list-valued field.
                items = next(
                    (v for v in data.values() if isinstance(v, list)), []
                )
        elif isinstance(data, list):
            items = data
        else:
            items = []

        result: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            fact = str(it.get("fact", "")).strip()
            category = str(it.get("category", "personal_info")).strip()
            if not fact:
                continue
            kind = str(it.get("kind", "") or "").strip().lower()
            if kind not in {"permanent", "event", "recurring", "transient"}:
                kind = "permanent"
            event_datetime = it.get("event_datetime") or None
            recurrence_rule = it.get("recurrence_rule") or None
            valid_until = it.get("valid_until") or None
            if kind == "recurring" and not (recurrence_rule or "").strip():
                # The extractor tags one-off items 'recurring' surprisingly
                # often ("User needs to call mom") - and recurring memories
                # without a valid_until never expire, so a finished errand
                # kept resurfacing in her replies for days. No recurrence
                # rule => it is not recurring: make it an event that ages out.
                kind = "event"
                if not event_datetime and not valid_until:
                    valid_until = (datetime.now(timezone.utc)
                                   + timedelta(days=3)).isoformat()
            result.append(
                {
                    "fact": fact,
                    "category": category,
                    "kind": kind,
                    "event_datetime": event_datetime,
                    "recurrence_rule": recurrence_rule,
                    "valid_from": it.get("valid_from") or None,
                    "valid_until": valid_until,
                }
            )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _embed_sync(self, text: str) -> List[float]:
        """Synchronous embedding via httpx for the (sync) manual-add path."""
        import httpx

        resp = httpx.post(
            f"{self.llm.base_url}/api/embeddings",
            json={"model": self.llm.embed_model, "prompt": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
