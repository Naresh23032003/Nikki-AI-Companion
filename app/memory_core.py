"""Memory scoring, consolidation and ranking — pure logic, zero I/O.

This module is deliberately free of ChromaDB, SQLite, Ollama and the network so
that every memory decision is unit-testable without a model or a GPU. The I/O
wiring lives in app/memory.py; the schema lives in app/db.py.

Three things live here:

1. SCORING       importance, confidence and time decay for a memory.
2. CONSOLIDATION what to do when a new fact resembles an existing one.
3. RANKING       fusing several retrieval channels and reranking the result.

Design provenance (see OPEN_SOURCE_COMPONENTS.md):

- The consolidation decision loop adapts Mem0's ADD/UPDATE/DELETE/NOOP pattern,
  but replaces DELETE with SUPERSEDE: a contradicted memory is invalidated and
  linked forward, never destroyed. The previous implementation overwrote the
  older fact in place, which silently lost history whenever the user corrected
  something ("I work at Google" -> "I work at Stripe" are near-identical
  vectors, so correction and duplication hit the same code path).
- Supersession semantics follow Zep/Graphiti's bi-temporal edge model: we track
  when a fact was *recorded* (t_created) separately from when it *stopped being
  true* (t_invalid). Nikki can then answer both "where do I work?" and "where
  did I used to work?" from the same table.
- Rank fusion is Reciprocal Rank Fusion, which combines ranked lists by
  position rather than by score, so the dense (cosine) and lexical (FTS5)
  channels don't need score calibration against each other.

None of those projects are taken as dependencies — see the plan for why.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Sequence

# --------------------------------------------------------------------------
# Tunables. Collected here so behaviour can be adjusted without hunting
# through the logic, and so tests can override them explicitly.
# --------------------------------------------------------------------------

# Cosine similarity bands used by consolidation.
SIM_DISTINCT = 0.72   # below this, an incoming fact is simply new
SIM_RELATED = 0.86    # between DISTINCT and this: related but not the same fact
SIM_SAME_TOPIC = 0.93  # above this: same subject; duplicate, update or contradiction

# Half-lives in days for retrieval decay, per memory kind. "permanent" is
# effectively non-decaying; importance stretches these further (see half_life).
BASE_HALF_LIFE_DAYS = {
    "permanent": 3650.0,
    "recurring": 365.0,
    "event": 45.0,
    "transient": 0.5,
}

# How much each signal contributes to the final rerank score. Relevance
# dominates on purpose: importance must never be able to drag an unrelated
# memory into the prompt on its own.
RERANK_WEIGHTS = {
    "relevance": 0.55,
    "importance": 0.20,
    "confidence": 0.15,
    "recency": 0.10,
}

# A candidate below this fused-relevance floor is dropped no matter how
# important it is. This is the fix for the old unconditional "recent N" union,
# which injected the last few memories into every single prompt whether or not
# they had anything to do with what was said.
#
# NOTE this floor is relative to the retrieved candidate POOL (raw_score/top
# in rerank() below) - the top-ranked candidate is always exactly 1.0 by
# construction, so this floor alone can never produce true abstention when
# nothing is actually relevant; it only ranks among whatever the dense/lexical
# channels handed it. DENSE_SIM_FLOOR below is what makes abstention real.
RELEVANCE_FLOOR = 0.15

# Absolute cosine-similarity floor for the DENSE (vector) retrieval channel,
# applied BEFORE fusion/reranking - see filter_dense_by_similarity(). Verified
# 2026-08-09 against real nomic-embed-text embeddings (HOST_VERIFICATION.md
# §3): a live eval run with only RELEVANCE_FLOOR passed 6/8 scenarios but
# FAILED both abstention cases 0/2 - "what is the capital of Peru" recalled
# "User's sister is named Meera" and "User dislikes coffee" anyway, because
# ANN search always returns its k-nearest neighbours regardless of whether
# any of them are actually similar, and the top one is always relevance=1.0
# under the relative floor above. Measured real cosine similarities: genuinely
# unrelated pairs clustered 0.24-0.38 (a "what is my sister called"-shaped
# query against an unrelated fact); genuinely relevant pairs clustered
# 0.43-0.79. 0.40 sits in the gap with margin on both sides.
DENSE_SIM_FLOOR = 0.40

# Categories that describe stable properties of a person rank above categories
# that describe passing states.
CATEGORY_IMPORTANCE = {
    "personal_info": 0.80,
    "relationship": 0.75,
    "preference": 0.60,
    "plan": 0.55,
    "event": 0.50,
    "emotion": 0.30,
}

KIND_IMPORTANCE = {
    "permanent": 0.15,
    "recurring": 0.10,
    "event": 0.05,
    "transient": -0.15,
}

# Words that signal a durable, identity-level fact rather than a passing one.
_STABILITY_CUES = re.compile(
    r"\b(always|never|every|birthday|born|married|allergic|hates?|loves?|"
    r"favou?rite|named|sister|brother|mother|father|mom|dad|wife|husband|"
    r"daughter|son|grandmother|grandfather|works? at|studies? at|lives? in)\b",
    re.I,
)

# Negation and reversal cues used by structural contradiction detection.
_NEGATIONS = re.compile(
    r"\b(not|no longer|never|isn'?t|aren'?t|doesn'?t|don'?t|didn'?t|"
    r"stopped|quit|left|ex-|former|used to)\b",
    re.I,
)

# Predicates whose object is single-valued for a person: you have one current
# employer, one home city, one birthday. Two different objects for the same
# predicate is a contradiction, not two coexisting facts.
_FUNCTIONAL_PREDICATES = {
    "works_at": re.compile(r"\bworks? (?:at|for)\b", re.I),
    "lives_in": re.compile(r"\blives? in\b", re.I),
    "studies_at": re.compile(r"\bstud(?:ies|ying) at\b", re.I),
    "birthday": re.compile(r"\bbirthday is\b", re.I),
    "name_is": re.compile(r"\bname is\b", re.I),
    "age_is": re.compile(r"\bis \d+ years old\b", re.I),
}

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "at",
    "for", "and", "or", "user", "users", "he", "she", "they", "it", "his",
    "her", "their", "its", "that", "this",
}

# Trailing words that qualify *when* a functional fact holds rather than naming
# a different object. Stripped before comparing two objects for conflict.
_OBJECT_MODIFIERS = {
    "now", "currently", "still", "again", "anymore", "already", "these", "days",
    "today", "recently", "lately", "the", "a", "an", "in", "at", "as", "of",
}


class Consolidation(str, Enum):
    """What to do with an incoming fact given the memories it resembles."""

    ADD = "add"              # genuinely new information
    REINFORCE = "reinforce"  # already known; raise confidence, don't rewrite
    UPDATE = "update"        # same fact, but the new phrasing carries more detail
    SUPERSEDE = "supersede"  # conflicts with a stored fact; invalidate the old one
    NOOP = "noop"            # nothing worth storing


@dataclass
class MemoryRecord:
    """A memory as the ranking layer sees it. Mirrors the `memories` row."""

    id: int
    fact: str
    category: str = "personal_info"
    kind: str = "permanent"
    importance: float = 0.5
    confidence: float = 0.7
    created_at: datetime | None = None
    last_accessed: datetime | None = None
    access_count: int = 0
    event_datetime: datetime | None = None
    valid_until: datetime | None = None
    t_invalid: datetime | None = None       # when the fact stopped being true
    superseded_by: int | None = None
    reinforced_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """A superseded memory stays queryable for history but never ranks."""
        return self.superseded_by is None and self.t_invalid is None


@dataclass
class ConsolidationResult:
    decision: Consolidation
    target_id: int | None = None
    reason: str = ""
    confidence_delta: float = 0.0
    merged_fact: str | None = None


# --------------------------------------------------------------------------
# 1. Scoring
# --------------------------------------------------------------------------


def score_importance(
    fact: str,
    category: str,
    kind: str = "permanent",
    *,
    reinforced_count: int = 0,
    has_entities: bool = False,
) -> float:
    """Heuristic importance in [0, 1].

    Deliberately heuristic rather than a model call: importance is computed on
    every stored fact, and spending an LLM round-trip per fact would put a
    local model on the critical path of the background extraction queue. The
    signals below are cheap and, in practice, separate "user's sister is called
    Meera" from "user had chai this morning" reliably enough to rank with.
    """
    score = CATEGORY_IMPORTANCE.get(category, 0.5)
    score += KIND_IMPORTANCE.get(kind, 0.0)

    if _STABILITY_CUES.search(fact):
        score += 0.12
    if has_entities:
        score += 0.06
    # Each independent restatement is evidence the fact matters, with
    # diminishing returns so a repeated triviality can't outrank a birthday.
    score += min(0.15, 0.05 * reinforced_count)

    # Very short facts are usually fragments; very long ones usually bundle
    # several claims and retrieve poorly.
    words = len(fact.split())
    if words < 4:
        score -= 0.10
    elif words > 40:
        score -= 0.05

    return _clamp(score)


def half_life_days(kind: str, importance: float, access_count: int = 0) -> float:
    """Effective decay half-life for a memory.

    Importance stretches the half-life (something that matters is forgotten
    more slowly) and so does repeated retrieval, which is a crude spaced-
    repetition effect: a memory you keep needing stays available.
    """
    base = BASE_HALF_LIFE_DAYS.get(kind, BASE_HALF_LIFE_DAYS["permanent"])
    importance_factor = 0.5 + importance          # 0.5x .. 1.5x
    access_factor = 1.0 + min(1.0, 0.15 * access_count)  # up to 2x
    return base * importance_factor * access_factor


def recency_weight(
    record: MemoryRecord, now: datetime | None = None
) -> float:
    """Exponential decay in [0, 1] based on time since last contact.

    Uses last_accessed when present, else created_at: a memory that was
    retrieved yesterday is "fresh" even if it was first learned a year ago.
    """
    now = now or _utcnow()
    anchor = record.last_accessed or record.created_at
    if anchor is None:
        return 0.5
    age_days = max(0.0, (now - _aware(anchor)).total_seconds() / 86400.0)
    hl = half_life_days(record.kind, record.importance, record.access_count)
    if hl <= 0:
        return 0.0
    return math.exp(-math.log(2) * age_days / hl)


def adjust_confidence(
    current: float,
    *,
    reinforced: bool = False,
    contradicted: bool = False,
    user_asserted: bool = False,
) -> float:
    """Move a confidence value in response to new evidence.

    A direct user assertion pins confidence high — the user correcting Nikki is
    the strongest signal available and must immediately outrank whatever was
    previously inferred, otherwise corrections take several turns to take hold.
    """
    if user_asserted:
        return 0.95
    value = current
    if reinforced:
        # Asymptotic approach to 1.0 so repetition never manufactures certainty.
        value += (1.0 - value) * 0.35
    if contradicted:
        value *= 0.5
    return _clamp(value, lo=0.05)


# --------------------------------------------------------------------------
# 2. Consolidation
# --------------------------------------------------------------------------


def detect_contradiction(new_fact: str, old_fact: str) -> tuple[bool, str]:
    """Structural contradiction check. Returns (is_contradiction, reason).

    Runs before any LLM adjudication because it is free and catches the common
    cases. Two rules:

    1. Functional predicates ("works at X", "lives in Y") are single-valued.
       Same predicate + different object = contradiction.
    2. Polarity flip: one statement negates the other over a shared subject.

    Returns False for "unknown" rather than guessing — the caller escalates
    ambiguous, highly-similar pairs to the LLM adjudicator.
    """
    for name, pattern in _FUNCTIONAL_PREDICATES.items():
        if pattern.search(new_fact) and pattern.search(old_fact):
            new_obj = _object_after(new_fact, pattern)
            old_obj = _object_after(old_fact, pattern)
            if new_obj and old_obj and _objects_conflict(new_obj, old_obj):
                return True, (
                    f"conflicting {name}: {' '.join(sorted(old_obj))!r} -> "
                    f"{' '.join(sorted(new_obj))!r}"
                )
            return False, f"same {name}"

    new_neg = bool(_NEGATIONS.search(new_fact))
    old_neg = bool(_NEGATIONS.search(old_fact))
    if new_neg != old_neg:
        # Polarity differs; only a contradiction if they're about the same
        # thing, which we approximate with content-word overlap.
        overlap = _content_overlap(new_fact, old_fact)
        if overlap >= 0.5:
            return True, "polarity flip on shared subject"
    return False, ""


def decide_consolidation(
    new_fact: str,
    candidates: Sequence[tuple[MemoryRecord, float]],
    *,
    user_asserted: bool = False,
    sim_distinct: float = SIM_DISTINCT,
    sim_same_topic: float = SIM_SAME_TOPIC,
) -> ConsolidationResult:
    """Decide what to do with `new_fact` given similar existing memories.

    `candidates` is [(record, cosine_similarity)] ordered best-first.

    This replaces the old `_nearest_duplicate` -> `update_memory` path, which
    treated every near-match as a duplicate and overwrote it. The critical
    difference: a contradiction now SUPERSEDES (invalidate old, insert new,
    link them) instead of destroying the previous fact.
    """
    if not new_fact.strip():
        return ConsolidationResult(Consolidation.NOOP, reason="empty fact")

    active = [(r, s) for r, s in candidates if r.is_active]
    if not active:
        return ConsolidationResult(Consolidation.ADD, reason="no active candidates")

    # Structural contradictions are checked across ALL candidates and are NOT
    # gated by the similarity threshold. A functional-predicate conflict
    # ("works at X" vs "works at Y") is stronger evidence of a changed fact
    # than cosine distance is of an unchanged one — embedders routinely score
    # such pairs as merely "related", and gating on similarity let both facts
    # coexist, which is how a companion ends up believing you work at two
    # places at once.
    for record, _sim in active:
        contradicts, why = detect_contradiction(new_fact, record.fact)
        if contradicts:
            return ConsolidationResult(
                Consolidation.SUPERSEDE,
                target_id=record.id,
                reason=why,
                confidence_delta=-0.5,
            )

    best, sim = active[0]

    if sim < sim_distinct:
        return ConsolidationResult(
            Consolidation.ADD, reason=f"nearest similarity {sim:.2f} below distinct threshold"
        )

    if sim >= sim_same_topic:
        normalized_new = _normalize(new_fact)
        normalized_old = _normalize(best.fact)
        if normalized_new == normalized_old:
            return ConsolidationResult(
                Consolidation.REINFORCE,
                target_id=best.id,
                reason="identical restatement",
                confidence_delta=0.1,
            )
        # Same topic, different wording: keep whichever carries more
        # information rather than blindly preferring the newer phrasing.
        if _information_gain(new_fact, best.fact) > 0:
            return ConsolidationResult(
                Consolidation.UPDATE,
                target_id=best.id,
                reason="new phrasing adds detail",
                merged_fact=new_fact,
                confidence_delta=0.05,
            )
        return ConsolidationResult(
            Consolidation.REINFORCE,
            target_id=best.id,
            reason="no new information",
            confidence_delta=0.1,
        )

    # Related but not the same fact — store separately.
    return ConsolidationResult(
        Consolidation.ADD, reason=f"related (sim {sim:.2f}) but distinct"
    )


def needs_llm_adjudication(
    new_fact: str, candidates: Sequence[tuple[MemoryRecord, float]]
) -> bool:
    """True when a pair is similar enough to be risky but not structurally clear.

    Keeps the expensive path rare: the local structural rules settle the common
    cases, and only genuinely ambiguous high-similarity pairs cost a model call.
    """
    if not candidates:
        return False
    best, sim = candidates[0]
    if sim < SIM_RELATED:
        return False
    contradicts, _ = detect_contradiction(new_fact, best.fact)
    if contradicts:
        return False  # already settled
    # High similarity, no structural verdict, and the wording differs enough
    # that a silent overwrite would be a guess.
    return sim >= SIM_RELATED and _normalize(new_fact) != _normalize(best.fact)


# --------------------------------------------------------------------------
# 3. Ranking
# --------------------------------------------------------------------------


def filter_dense_by_similarity(
    ids_and_distances: Sequence[tuple[int, float]],
    *,
    sim_floor: float = DENSE_SIM_FLOOR,
) -> List[int]:
    """Drop ANN neighbours that are not actually similar to the query.

    A vector index returns its k-nearest neighbours unconditionally - if
    nothing in the collection is relevant, it still returns the LEAST
    irrelevant ones. Call this on the raw (id, cosine_distance) pairs a
    ChromaDB query returns (space="cosine", so similarity = 1 - distance)
    BEFORE the ids enter rrf_fuse()/rerank(): rerank()'s RELEVANCE_FLOOR is
    relative to the retrieved pool (the top candidate is always exactly 1.0
    by construction), so it cannot by itself produce true abstention when the
    whole pool is irrelevant. This is what does.
    """
    return [doc_id for doc_id, distance in ids_and_distances
            if (1.0 - distance) >= sim_floor]


def rrf_fuse(
    rankings: Mapping[str, Sequence[int]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> Dict[int, float]:
    """Reciprocal Rank Fusion over several ranked id lists.

    score(d) = sum over channels of weight / (k + rank(d))

    Position-based rather than score-based, so the dense cosine channel and the
    lexical FTS5 channel can be combined without normalising two incompatible
    score distributions against each other.
    """
    weights = weights or {}
    fused: Dict[int, float] = {}
    for channel, ids in rankings.items():
        w = weights.get(channel, 1.0)
        for rank, doc_id in enumerate(ids, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
    return fused


def rerank(
    candidates: Sequence[MemoryRecord],
    fused_scores: Mapping[int, float],
    *,
    now: datetime | None = None,
    weights: Mapping[str, float] | None = None,
    relevance_floor: float = RELEVANCE_FLOOR,
    limit: int | None = None,
) -> List[tuple[MemoryRecord, float]]:
    """Final ordering over fused candidates.

    Combines normalised fused relevance with importance, confidence and recency
    decay. Anything below `relevance_floor` is dropped regardless of its other
    scores — that floor is what stops unrelated-but-recent or
    unrelated-but-important memories being injected into every prompt.
    """
    now = now or _utcnow()
    w = {**RERANK_WEIGHTS, **(weights or {})}

    active = [c for c in candidates if c.is_active]
    if not active:
        return []

    raw = [fused_scores.get(c.id, 0.0) for c in active]
    top = max(raw) if raw else 0.0

    scored: List[tuple[MemoryRecord, float]] = []
    for record, raw_score in zip(active, raw):
        relevance = (raw_score / top) if top > 0 else 0.0
        if relevance < relevance_floor:
            continue
        score = (
            w["relevance"] * relevance
            + w["importance"] * record.importance
            + w["confidence"] * record.confidence
            + w["recency"] * recency_weight(record, now)
        )
        scored.append((record, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:limit] if limit else scored


def is_temporally_current(
    record: MemoryRecord, now: datetime | None = None, event_grace_days: int = 7
) -> bool:
    """Whether a memory should be eligible for injection right now.

    Preserves the existing product behaviour: an event stays current for a week
    after it happens so Nikki can ask how it went, then ages out.
    """
    now = now or _utcnow()
    if not record.is_active:
        return False

    kind = (record.kind or "permanent").lower()
    if kind == "permanent":
        return True

    if record.valid_until and now > _aware(record.valid_until):
        # An event explicitly closed out before its date (marked done) is done.
        if kind != "event" or not record.event_datetime:
            return False
        if _aware(record.valid_until) < _aware(record.event_datetime):
            return False

    if kind == "transient":
        return record.valid_until is not None and now <= _aware(record.valid_until)

    if kind == "event":
        if record.event_datetime is None:
            return True
        return now <= _aware(record.event_datetime) + timedelta(days=event_grace_days)

    return True


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    """Treat naive timestamps as UTC so comparisons never raise."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def _content_words(text: str) -> set[str]:
    return {w for w in _normalize(text).split() if w and w not in _STOPWORDS}


def _content_overlap(a: str, b: str) -> float:
    """Jaccard overlap of content words, used to decide whether two statements
    are even about the same thing before calling a polarity flip a conflict."""
    wa, wb = _content_words(a), _content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _object_after(text: str, pattern: re.Pattern[str]) -> set[str]:
    """Token set for the object of a functional predicate ("works at" -> {stripe}).

    Trailing modifiers are stripped: "works at Google now" and "works at Google"
    name the same employer, and treating them as different objects produced a
    false contradiction that superseded a perfectly correct memory.
    """
    match = pattern.search(text)
    if not match:
        return set()
    tail = text[match.end():].strip()
    tail = re.split(r"[,.;]| and | but | since | as of | because ", tail)[0]
    return {w for w in _normalize(tail).split() if w and w not in _OBJECT_MODIFIERS}


def _objects_conflict(new_obj: set[str], old_obj: set[str]) -> bool:
    """Whether two objects of the same functional predicate genuinely disagree.

    Containment is refinement, not conflict: "Google" -> "Google India" is the
    user being more specific about the same employer. Disagreement requires
    neither object to contain the other.
    """
    if not new_obj or not old_obj:
        return False
    if new_obj == old_obj:
        return False
    return not (new_obj <= old_obj or old_obj <= new_obj)


def _information_gain(new_fact: str, old_fact: str) -> int:
    """Content words the new phrasing adds that the old one lacked."""
    return len(_content_words(new_fact) - _content_words(old_fact))


def build_record(row: Mapping[str, Any]) -> MemoryRecord:
    """Adapt a sqlite3.Row / dict from the `memories` table into a MemoryRecord."""
    return MemoryRecord(
        id=int(row["id"]),
        fact=str(row["fact"]),
        category=str(row["category"] or "personal_info"),
        kind=str(row["kind"] or "permanent").lower(),
        importance=_coerce_float(row, "importance", 0.5),
        confidence=_coerce_float(row, "confidence", 0.7),
        created_at=parse_dt(_get(row, "created_at")),
        last_accessed=parse_dt(_get(row, "last_accessed")),
        access_count=int(_get(row, "access_count") or 0),
        event_datetime=parse_dt(_get(row, "event_datetime")),
        valid_until=parse_dt(_get(row, "valid_until")),
        t_invalid=parse_dt(_get(row, "t_invalid")),
        superseded_by=_coerce_int(_get(row, "superseded_by")),
        reinforced_count=int(_get(row, "reinforced_count") or 0),
    )


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return _aware(value)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return _aware(datetime.fromisoformat(text))
    except ValueError:
        return None


def _get(row: Mapping[str, Any], key: str) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def _coerce_float(row: Mapping[str, Any], key: str, default: float) -> float:
    value = _get(row, key)
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
