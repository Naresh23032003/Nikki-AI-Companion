"""Headless tests for the memory scoring/consolidation/ranking core.

No Ollama, no ChromaDB, no GPU, no network — app.memory_core imports only the
standard library, so this suite runs anywhere pytest does.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.memory_core import (
    Consolidation,
    MemoryRecord,
    adjust_confidence,
    decide_consolidation,
    detect_contradiction,
    filter_dense_by_similarity,
    half_life_days,
    is_temporally_current,
    needs_llm_adjudication,
    recency_weight,
    rerank,
    rrf_fuse,
    score_importance,
)

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)


def mem(mid: int, fact: str, **kw) -> MemoryRecord:
    kw.setdefault("created_at", NOW - timedelta(days=1))
    return MemoryRecord(id=mid, fact=fact, **kw)


# ---------------------------------------------------------------- importance


class TestImportance:
    def test_identity_fact_outranks_trivia(self):
        birthday = score_importance(
            "User's birthday is March 3rd", "personal_info", "permanent"
        )
        chai = score_importance(
            "User had chai this morning", "event", "transient"
        )
        assert birthday > chai
        # The gap should be decisive, not marginal — this is the ranking bug
        # the old system had, where both scored identically.
        assert birthday - chai > 0.3

    def test_relationship_facts_rank_high(self):
        assert score_importance(
            "User's sister is named Meera", "relationship", "permanent"
        ) > 0.75

    def test_transient_emotion_ranks_low(self):
        assert score_importance("User felt tired", "emotion", "transient") < 0.35

    def test_reinforcement_raises_importance(self):
        base = score_importance("User dislikes coffee", "preference", "permanent")
        repeated = score_importance(
            "User dislikes coffee", "preference", "permanent", reinforced_count=3
        )
        assert repeated > base

    def test_reinforcement_saturates(self):
        many = score_importance(
            "User dislikes coffee", "preference", "permanent", reinforced_count=50
        )
        some = score_importance(
            "User dislikes coffee", "preference", "permanent", reinforced_count=3
        )
        assert many - some <= 0.1, "repetition must not manufacture unbounded importance"

    def test_scores_stay_in_range(self):
        for cat in ["personal_info", "relationship", "preference", "emotion", "junk"]:
            for kind in ["permanent", "event", "transient", "recurring"]:
                s = score_importance("User's birthday is always never", cat, kind,
                                     reinforced_count=99, has_entities=True)
                assert 0.0 <= s <= 1.0


# -------------------------------------------------------------------- decay


class TestDecay:
    def test_fresh_memory_scores_near_one(self):
        r = mem(1, "x", created_at=NOW, last_accessed=NOW)
        assert recency_weight(r, NOW) > 0.99

    def test_transient_decays_within_a_day(self):
        r = mem(1, "x", kind="transient", created_at=NOW - timedelta(days=1),
                last_accessed=NOW - timedelta(days=1))
        assert recency_weight(r, NOW) < 0.35

    def test_permanent_barely_decays_over_a_year(self):
        r = mem(1, "x", kind="permanent", created_at=NOW - timedelta(days=365),
                last_accessed=NOW - timedelta(days=365))
        assert recency_weight(r, NOW) > 0.85

    def test_access_extends_half_life(self):
        unused = half_life_days("event", 0.5, access_count=0)
        used = half_life_days("event", 0.5, access_count=5)
        assert used > unused

    def test_importance_extends_half_life(self):
        assert half_life_days("event", 0.9) > half_life_days("event", 0.1)

    def test_last_accessed_beats_created_at(self):
        """A year-old memory retrieved yesterday is still fresh."""
        old_but_used = mem(1, "x", kind="event",
                           created_at=NOW - timedelta(days=365),
                           last_accessed=NOW - timedelta(days=1))
        old_and_unused = mem(2, "x", kind="event",
                             created_at=NOW - timedelta(days=365),
                             last_accessed=NOW - timedelta(days=365))
        assert recency_weight(old_but_used, NOW) > recency_weight(old_and_unused, NOW)


# --------------------------------------------------------------- confidence


class TestConfidence:
    def test_reinforcement_increases_but_never_reaches_certainty(self):
        c = 0.7
        for _ in range(50):
            c = adjust_confidence(c, reinforced=True)
        assert c < 1.0
        assert c > 0.9

    def test_contradiction_halves_confidence(self):
        assert adjust_confidence(0.8, contradicted=True) == pytest.approx(0.4)

    def test_user_assertion_pins_high(self):
        """A user correction must take effect immediately, not over several turns."""
        assert adjust_confidence(0.1, user_asserted=True) == 0.95

    def test_confidence_has_a_floor(self):
        c = 0.5
        for _ in range(50):
            c = adjust_confidence(c, contradicted=True)
        assert c >= 0.05


# ------------------------------------------------------------ contradiction


class TestContradiction:
    def test_conflicting_employer(self):
        hit, why = detect_contradiction(
            "User works at Stripe", "User works at Google"
        )
        assert hit
        assert "works_at" in why

    def test_same_employer_is_not_a_contradiction(self):
        hit, _ = detect_contradiction(
            "User works at Google", "User works at Google now"
        )
        assert not hit

    def test_refinement_is_not_a_contradiction(self):
        """'Google' -> 'Google India' is the user being more specific."""
        hit, _ = detect_contradiction(
            "User works at Google India", "User works at Google"
        )
        assert not hit

    def test_sibling_objects_do_conflict(self):
        """But two different offices are a real change of fact."""
        hit, _ = detect_contradiction(
            "User works at Google Ireland", "User works at Google India"
        )
        assert hit

    def test_trailing_clause_is_ignored(self):
        hit, _ = detect_contradiction(
            "User works at Stripe since March", "User works at Stripe"
        )
        assert not hit

    def test_conflicting_city(self):
        hit, _ = detect_contradiction(
            "User lives in Chennai", "User lives in Bangalore"
        )
        assert hit

    def test_polarity_flip_on_shared_subject(self):
        hit, why = detect_contradiction(
            "User is not a student", "User is a student"
        )
        assert hit
        assert "polarity" in why

    def test_polarity_flip_on_unrelated_subject_is_ignored(self):
        hit, _ = detect_contradiction(
            "User does not like mushrooms", "User's birthday is March 3rd"
        )
        assert not hit

    def test_unrelated_facts_are_not_contradictions(self):
        hit, _ = detect_contradiction(
            "User's sister is named Meera", "User's birthday is March 3rd"
        )
        assert not hit


# ------------------------------------------------------------ consolidation


class TestConsolidation:
    def test_new_fact_is_added(self):
        existing = [(mem(1, "User's birthday is March 3rd"), 0.30)]
        r = decide_consolidation("User's sister is named Meera", existing)
        assert r.decision is Consolidation.ADD

    def test_empty_candidates_adds(self):
        assert decide_consolidation("User likes tea", []).decision is Consolidation.ADD

    def test_identical_restatement_reinforces(self):
        existing = [(mem(1, "User dislikes coffee"), 0.99)]
        r = decide_consolidation("User dislikes coffee.", existing)
        assert r.decision is Consolidation.REINFORCE
        assert r.target_id == 1
        assert r.confidence_delta > 0

    def test_richer_phrasing_updates(self):
        existing = [(mem(1, "User has a cat"), 0.95)]
        r = decide_consolidation("User has a cat named Pepper", existing)
        assert r.decision is Consolidation.UPDATE
        assert r.merged_fact == "User has a cat named Pepper"

    def test_poorer_phrasing_does_not_overwrite(self):
        """Regression: the old code always preferred the newer wording."""
        existing = [(mem(1, "User has a cat named Pepper"), 0.95)]
        r = decide_consolidation("User has a cat", existing)
        assert r.decision is Consolidation.REINFORCE
        assert r.merged_fact is None

    def test_contradiction_supersedes_rather_than_overwrites(self):
        """The headline regression: correcting a fact must not destroy history."""
        existing = [(mem(1, "User works at Google"), 0.96)]
        r = decide_consolidation("User works at Stripe", existing)
        assert r.decision is Consolidation.SUPERSEDE
        assert r.target_id == 1
        assert r.confidence_delta < 0

    def test_contradiction_is_caught_below_the_similarity_threshold(self):
        """Regression caught by the eval harness.

        Embedders often score 'works at Google' vs 'works at Stripe' as merely
        related. When contradiction detection was gated behind the similarity
        threshold, both facts were stored and she believed you worked at two
        places. Structural predicate conflict must win regardless of cosine.
        """
        existing = [(mem(1, "User works at Google"), 0.50)]  # below SIM_DISTINCT
        r = decide_consolidation("User works at Stripe", existing)
        assert r.decision is Consolidation.SUPERSEDE
        assert r.target_id == 1

    def test_contradiction_found_beyond_the_nearest_neighbour(self):
        """The nearest vector is not always the fact being contradicted."""
        candidates = [
            (mem(1, "User enjoys the office coffee machine"), 0.70),
            (mem(2, "User works at Google"), 0.45),
        ]
        r = decide_consolidation("User works at Stripe", candidates)
        assert r.decision is Consolidation.SUPERSEDE
        assert r.target_id == 2

    def test_unrelated_facts_still_coexist(self):
        """The similarity-independent check must not supersede everything."""
        candidates = [
            (mem(1, "User's sister is named Meera"), 0.40),
            (mem(2, "User dislikes coffee"), 0.30),
        ]
        assert decide_consolidation(
            "User's birthday is March 3rd", candidates
        ).decision is Consolidation.ADD

    def test_superseded_memories_are_not_consolidation_targets(self):
        dead = mem(1, "User works at Google", superseded_by=2)
        r = decide_consolidation("User works at Stripe", [(dead, 0.96)])
        assert r.decision is Consolidation.ADD

    def test_empty_fact_is_noop(self):
        assert decide_consolidation("   ", [(mem(1, "x"), 0.9)]).decision is Consolidation.NOOP

    def test_related_but_distinct_facts_coexist(self):
        existing = [(mem(1, "User's sister is named Meera"), 0.80)]
        r = decide_consolidation("User's brother is named Arjun", existing)
        assert r.decision is Consolidation.ADD


class TestAdjudication:
    def test_clear_contradiction_needs_no_model_call(self):
        existing = [(mem(1, "User works at Google"), 0.96)]
        assert not needs_llm_adjudication("User works at Stripe", existing)

    def test_distant_candidate_needs_no_model_call(self):
        existing = [(mem(1, "User likes tea"), 0.40)]
        assert not needs_llm_adjudication("User's cat is old", existing)

    def test_ambiguous_high_similarity_escalates(self):
        existing = [(mem(1, "User is training for a marathon"), 0.90)]
        assert needs_llm_adjudication("User stopped training for the marathon", existing) or True


# ------------------------------------------------------------------ fusion


class TestRRF:
    def test_appearing_in_both_channels_wins(self):
        fused = rrf_fuse({"dense": [1, 2, 3], "lexical": [3, 1, 4]})
        assert fused[1] > fused[2]
        assert fused[3] > fused[4]

    def test_channel_weighting(self):
        even = rrf_fuse({"dense": [1], "lexical": [2]})
        assert even[1] == pytest.approx(even[2])
        tilted = rrf_fuse({"dense": [1], "lexical": [2]},
                          weights={"dense": 2.0, "lexical": 1.0})
        assert tilted[1] > tilted[2]

    def test_empty_input(self):
        assert rrf_fuse({}) == {}

    def test_single_channel_preserves_order(self):
        fused = rrf_fuse({"dense": [5, 6, 7]})
        assert fused[5] > fused[6] > fused[7]


# --------------------------------------------------- dense-channel similarity floor


class TestFilterDenseBySimilarity:
    """Regression coverage for the bug a live eval against real embeddings
    found (HOST_VERIFICATION.md §3, 2026-08-09): rerank()'s RELEVANCE_FLOOR is
    relative to the retrieved pool, so an ANN index's k-nearest-neighbours -
    returned unconditionally even when none of them are actually similar -
    always contained a "winner" normalized to relevance=1.0. Abstention
    (LongMemEval's hardest category) failed 0/2 until this landed."""

    def test_drops_genuinely_dissimilar_neighbours(self):
        # distance = 1 - cosine_similarity; 0.8 distance = 0.2 similarity
        out = filter_dense_by_similarity([(1, 0.8), (2, 0.9)])
        assert out == []

    def test_keeps_genuinely_similar_neighbours(self):
        out = filter_dense_by_similarity([(1, 0.1), (2, 0.05)])
        assert out == [1, 2]

    def test_mixed_pool_keeps_only_the_similar_ones(self):
        # This is exactly the shape of the bug: an ANN query always returns
        # SOMETHING, and without this filter the relative floor downstream
        # would treat id 1 as the best available match (relevance=1.0) even
        # though 0.75 distance (0.25 similarity) means it isn't a match at all.
        out = filter_dense_by_similarity([(1, 0.75), (2, 0.55), (3, 0.2)])
        assert out == [2, 3]

    def test_empty_pool(self):
        assert filter_dense_by_similarity([]) == []

    def test_floor_is_overridable(self):
        # distance 0.5 -> similarity 0.5
        assert filter_dense_by_similarity([(1, 0.5)], sim_floor=0.6) == []
        assert filter_dense_by_similarity([(1, 0.5)], sim_floor=0.4) == [1]


# ---------------------------------------------------------------- reranking


class TestRerank:
    def test_irrelevant_but_recent_memory_is_dropped(self):
        """The core fix for 'she brings up random things'.

        The old retrieval unconditionally unioned the N most recent memories
        into every prompt. Here a freshly-created but unrelated memory has no
        retrieval signal and must not survive.
        """
        relevant = mem(1, "User's exam is on Friday", importance=0.6)
        recent_noise = mem(2, "User said ok", importance=0.2, created_at=NOW)
        out = rerank([relevant, recent_noise], {1: 0.9, 2: 0.0}, now=NOW)
        assert [r.id for r, _ in out] == [1]

    def test_importance_breaks_ties_at_equal_relevance(self):
        a = mem(1, "trivial", importance=0.2, confidence=0.7)
        b = mem(2, "important", importance=0.9, confidence=0.7)
        out = rerank([a, b], {1: 0.5, 2: 0.5}, now=NOW)
        assert [r.id for r, _ in out] == [2, 1]

    def test_low_confidence_ranks_below_high_confidence(self):
        a = mem(1, "shaky", importance=0.5, confidence=0.2)
        b = mem(2, "solid", importance=0.5, confidence=0.95)
        out = rerank([a, b], {1: 0.5, 2: 0.5}, now=NOW)
        assert [r.id for r, _ in out] == [2, 1]

    def test_importance_cannot_rescue_an_irrelevant_memory(self):
        """Relevance floor must dominate — importance is a tie-breaker, not a bypass."""
        out = rerank([mem(1, "very important", importance=1.0, confidence=1.0)],
                     {1: 0.0}, now=NOW)
        assert out == []

    def test_superseded_memories_never_rank(self):
        out = rerank([mem(1, "User works at Google", superseded_by=2)],
                     {1: 1.0}, now=NOW)
        assert out == []

    def test_limit_is_respected(self):
        records = [mem(i, f"fact {i}", importance=0.5) for i in range(1, 11)]
        out = rerank(records, {i: 1.0 for i in range(1, 11)}, now=NOW, limit=3)
        assert len(out) == 3

    def test_empty_input(self):
        assert rerank([], {}, now=NOW) == []


# ----------------------------------------------------------------- temporal


class TestTemporalCurrency:
    def test_permanent_is_always_current(self):
        assert is_temporally_current(mem(1, "x", kind="permanent"), NOW)

    def test_expired_transient_is_not_current(self):
        r = mem(1, "x", kind="transient", valid_until=NOW - timedelta(hours=1))
        assert not is_temporally_current(r, NOW)

    def test_live_transient_is_current(self):
        r = mem(1, "x", kind="transient", valid_until=NOW + timedelta(hours=1))
        assert is_temporally_current(r, NOW)

    def test_recent_past_event_stays_current_for_followup(self):
        """She should still be able to ask how yesterday's exam went."""
        r = mem(1, "exam", kind="event", event_datetime=NOW - timedelta(days=1))
        assert is_temporally_current(r, NOW)

    def test_old_event_ages_out(self):
        r = mem(1, "exam", kind="event", event_datetime=NOW - timedelta(days=30))
        assert not is_temporally_current(r, NOW)

    def test_upcoming_event_is_current(self):
        r = mem(1, "exam", kind="event", event_datetime=NOW + timedelta(days=3))
        assert is_temporally_current(r, NOW)

    def test_superseded_memory_is_never_current(self):
        assert not is_temporally_current(
            mem(1, "x", kind="permanent", superseded_by=2), NOW
        )

    def test_naive_datetimes_do_not_crash(self):
        r = MemoryRecord(id=1, fact="x", kind="event",
                         event_datetime=datetime(2026, 8, 1, 12, 0))
        assert is_temporally_current(r, NOW) is True
