# tests/test_retriever.py

import math
import pytest

from rag_pipeline.retriever import (
    DEFAULT_RESULT_LIMIT,
    DEFAULT_RRF_K,
    EXPECTED_VECTOR_DIMENSIONS,
    Candidate,
    InvalidRelationError,
    InvalidResultLimitError,
    InvalidSearchVectorError,
    LexicalQueryMissingError,
    ReciprocalRankFusion,
    RetrievalRequest,
    RetrievalResult,
    Retriever,
)


# =========================
# Test Helpers
# =========================

def valid_vector(dimensions: int = EXPECTED_VECTOR_DIMENSIONS):
    return [0.1] * dimensions


class FakeCandidateSource:
    def __init__(self, name, candidates=None):
        self._name = name
        self._candidates = candidates or []
        self.calls = []

    @property
    def name(self):
        return self._name

    def search(self, request: RetrievalRequest, limit: int):
        self.calls.append({"request": request, "limit": limit})
        return self._candidates[:limit]


class SpyFusion:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def fuse(self, candidates, limit):
        candidates = list(candidates)
        self.calls.append({"candidates": candidates, "limit": limit})
        return self.results[:limit]


# =========================
# Request / Validation Tests
# =========================

def test_retrieve_uses_default_result_limit_when_none_is_given():
    vector_source = FakeCandidateSource("vector")
    lexical_source = FakeCandidateSource("lexical")
    fusion = SpyFusion(results=[])

    retriever = Retriever(
        vector_source=vector_source,
        lexical_source=lexical_source,
        fusion=fusion,
    )

    retriever.retrieve(
        search_vector=valid_vector(),
        relation="wiki_embedding",
        query_text="What is pgai?",
    )

    assert vector_source.calls[0]["limit"] == DEFAULT_RESULT_LIMIT
    assert lexical_source.calls[0]["limit"] == DEFAULT_RESULT_LIMIT
    assert fusion.calls[0]["limit"] == DEFAULT_RESULT_LIMIT


@pytest.mark.parametrize(
    "bad_vector",
    [
        [],
        [0.1] * 767,
        [0.1] * 769,
    ],
)
def test_retrieve_rejects_vectors_that_are_not_768_dimensions(bad_vector):
    retriever = Retriever(
        vector_source=FakeCandidateSource("vector"),
        lexical_source=FakeCandidateSource("lexical"),
    )

    with pytest.raises(InvalidSearchVectorError):
        retriever.retrieve(
            search_vector=bad_vector,
            relation="wiki_embedding",
            query_text="test query",
        )


@pytest.mark.parametrize(
    "bad_relation",
    [
        "",
        " ",
        "wiki embedding",
        "wiki-embedding",
        "wiki_embedding;",
        "wiki_embedding; DROP TABLE users;",
    ],
)
def test_retrieve_rejects_invalid_relation_names(bad_relation):
    retriever = Retriever(
        vector_source=FakeCandidateSource("vector"),
        lexical_source=FakeCandidateSource("lexical"),
    )

    with pytest.raises(InvalidRelationError):
        retriever.retrieve(
            search_vector=valid_vector(),
            relation=bad_relation,
            query_text="test query",
        )


@pytest.mark.parametrize(
    "valid_relation",
    [
        "wiki_embedding",
        "public.wiki_embedding",
        "rag_chunks_v1",
    ],
)
def test_retrieve_accepts_valid_relation_names(valid_relation):
    vector_source = FakeCandidateSource("vector")
    lexical_source = FakeCandidateSource("lexical")
    fusion = SpyFusion(results=[])

    retriever = Retriever(
        vector_source=vector_source,
        lexical_source=lexical_source,
        fusion=fusion,
    )

    retriever.retrieve(
        search_vector=valid_vector(),
        relation=valid_relation,
        query_text="test query",
    )

    assert vector_source.calls[0]["request"].relation == valid_relation


@pytest.mark.parametrize("bad_limit", [0, -1, -10])
def test_retrieve_rejects_non_positive_result_limits(bad_limit):
    retriever = Retriever(
        vector_source=FakeCandidateSource("vector"),
        lexical_source=FakeCandidateSource("lexical"),
    )

    with pytest.raises(InvalidResultLimitError):
        retriever.retrieve(
            search_vector=valid_vector(),
            relation="wiki_embedding",
            result_limit=bad_limit,
            query_text="test query",
        )


def test_retrieve_rejects_missing_query_text_when_lexical_query_is_required():
    retriever = Retriever(
        vector_source=FakeCandidateSource("vector"),
        lexical_source=FakeCandidateSource("lexical"),
        require_lexical_query=True,
    )

    with pytest.raises(LexicalQueryMissingError):
        retriever.retrieve(
            search_vector=valid_vector(),
            relation="wiki_embedding",
            query_text=None,
        )


def test_retrieve_skips_lexical_search_when_query_text_is_missing_and_not_required():
    vector_candidate = Candidate(
        id=1,
        chunk="Vector-only result",
        rank=1,
        score=0.9,
        source="vector",
    )

    vector_source = FakeCandidateSource("vector", candidates=[vector_candidate])
    lexical_source = FakeCandidateSource("lexical")
    fusion = SpyFusion(
        results=[
            RetrievalResult(
                id=1,
                chunk="Vector-only result",
                rrf_score=1 / (DEFAULT_RRF_K + 1),
                vector_score=0.9,
                vector_rank=1,
            )
        ]
    )

    retriever = Retriever(
        vector_source=vector_source,
        lexical_source=lexical_source,
        fusion=fusion,
        require_lexical_query=False,
    )

    results = retriever.retrieve(
        search_vector=valid_vector(),
        relation="wiki_embedding",
        query_text=None,
    )

    assert len(vector_source.calls) == 1
    assert len(lexical_source.calls) == 0
    assert len(results) == 1
    assert results[0].id == 1


# =========================
# Orchestration Tests
# =========================

def test_retrieve_collects_vector_and_lexical_candidates_and_passes_them_to_fusion():
    vector_candidate = Candidate(
        id=1,
        chunk="Semantic result",
        rank=1,
        score=0.12,
        source="vector",
    )

    lexical_candidate = Candidate(
        id=2,
        chunk="Keyword result",
        rank=1,
        score=0.95,
        source="lexical",
    )

    final_result = RetrievalResult(
        id=1,
        chunk="Semantic result",
        rrf_score=0.1,
        vector_score=0.12,
        vector_rank=1,
    )

    vector_source = FakeCandidateSource("vector", candidates=[vector_candidate])
    lexical_source = FakeCandidateSource("lexical", candidates=[lexical_candidate])
    fusion = SpyFusion(results=[final_result])

    retriever = Retriever(
        vector_source=vector_source,
        lexical_source=lexical_source,
        fusion=fusion,
    )

    results = retriever.retrieve(
        search_vector=valid_vector(),
        relation="wiki_embedding",
        result_limit=5,
        query_text="What is pgai?",
    )

    assert len(vector_source.calls) == 1
    assert len(lexical_source.calls) == 1

    fused_candidates = fusion.calls[0]["candidates"]
    assert vector_candidate in fused_candidates
    assert lexical_candidate in fused_candidates

    assert results == [final_result]


def test_retrieve_returns_at_most_requested_number_of_results():
    fusion_results = [
        RetrievalResult(id=i, chunk=f"Result {i}", rrf_score=1.0 / i)
        for i in range(1, 10)
    ]

    retriever = Retriever(
        vector_source=FakeCandidateSource("vector"),
        lexical_source=FakeCandidateSource("lexical"),
        fusion=SpyFusion(results=fusion_results),
    )

    results = retriever.retrieve(
        search_vector=valid_vector(),
        relation="wiki_embedding",
        result_limit=3,
        query_text="test query",
    )

    assert len(results) == 3


# =========================
# RRF Tests
# =========================

def test_rrf_contribution_uses_expected_formula():
    fusion = ReciprocalRankFusion(k=60)

    assert fusion.contribution(rank=1) == pytest.approx(1 / 61)
    assert fusion.contribution(rank=5) == pytest.approx(1 / 65)
    assert fusion.contribution(rank=10) == pytest.approx(1 / 70)


def test_rrf_fuse_combines_scores_for_same_candidate_id():
    candidates = [
        Candidate(
            id=1,
            chunk="Same chunk",
            rank=1,
            score=0.10,
            source="vector",
        ),
        Candidate(
            id=1,
            chunk="Same chunk",
            rank=3,
            score=0.80,
            source="lexical",
        ),
    ]

    fusion = ReciprocalRankFusion(k=60)

    results = fusion.fuse(candidates, limit=5)

    assert len(results) == 1

    expected_score = (1 / 61) + (1 / 63)

    assert results[0].id == 1
    assert results[0].chunk == "Same chunk"
    assert results[0].rrf_score == pytest.approx(expected_score)
    assert results[0].vector_score == 0.10
    assert results[0].lexical_score == 0.80
    assert results[0].vector_rank == 1
    assert results[0].lexical_rank == 3


def test_rrf_fuse_sorts_results_by_descending_rrf_score():
    candidates = [
        Candidate(
            id=1,
            chunk="Weak result",
            rank=10,
            score=0.1,
            source="vector",
        ),
        Candidate(
            id=2,
            chunk="Strong result",
            rank=1,
            score=0.9,
            source="vector",
        ),
    ]

    fusion = ReciprocalRankFusion(k=60)

    results = fusion.fuse(candidates, limit=5)

    assert [result.id for result in results] == [2, 1]
    assert results[0].rrf_score > results[1].rrf_score


def test_rrf_fuse_applies_limit_after_sorting():
    candidates = [
        Candidate(id=1, chunk="Rank 3", rank=3, score=0.3, source="vector"),
        Candidate(id=2, chunk="Rank 1", rank=1, score=0.9, source="vector"),
        Candidate(id=3, chunk="Rank 2", rank=2, score=0.7, source="vector"),
    ]

    fusion = ReciprocalRankFusion(k=60)

    results = fusion.fuse(candidates, limit=2)

    assert len(results) == 2
    assert [result.id for result in results] == [2, 3]


def test_rrf_fuse_preserves_metadata_from_candidate():
    candidates = [
        Candidate(
            id=1,
            chunk="Chunk with metadata",
            rank=1,
            score=0.9,
            source="vector",
            metadata={"document_id": "doc-1", "title": "Test Document"},
        )
    ]

    fusion = ReciprocalRankFusion(k=60)

    results = fusion.fuse(candidates, limit=5)

    assert results[0].metadata == {
        "document_id": "doc-1",
        "title": "Test Document",
    }