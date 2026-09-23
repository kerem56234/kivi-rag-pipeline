# retriever.py

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence


# =========================
# Constants
# =========================

DEFAULT_RESULT_LIMIT = 5
EXPECTED_VECTOR_DIMENSIONS = 768
DEFAULT_RRF_K = 60


# =========================
# Exceptions
# =========================

class RetrieverError(Exception):
    """Base exception for retriever errors."""


class InvalidSearchVectorError(RetrieverError):
    """Raised when the search vector is missing or does not have 768 dimensions."""


class InvalidRelationError(RetrieverError):
    """Raised when the table/view name is invalid."""


class InvalidResultLimitError(RetrieverError):
    """Raised when the requested result limit is invalid."""


class LexicalQueryMissingError(RetrieverError):
    """Raised when lexical search is required but no query text was provided."""


# =========================
# Database Protocol
# =========================

class DatabaseClient(Protocol):
    """
    Minimal protocol for database access.

    Unit tests can use a fake object implementing this protocol.
    A production implementation can wrap psycopg, asyncpg, SQLAlchemy, etc.
    """

    def fetch_all(
        self,
        query: str,
        params: Optional[Sequence[Any]] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


# =========================
# Lexical Search Options
# =========================

class LexicalSearchMode(str, Enum):
    """
    PostgreSQL lexical search modes we may support.
    """

    FTS_PLAIN = "fts_plain"
    FTS_WEB = "fts_web"
    FTS_PHRASE = "fts_phrase"
    TRIGRAM = "trigram"
    ILIKE = "ilike"


# =========================
# Data Models
# =========================

@dataclass(frozen=True)
class RetrievalRequest:
    search_vector: Sequence[float]
    relation: str
    result_limit: int = DEFAULT_RESULT_LIMIT
    query_text: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    id: Any
    chunk: str
    rank: int
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    id: Any
    chunk: str
    rrf_score: float
    vector_score: Optional[float] = None
    lexical_score: Optional[float] = None
    vector_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =========================
# Internal Validation Helpers
# =========================

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RELATION_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$"
)


def _validate_identifier(identifier: str) -> None:
    if not isinstance(identifier, str) or not _IDENTIFIER_RE.fullmatch(identifier):
        raise InvalidRelationError(f"Invalid SQL identifier: {identifier!r}")


def _validate_relation_name(relation: str) -> None:
    if not isinstance(relation, str) or not _RELATION_RE.fullmatch(relation.strip()):
        raise InvalidRelationError(f"Invalid relation name: {relation!r}")


# =========================
# Candidate Sources
# =========================

class CandidateSource(ABC):
    """Base class for retrieval sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def search(self, request: RetrievalRequest, limit: int) -> List[Candidate]:
        raise NotImplementedError


class PgVectorCandidateSource(CandidateSource):
    """
    Candidate source for pgvector cosine similarity search.

    Expected SQL shape:

        SELECT id, chunk, embedding <=> query_vector AS score
        FROM relation
        ORDER BY embedding <=> query_vector
        LIMIT limit;

    With pgvector cosine distance, lower distance is better.
    """

    def __init__(
        self,
        db: DatabaseClient,
        embedding_column: str = "embedding",
        id_column: str = "id",
        chunk_column: str = "chunk",
    ) -> None:
        _validate_identifier(embedding_column)
        _validate_identifier(id_column)
        _validate_identifier(chunk_column)

        self._db = db
        self._embedding_column = embedding_column
        self._id_column = id_column
        self._chunk_column = chunk_column

    @property
    def name(self) -> str:
        return "vector"

    def search(self, request: RetrievalRequest, limit: int) -> List[Candidate]:
        _validate_relation_name(request.relation)

        sql = self._build_sql(request.relation)
        rows = self._db.fetch_all(
            sql,
            params=[
                list(request.search_vector),
                list(request.search_vector),
                limit,
            ],
        )

        return [
            self._row_to_candidate(row=row, rank=rank)
            for rank, row in enumerate(rows, start=1)
        ]

    def _build_sql(self, relation: str) -> str:
        _validate_relation_name(relation)

        return f"""
            SELECT
                {self._id_column} AS id,
                {self._chunk_column} AS chunk,
                {self._embedding_column} <=> %s::vector AS score
            FROM {relation}
            ORDER BY {self._embedding_column} <=> %s::vector
            LIMIT %s
        """

    def _row_to_candidate(self, row: Dict[str, Any], rank: int) -> Candidate:
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"id", "chunk", "score"}
        }

        return Candidate(
            id=row["id"],
            chunk=row["chunk"],
            rank=rank,
            score=float(row["score"]),
            source=self.name,
            metadata=metadata,
        )


class PostgresLexicalCandidateSource(CandidateSource):
    """
    Candidate source for PostgreSQL lexical search.

    Supported mock-friendly SQL strategies:
    - Full-text search with plainto_tsquery
    - Full-text search with websearch_to_tsquery
    - Full-text search with phraseto_tsquery
    - pg_trgm similarity search
    - ILIKE fallback
    """

    def __init__(
        self,
        db: DatabaseClient,
        mode: LexicalSearchMode = LexicalSearchMode.FTS_WEB,
        id_column: str = "id",
        chunk_column: str = "chunk",
        text_column: str = "chunk",
        language_config: str = "english",
    ) -> None:
        _validate_identifier(id_column)
        _validate_identifier(chunk_column)
        _validate_identifier(text_column)

        self._db = db
        self._mode = mode
        self._id_column = id_column
        self._chunk_column = chunk_column
        self._text_column = text_column
        self._language_config = language_config

    @property
    def name(self) -> str:
        return "lexical"

    def search(self, request: RetrievalRequest, limit: int) -> List[Candidate]:
        if request.query_text is None or not request.query_text.strip():
            return []

        _validate_relation_name(request.relation)

        sql = self._build_sql(request.relation)
        params = self._build_params(request.query_text, limit)
        rows = self._db.fetch_all(sql, params=params)

        return [
            self._row_to_candidate(row=row, rank=rank)
            for rank, row in enumerate(rows, start=1)
        ]

    def _build_sql(self, relation: str) -> str:
        if self._mode == LexicalSearchMode.FTS_PLAIN:
            return self._build_fts_plain_sql(relation)

        if self._mode == LexicalSearchMode.FTS_WEB:
            return self._build_fts_web_sql(relation)

        if self._mode == LexicalSearchMode.FTS_PHRASE:
            return self._build_fts_phrase_sql(relation)

        if self._mode == LexicalSearchMode.TRIGRAM:
            return self._build_trigram_sql(relation)

        if self._mode == LexicalSearchMode.ILIKE:
            return self._build_ilike_sql(relation)

        raise RetrieverError(f"Unsupported lexical search mode: {self._mode}")

    def _build_fts_plain_sql(self, relation: str) -> str:
        _validate_relation_name(relation)

        return f"""
            WITH query AS (
                SELECT plainto_tsquery(%s::regconfig, %s) AS q
            )
            SELECT
                {self._id_column} AS id,
                {self._chunk_column} AS chunk,
                ts_rank_cd(
                    to_tsvector(%s::regconfig, COALESCE({self._text_column}, '')),
                    query.q
                ) AS score
            FROM {relation}, query
            WHERE to_tsvector(%s::regconfig, COALESCE({self._text_column}, '')) @@ query.q
            ORDER BY score DESC
            LIMIT %s
        """

    def _build_fts_web_sql(self, relation: str) -> str:
        _validate_relation_name(relation)

        return f"""
            WITH query AS (
                SELECT websearch_to_tsquery(%s::regconfig, %s) AS q
            )
            SELECT
                {self._id_column} AS id,
                {self._chunk_column} AS chunk,
                ts_rank_cd(
                    to_tsvector(%s::regconfig, COALESCE({self._text_column}, '')),
                    query.q
                ) AS score
            FROM {relation}, query
            WHERE to_tsvector(%s::regconfig, COALESCE({self._text_column}, '')) @@ query.q
            ORDER BY score DESC
            LIMIT %s
        """

    def _build_fts_phrase_sql(self, relation: str) -> str:
        _validate_relation_name(relation)

        return f"""
            WITH query AS (
                SELECT phraseto_tsquery(%s::regconfig, %s) AS q
            )
            SELECT
                {self._id_column} AS id,
                {self._chunk_column} AS chunk,
                ts_rank_cd(
                    to_tsvector(%s::regconfig, COALESCE({self._text_column}, '')),
                    query.q
                ) AS score
            FROM {relation}, query
            WHERE to_tsvector(%s::regconfig, COALESCE({self._text_column}, '')) @@ query.q
            ORDER BY score DESC
            LIMIT %s
        """

    def _build_trigram_sql(self, relation: str) -> str:
        _validate_relation_name(relation)

        return f"""
            SELECT
                {self._id_column} AS id,
                {self._chunk_column} AS chunk,
                similarity({self._text_column}, %s) AS score
            FROM {relation}
            WHERE {self._text_column} %% %s
            ORDER BY score DESC
            LIMIT %s
        """

    def _build_ilike_sql(self, relation: str) -> str:
        _validate_relation_name(relation)

        return f"""
            SELECT
                {self._id_column} AS id,
                {self._chunk_column} AS chunk,
                1.0 AS score
            FROM {relation}
            WHERE {self._text_column} ILIKE %s
            LIMIT %s
        """

    def _build_params(self, query_text: str, limit: int) -> List[Any]:
        if self._mode in {
            LexicalSearchMode.FTS_PLAIN,
            LexicalSearchMode.FTS_WEB,
            LexicalSearchMode.FTS_PHRASE,
        }:
            return [
                self._language_config,
                query_text,
                self._language_config,
                self._language_config,
                limit,
            ]

        if self._mode == LexicalSearchMode.TRIGRAM:
            return [
                query_text,
                query_text,
                limit,
            ]

        if self._mode == LexicalSearchMode.ILIKE:
            return [
                f"%{query_text}%",
                limit,
            ]

        raise RetrieverError(f"Unsupported lexical search mode: {self._mode}")

    def _row_to_candidate(self, row: Dict[str, Any], rank: int) -> Candidate:
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"id", "chunk", "score"}
        }

        return Candidate(
            id=row["id"],
            chunk=row["chunk"],
            rank=rank,
            score=float(row["score"]),
            source=self.name,
            metadata=metadata,
        )


# =========================
# RRF Fusion
# =========================

class ReciprocalRankFusion:
    """Fuses ranked candidates using Reciprocal Rank Fusion."""

    def __init__(self, k: int = DEFAULT_RRF_K) -> None:
        if not isinstance(k, int) or k < 0:
            raise InvalidResultLimitError("RRF k must be a non-negative integer.")

        self._k = k

    def fuse(
        self,
        candidates: Iterable[Candidate],
        limit: int,
    ) -> List[RetrievalResult]:
        if not isinstance(limit, int) or limit <= 0:
            raise InvalidResultLimitError("Fusion limit must be a positive integer.")

        fused_by_key: Dict[Any, RetrievalResult] = {}

        for candidate in candidates:
            key = self._candidate_key(candidate)
            existing = fused_by_key.get(key)
            fused_by_key[key] = self._merge_candidate(existing, candidate)

        sorted_results = self._sort_results(fused_by_key.values())

        return sorted_results[:limit]

    def contribution(self, rank: int) -> float:
        if not isinstance(rank, int) or rank <= 0:
            raise RetrieverError("Rank must be a positive integer.")

        return 1.0 / (self._k + rank)

    def _candidate_key(self, candidate: Candidate) -> Any:
        return candidate.id

    def _merge_candidate(
        self,
        existing: Optional[RetrievalResult],
        candidate: Candidate,
    ) -> RetrievalResult:
        contribution = self.contribution(candidate.rank)

        if existing is None:
            return RetrievalResult(
                id=candidate.id,
                chunk=candidate.chunk,
                rrf_score=contribution,
                vector_score=(
                    candidate.score if candidate.source == "vector" else None
                ),
                lexical_score=(
                    candidate.score if candidate.source == "lexical" else None
                ),
                vector_rank=(
                    candidate.rank if candidate.source == "vector" else None
                ),
                lexical_rank=(
                    candidate.rank if candidate.source == "lexical" else None
                ),
                metadata=dict(candidate.metadata),
            )

        return RetrievalResult(
            id=existing.id,
            chunk=existing.chunk,
            rrf_score=existing.rrf_score + contribution,
            vector_score=self._choose_source_score(
                old_score=existing.vector_score,
                old_rank=existing.vector_rank,
                candidate=candidate,
                source="vector",
            ),
            lexical_score=self._choose_source_score(
                old_score=existing.lexical_score,
                old_rank=existing.lexical_rank,
                candidate=candidate,
                source="lexical",
            ),
            vector_rank=self._choose_source_rank(
                old_rank=existing.vector_rank,
                candidate=candidate,
                source="vector",
            ),
            lexical_rank=self._choose_source_rank(
                old_rank=existing.lexical_rank,
                candidate=candidate,
                source="lexical",
            ),
            metadata={
                **existing.metadata,
                **candidate.metadata,
            },
        )

    def _choose_source_score(
        self,
        old_score: Optional[float],
        old_rank: Optional[int],
        candidate: Candidate,
        source: str,
    ) -> Optional[float]:
        if candidate.source != source:
            return old_score

        if old_rank is None:
            return candidate.score

        if candidate.rank < old_rank:
            return candidate.score

        return old_score

    def _choose_source_rank(
        self,
        old_rank: Optional[int],
        candidate: Candidate,
        source: str,
    ) -> Optional[int]:
        if candidate.source != source:
            return old_rank

        if old_rank is None:
            return candidate.rank

        return min(old_rank, candidate.rank)

    def _sort_results(
        self,
        results: Iterable[RetrievalResult],
    ) -> List[RetrievalResult]:
        return sorted(
            results,
            key=lambda result: (
                -result.rrf_score,
                str(result.id),
            ),
        )


# =========================
# Main Retriever
# =========================

class Retriever:
    """
    Main object-oriented retriever.

    Public flow:
    1. Validate request.
    2. Run vector search.
    3. Optionally run lexical search.
    4. Fuse results with RRF.
    5. Return up to result_limit results.
    """

    def __init__(
        self,
        vector_source: CandidateSource,
        lexical_source: CandidateSource,
        fusion: Optional[ReciprocalRankFusion] = None,
        require_lexical_query: bool = False,
    ) -> None:
        self._vector_source = vector_source
        self._lexical_source = lexical_source
        self._fusion = fusion or ReciprocalRankFusion()
        self._require_lexical_query = require_lexical_query

    def retrieve(
        self,
        search_vector: Sequence[float],
        relation: str,
        result_limit: int = DEFAULT_RESULT_LIMIT,
        query_text: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        request = self._build_request(
            search_vector=search_vector,
            relation=relation,
            result_limit=result_limit,
            query_text=query_text,
            filters=filters,
        )

        self._validate_request(request)

        candidates = self._collect_candidates(request)

        return self._fusion.fuse(
            candidates=candidates,
            limit=request.result_limit,
        )

    def _build_request(
        self,
        search_vector: Sequence[float],
        relation: str,
        result_limit: int,
        query_text: Optional[str],
        filters: Optional[Dict[str, Any]],
    ) -> RetrievalRequest:
        return RetrievalRequest(
            search_vector=search_vector,
            relation=relation,
            result_limit=result_limit,
            query_text=query_text,
            filters=filters or {},
        )

    def _validate_request(self, request: RetrievalRequest) -> None:
        self._validate_search_vector(request.search_vector)
        self._validate_relation(request.relation)
        self._validate_result_limit(request.result_limit)
        self._validate_query_text(request.query_text)

    def _validate_search_vector(self, search_vector: Sequence[float]) -> None:
        if search_vector is None:
            raise InvalidSearchVectorError("Search vector must not be None.")

        if len(search_vector) != EXPECTED_VECTOR_DIMENSIONS:
            raise InvalidSearchVectorError(
                f"Search vector must have "
                f"{EXPECTED_VECTOR_DIMENSIONS} dimensions."
            )

        for value in search_vector:
            if not isinstance(value, (int, float)):
                raise InvalidSearchVectorError(
                    "Search vector must contain only numeric values."
                )

    def _validate_relation(self, relation: str) -> None:
        _validate_relation_name(relation)

    def _validate_result_limit(self, result_limit: int) -> None:
        if not isinstance(result_limit, int) or result_limit <= 0:
            raise InvalidResultLimitError(
                "Result limit must be a positive integer."
            )

    def _validate_query_text(self, query_text: Optional[str]) -> None:
        if not self._require_lexical_query:
            return

        if query_text is None or not query_text.strip():
            raise LexicalQueryMissingError(
                "query_text is required when lexical search is mandatory."
            )

    def _collect_candidates(self, request: RetrievalRequest) -> List[Candidate]:
        candidates: List[Candidate] = []

        vector_candidates = self._vector_source.search(
            request=request,
            limit=request.result_limit,
        )
        candidates.extend(vector_candidates)

        if self._should_run_lexical_search(request):
            lexical_candidates = self._lexical_source.search(
                request=request,
                limit=request.result_limit,
            )
            candidates.extend(lexical_candidates)

        return candidates

    def _should_run_lexical_search(self, request: RetrievalRequest) -> bool:
        return request.query_text is not None and bool(request.query_text.strip())