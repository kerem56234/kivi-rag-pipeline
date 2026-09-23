# integrator.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence


# =========================
# Constants
# =========================

INSUFFICIENT_CONTEXT_TOKEN = "INSUFFICIENT_CONTEXT"

DEFAULT_GENERATOR_SYSTEM_PROMPT = """\
You are a grounded RAG answer generator.

Your task is to answer the user's question using only the provided context.
Each context item contains an id and a text chunk.

Use the context carefully and cite relevant ids when useful.
Do not use outside knowledge.

If the context is empty, irrelevant, or does not contain enough information
to answer the user's question reliably, respond exactly with:

INSUFFICIENT_CONTEXT

Only use INSUFFICIENT_CONTEXT when the provided context truly cannot answer
the question. If the context partially answers the question, provide the
partial answer and clearly state what is missing.

User question:
{user_prompt}

Context:
{context}

Answer:
"""


# =========================
# Exceptions
# =========================

class IntegratorError(Exception):
    """Base exception for integrator errors."""


class EmptyUserPromptError(IntegratorError):
    """Raised when the user prompt is empty."""


class InvalidRetrievalViewError(IntegratorError):
    """Raised when the retrieval view is empty or invalid."""


class EmbeddingModelError(IntegratorError):
    """Raised when the embedding model fails."""


class RetrieverCallError(IntegratorError):
    """Raised when the retriever fails."""


class GeneratorModelError(IntegratorError):
    """Raised when the generator model fails."""


# =========================
# Protocols / Interfaces
# =========================

class EmbeddingModel(Protocol):
    """Interface for the embedding model."""

    def embed(self, text: str) -> Sequence[float]:
        raise NotImplementedError


class Retriever(Protocol):
    """Interface for the retriever."""

    def retrieve(
        self,
        search_vector: Sequence[float],
        relation: str,
        result_limit: int = 5,
        query_text: Optional[str] = None,
    ) -> List[Any]:
        raise NotImplementedError


class GeneratorModel(Protocol):
    """Interface for the generating model."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


# =========================
# Data Models
# =========================

@dataclass(frozen=True)
class RetrievedSource:
    """
    Normalized source item passed from retriever to integrator.

    id:
        Source or chunk id.

    chunk:
        Retrieved text chunk.

    score:
        Optional final retrieval score, e.g. RRF score.

    metadata:
        Optional additional metadata.
    """
    id: Any
    chunk: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntegratorRequest:
    user_prompt: str
    retrieval_view: str
    result_limit: int = 5


@dataclass(frozen=True)
class IntegratorResponse:
    answer: str
    sources: List[RetrievedSource] = field(default_factory=list)
    needs_new_search_query: bool = False
    insufficient_context: bool = False


# =========================
# Integrator
# =========================

class RAGIntegrator:
    """
    Coordinates the complete RAG pipeline.

    Flow:
    1. Validate user prompt and retrieval view.
    2. Embed user prompt.
    3. Call retriever with search vector, retrieval view, and query text.
    4. Form context from retrieved results.
    5. If no retrieved results, ask for another search query.
    6. Otherwise call generator.
    7. If generator returns INSUFFICIENT_CONTEXT, ask for another search query.
    8. Otherwise render answer.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        retriever: Retriever,
        generator_model: GeneratorModel,
        generator_prompt_template: str = DEFAULT_GENERATOR_SYSTEM_PROMPT,
    ) -> None:
        self._embedding_model = embedding_model
        self._retriever = retriever
        self._generator_model = generator_model
        self._generator_prompt_template = generator_prompt_template

    def answer(
        self,
        user_prompt: str,
        retrieval_view: str,
        result_limit: int = 5,
    ) -> IntegratorResponse:
        """Main public method for answering a user question."""
        raise NotImplementedError

    def embed_and_retrieve(
        self,
        user_prompt: str,
        retrieval_view: str,
        result_limit: int = 5,
    ) -> List[RetrievedSource]:
        """Embed the user prompt and retrieve relevant chunks."""
        raise NotImplementedError

    def form_context(self, retrieved_results: List[RetrievedSource]) -> str:
        """Format retrieved chunks into generator-ready context."""
        raise NotImplementedError

    def call_generator(self, context: str, user_prompt: str) -> str:
        """Create the final generator prompt and call the generator model."""
        raise NotImplementedError

    def render_answer(
        self,
        generated_answer: str,
        retrieved_sources: List[RetrievedSource],
    ) -> IntegratorResponse:
        """Render the final answer and attach retrieved sources."""
        raise NotImplementedError

    def ask_for_new_search_query(self) -> IntegratorResponse:
        """Return a response asking the user whether to try another search query."""
        raise NotImplementedError

    def build_generator_prompt(self, context: str, user_prompt: str) -> str:
        """Fill the generator prompt template with context and user prompt."""
        raise NotImplementedError

    def normalize_retrieved_result(self, raw_result: Any) -> RetrievedSource:
        """Convert retriever-specific result objects into RetrievedSource."""
        raise NotImplementedError

    def is_insufficient_context_response(self, generated_answer: str) -> bool:
        """Check whether the generator returned the insufficient-context token."""
        raise NotImplementedError

    def _validate_request(self, request: IntegratorRequest) -> None:
        """Validate the integrator request."""
        raise NotImplementedError

    def _validate_user_prompt(self, user_prompt: str) -> None:
        """Validate that the user prompt is non-empty."""
        raise NotImplementedError

    def _validate_retrieval_view(self, retrieval_view: str) -> None:
        """Validate that the retrieval view is non-empty."""
        raise NotImplementedError