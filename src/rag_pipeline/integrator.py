# integrator.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence

import re

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
# Embedding Model Implementation
# =========================




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

    _RETRIEVAL_VIEW_PATTERN = re.compile(
        r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$"
    )

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
        """Run the complete RAG pipeline for one user question."""
        request = IntegratorRequest(
            user_prompt=user_prompt,
            retrieval_view=retrieval_view,
            result_limit=result_limit,
        )
        self._validate_request(request)

        retrieved_sources = self.embed_and_retrieve(
            user_prompt=request.user_prompt,
            retrieval_view=request.retrieval_view,
            result_limit=request.result_limit,
        )

        # First safeguard: do not call the generator when retrieval produced
        # no context at all.
        if not retrieved_sources:
            return self.ask_for_new_search_query()

        context = self.form_context(retrieved_sources)

        # Defensive safeguard in case all retrieved chunks were empty.
        if not context.strip():
            response = self.ask_for_new_search_query()
            return IntegratorResponse(
                answer=response.answer,
                sources=retrieved_sources,
                needs_new_search_query=True,
                insufficient_context=True,
            )

        generated_answer = self.call_generator(
            context=context,
            user_prompt=request.user_prompt,
        )

        # Second safeguard: the generator may determine that non-empty
        # retrieval results are nevertheless semantically inadequate.
        if self.is_insufficient_context_response(generated_answer):
            response = self.ask_for_new_search_query()
            return IntegratorResponse(
                answer=response.answer,
                sources=retrieved_sources,
                needs_new_search_query=True,
                insufficient_context=True,
            )

        return self.render_answer(
            generated_answer=generated_answer,
            retrieved_sources=retrieved_sources,
        )

    def embed_and_retrieve(
        self,
        user_prompt: str,
        retrieval_view: str,
        result_limit: int = 5,
    ) -> List[RetrievedSource]:
        """Embed the user prompt and retrieve relevant chunks."""
        try:
            search_vector = self._embedding_model.embed(user_prompt)
        except Exception as exc:
            raise EmbeddingModelError(
                "The embedding model failed to embed the user prompt."
            ) from exc

        if search_vector is None:
            raise EmbeddingModelError(
                "The embedding model returned no search vector."
            )

        try:
            raw_results = self._retriever.retrieve(
                search_vector=search_vector,
                relation=retrieval_view,
                result_limit=result_limit,
                query_text=user_prompt,
            )
        except Exception as exc:
            raise RetrieverCallError(
                "The retriever failed while searching for relevant context."
            ) from exc

        if raw_results is None:
            raise RetrieverCallError(
                "The retriever returned None instead of a result list."
            )

        try:
            return [
                self.normalize_retrieved_result(raw_result)
                for raw_result in raw_results
            ]
        except RetrieverCallError:
            raise
        except Exception as exc:
            raise RetrieverCallError(
                "A retrieved result could not be normalized."
            ) from exc

    def form_context(
        self,
        retrieved_results: List[RetrievedSource],
    ) -> str:
        """Format retrieved chunks, including their IDs, for the generator."""
        context_items: List[str] = []

        for source in retrieved_results:
            context_items.append(
                f"[ID: {source.id}]\n{source.chunk}"
            )

        return "\n\n".join(context_items)

    def call_generator(
        self,
        context: str,
        user_prompt: str,
    ) -> str:
        """Create the grounded prompt and call the generator model."""
        prompt = self.build_generator_prompt(
            context=context,
            user_prompt=user_prompt,
        )

        try:
            generated_answer = self._generator_model.generate(prompt)
        except Exception as exc:
            raise GeneratorModelError(
                "The generator model failed to produce an answer."
            ) from exc

        if not isinstance(generated_answer, str):
            raise GeneratorModelError(
                "The generator model must return a string."
            )

        return generated_answer

    def render_answer(
        self,
        generated_answer: str,
        retrieved_sources: List[RetrievedSource],
    ) -> IntegratorResponse:
        """Return the generated answer with its retrieved sources."""
        return IntegratorResponse(
            answer=generated_answer,
            sources=list(retrieved_sources),
            needs_new_search_query=False,
            insufficient_context=False,
        )

    def ask_for_new_search_query(self) -> IntegratorResponse:
        """Ask the user whether another search should be attempted."""
        return IntegratorResponse(
            answer=(
                "I could not find enough relevant information to answer "
                "that reliably. Would you like me to try another search query?"
            ),
            sources=[],
            needs_new_search_query=True,
            insufficient_context=True,
        )

    def build_generator_prompt(
        self,
        context: str,
        user_prompt: str,
    ) -> str:
        """Fill the configured generator prompt template."""
        try:
            return self._generator_prompt_template.format(
                user_prompt=user_prompt,
                context=context,
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise GeneratorModelError(
                "The generator prompt template is invalid."
            ) from exc

    def normalize_retrieved_result(
        self,
        raw_result: Any,
    ) -> RetrievedSource:
        """Convert a dictionary or retriever result object into a source."""
        if isinstance(raw_result, RetrievedSource):
            return raw_result

        if isinstance(raw_result, dict):
            return self._normalize_dictionary_result(raw_result)

        return self._normalize_object_result(raw_result)

    def _normalize_dictionary_result(
        self,
        raw_result: Dict[str, Any],
    ) -> RetrievedSource:
        if "id" not in raw_result:
            raise RetrieverCallError(
                "A retrieved dictionary result is missing its 'id'."
            )

        if "chunk" not in raw_result:
            raise RetrieverCallError(
                "A retrieved dictionary result is missing its 'chunk'."
            )

        score = raw_result.get("rrf_score")
        if score is None:
            score = raw_result.get("score")

        metadata = raw_result.get("metadata") or {}

        if not isinstance(metadata, dict):
            raise RetrieverCallError(
                "Retrieved-result metadata must be a dictionary."
            )

        return RetrievedSource(
            id=raw_result["id"],
            chunk=raw_result["chunk"],
            score=score,
            metadata=dict(metadata),
        )

    def _normalize_object_result(
        self,
        raw_result: Any,
    ) -> RetrievedSource:
        if not hasattr(raw_result, "id"):
            raise RetrieverCallError(
                "A retrieved result is missing its 'id' attribute."
            )

        if not hasattr(raw_result, "chunk"):
            raise RetrieverCallError(
                "A retrieved result is missing its 'chunk' attribute."
            )

        score = getattr(raw_result, "rrf_score", None)
        if score is None:
            score = getattr(raw_result, "score", None)

        metadata = getattr(raw_result, "metadata", None) or {}

        if not isinstance(metadata, dict):
            raise RetrieverCallError(
                "Retrieved-result metadata must be a dictionary."
            )

        return RetrievedSource(
            id=getattr(raw_result, "id"),
            chunk=getattr(raw_result, "chunk"),
            score=score,
            metadata=dict(metadata),
        )

    def is_insufficient_context_response(
        self,
        generated_answer: str,
    ) -> bool:
        """
        Return True only when the model returns the exact control token.

        Requiring an exact match prevents ordinary answers that merely mention
        the token from accidentally triggering another search.
        """
        if not isinstance(generated_answer, str):
            return False

        return generated_answer.strip() == INSUFFICIENT_CONTEXT_TOKEN

    def _validate_request(
        self,
        request: IntegratorRequest,
    ) -> None:
        """Validate all public request values before invoking dependencies."""
        self._validate_user_prompt(request.user_prompt)
        self._validate_retrieval_view(request.retrieval_view)

        if (
            not isinstance(request.result_limit, int)
            or isinstance(request.result_limit, bool)
            or request.result_limit <= 0
        ):
            raise IntegratorError(
                "result_limit must be a positive integer."
            )

    def _validate_user_prompt(
        self,
        user_prompt: str,
    ) -> None:
        """Validate that the user prompt is a non-empty string."""
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise EmptyUserPromptError(
                "The user prompt must not be empty."
            )

    def _validate_retrieval_view(
        self,
        retrieval_view: str,
    ) -> None:
        """Validate a plain or schema-qualified PostgreSQL view name."""
        if not isinstance(retrieval_view, str):
            raise InvalidRetrievalViewError(
                "The retrieval view must be a string."
            )

        if not self._RETRIEVAL_VIEW_PATTERN.fullmatch(retrieval_view):
            raise InvalidRetrievalViewError(
                f"Invalid retrieval view: {retrieval_view!r}."
            )