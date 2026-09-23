# tests/test_integrator.py

from dataclasses import dataclass

import pytest

from integrator import (
    DEFAULT_GENERATOR_SYSTEM_PROMPT,
    INSUFFICIENT_CONTEXT_TOKEN,
    EmptyUserPromptError,
    IntegratorResponse,
    InvalidRetrievalViewError,
    RAGIntegrator,
    RetrievedSource,
)


# ============================================================
# Test doubles
# ============================================================

class FakeEmbeddingModel:
    def __init__(self, vector=None):
        self.vector = vector if vector is not None else [0.1] * 768
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return self.vector


class FakeRetriever:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    def retrieve(
        self,
        search_vector,
        relation,
        result_limit=5,
        query_text=None,
    ):
        self.calls.append(
            {
                "search_vector": search_vector,
                "relation": relation,
                "result_limit": result_limit,
                "query_text": query_text,
            }
        )
        return self.results[:result_limit]


class FakeGeneratorModel:
    def __init__(self, response="Generated answer"):
        self.response = response
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return self.response


@dataclass
class FakeRetrieverResult:
    id: int
    chunk: str
    rrf_score: float
    metadata: dict


def create_integrator(
    *,
    embedding_vector=None,
    retrieval_results=None,
    generator_response="Generated answer",
    prompt_template=DEFAULT_GENERATOR_SYSTEM_PROMPT,
):
    embedding_model = FakeEmbeddingModel(embedding_vector)
    retriever = FakeRetriever(retrieval_results)
    generator_model = FakeGeneratorModel(generator_response)

    integrator = RAGIntegrator(
        embedding_model=embedding_model,
        retriever=retriever,
        generator_model=generator_model,
        generator_prompt_template=prompt_template,
    )

    return integrator, embedding_model, retriever, generator_model


# ============================================================
# embed_and_retrieve
# ============================================================

def test_embed_and_retrieve_embeds_the_original_user_prompt():
    integrator, embedding_model, _, _ = create_integrator()

    integrator.embed_and_retrieve(
        user_prompt="How does pgai work?",
        retrieval_view="wiki_embedding",
    )

    assert embedding_model.calls == ["How does pgai work?"]


def test_embed_and_retrieve_passes_all_arguments_to_retriever():
    vector = [0.25] * 768
    integrator, _, retriever, _ = create_integrator(
        embedding_vector=vector,
    )

    integrator.embed_and_retrieve(
        user_prompt="How does pgai work?",
        retrieval_view="public.wiki_embedding",
        result_limit=8,
    )

    assert retriever.calls == [
        {
            "search_vector": vector,
            "relation": "public.wiki_embedding",
            "result_limit": 8,
            "query_text": "How does pgai work?",
        }
    ]


def test_embed_and_retrieve_normalizes_retriever_results():
    raw_results = [
        FakeRetrieverResult(
            id=11,
            chunk="First chunk",
            rrf_score=0.032,
            metadata={"title": "Document A"},
        ),
        FakeRetrieverResult(
            id=12,
            chunk="Second chunk",
            rrf_score=0.021,
            metadata={"title": "Document B"},
        ),
    ]
    integrator, _, _, _ = create_integrator(
        retrieval_results=raw_results,
    )

    results = integrator.embed_and_retrieve(
        user_prompt="question",
        retrieval_view="wiki_embedding",
    )

    assert results == [
        RetrievedSource(
            id=11,
            chunk="First chunk",
            score=0.032,
            metadata={"title": "Document A"},
        ),
        RetrievedSource(
            id=12,
            chunk="Second chunk",
            score=0.021,
            metadata={"title": "Document B"},
        ),
    ]


def test_embed_and_retrieve_uses_default_result_limit_of_five():
    integrator, _, retriever, _ = create_integrator()

    integrator.embed_and_retrieve(
        user_prompt="question",
        retrieval_view="wiki_embedding",
    )

    assert retriever.calls[0]["result_limit"] == 5


# ============================================================
# Result normalization
# ============================================================

def test_normalize_retrieved_result_maps_rrf_score_to_score():
    integrator, _, _, _ = create_integrator()
    raw_result = FakeRetrieverResult(
        id=42,
        chunk="Relevant information",
        rrf_score=0.05,
        metadata={"document_id": "doc-7"},
    )

    result = integrator.normalize_retrieved_result(raw_result)

    assert result == RetrievedSource(
        id=42,
        chunk="Relevant information",
        score=0.05,
        metadata={"document_id": "doc-7"},
    )


def test_normalize_retrieved_result_accepts_dictionary_results():
    integrator, _, _, _ = create_integrator()
    raw_result = {
        "id": "chunk-1",
        "chunk": "Dictionary result",
        "rrf_score": 0.04,
        "metadata": {"title": "Example"},
    }

    result = integrator.normalize_retrieved_result(raw_result)

    assert result == RetrievedSource(
        id="chunk-1",
        chunk="Dictionary result",
        score=0.04,
        metadata={"title": "Example"},
    )


# ============================================================
# Context formation
# ============================================================

def test_form_context_includes_source_ids_and_chunks():
    integrator, _, _, _ = create_integrator()
    sources = [
        RetrievedSource(id=7, chunk="The first relevant passage."),
        RetrievedSource(id=9, chunk="The second relevant passage."),
    ]

    context = integrator.form_context(sources)

    assert "7" in context
    assert "The first relevant passage." in context
    assert "9" in context
    assert "The second relevant passage." in context


def test_form_context_preserves_result_order():
    integrator, _, _, _ = create_integrator()
    sources = [
        RetrievedSource(id="first", chunk="First passage"),
        RetrievedSource(id="second", chunk="Second passage"),
    ]

    context = integrator.form_context(sources)

    assert context.index("first") < context.index("second")
    assert context.index("First passage") < context.index("Second passage")


def test_form_context_returns_empty_string_for_no_results():
    integrator, _, _, _ = create_integrator()

    assert integrator.form_context([]) == ""


# ============================================================
# Generator prompt
# ============================================================

def test_build_generator_prompt_includes_question_and_context():
    integrator, _, _, _ = create_integrator()

    prompt = integrator.build_generator_prompt(
        context="[ID: 5]\nPostgreSQL stores the embeddings.",
        user_prompt="Where are embeddings stored?",
    )

    assert "Where are embeddings stored?" in prompt
    assert "[ID: 5]" in prompt
    assert "PostgreSQL stores the embeddings." in prompt


def test_build_generator_prompt_includes_insufficient_context_instruction():
    integrator, _, _, _ = create_integrator()

    prompt = integrator.build_generator_prompt(
        context="[ID: 1]\nSome context",
        user_prompt="A question",
    )

    assert INSUFFICIENT_CONTEXT_TOKEN in prompt
    assert "only the provided context" in prompt.lower()
    assert "do not use outside knowledge" in prompt.lower()


def test_build_generator_prompt_uses_injected_template():
    template = "QUESTION={user_prompt}\nDATA={context}"
    integrator, _, _, _ = create_integrator(prompt_template=template)

    prompt = integrator.build_generator_prompt(
        context="context value",
        user_prompt="question value",
    )

    assert prompt == "QUESTION=question value\nDATA=context value"


def test_call_generator_sends_built_prompt_to_model():
    integrator, _, _, generator = create_integrator(
        generator_response="The grounded answer"
    )

    answer = integrator.call_generator(
        context="[ID: 4]\nRelevant text",
        user_prompt="What is relevant?",
    )

    assert answer == "The grounded answer"
    assert len(generator.calls) == 1
    assert "What is relevant?" in generator.calls[0]
    assert "[ID: 4]" in generator.calls[0]


# ============================================================
# Insufficient-context detection
# ============================================================

@pytest.mark.parametrize(
    "response",
    [
        "INSUFFICIENT_CONTEXT",
        " INSUFFICIENT_CONTEXT ",
        "\nINSUFFICIENT_CONTEXT\n",
    ],
)
def test_detects_exact_insufficient_context_response(response):
    integrator, _, _, _ = create_integrator()

    assert integrator.is_insufficient_context_response(response) is True


@pytest.mark.parametrize(
    "response",
    [
        "There is sufficient context.",
        "The token INSUFFICIENT_CONTEXT is mentioned in this answer.",
        "INSUFFICIENT_CONTEXT but here is an answer anyway.",
        "",
    ],
)
def test_does_not_treat_non_exact_response_as_insufficient(response):
    integrator, _, _, _ = create_integrator()

    assert integrator.is_insufficient_context_response(response) is False


# ============================================================
# Complete orchestration
# ============================================================

def test_answer_runs_complete_pipeline_and_returns_answer_with_sources():
    raw_results = [
        FakeRetrieverResult(
            id=3,
            chunk="pgai synchronizes vector embeddings.",
            rrf_score=0.03,
            metadata={"title": "pgai"},
        )
    ]
    integrator, embedding, retriever, generator = create_integrator(
        retrieval_results=raw_results,
        generator_response="pgai synchronizes embeddings using a vectorizer.",
    )

    response = integrator.answer(
        user_prompt="What does pgai synchronize?",
        retrieval_view="wiki_embedding",
        result_limit=4,
    )

    assert embedding.calls == ["What does pgai synchronize?"]
    assert retriever.calls[0]["relation"] == "wiki_embedding"
    assert retriever.calls[0]["query_text"] == "What does pgai synchronize?"
    assert retriever.calls[0]["result_limit"] == 4
    assert len(generator.calls) == 1

    assert response == IntegratorResponse(
        answer="pgai synchronizes embeddings using a vectorizer.",
        sources=[
            RetrievedSource(
                id=3,
                chunk="pgai synchronizes vector embeddings.",
                score=0.03,
                metadata={"title": "pgai"},
            )
        ],
        needs_new_search_query=False,
        insufficient_context=False,
    )


def test_answer_does_not_call_generator_when_retriever_returns_no_results():
    integrator, embedding, retriever, generator = create_integrator(
        retrieval_results=[]
    )

    response = integrator.answer(
        user_prompt="What is not in the database?",
        retrieval_view="wiki_embedding",
    )

    assert len(embedding.calls) == 1
    assert len(retriever.calls) == 1
    assert generator.calls == []

    assert response.needs_new_search_query is True
    assert response.insufficient_context is True
    assert response.sources == []
    assert "another search" in response.answer.lower()


def test_answer_asks_for_new_query_when_generator_rejects_context():
    raw_results = [
        FakeRetrieverResult(
            id=20,
            chunk="Retrieved but semantically irrelevant text.",
            rrf_score=0.01,
            metadata={},
        )
    ]
    integrator, _, _, generator = create_integrator(
        retrieval_results=raw_results,
        generator_response=INSUFFICIENT_CONTEXT_TOKEN,
    )

    response = integrator.answer(
        user_prompt="Question unrelated to the retrieved text",
        retrieval_view="wiki_embedding",
    )

    assert len(generator.calls) == 1
    assert response.needs_new_search_query is True
    assert response.insufficient_context is True
    assert "another search" in response.answer.lower()


def test_answer_keeps_sources_when_generator_rejects_context():
    raw_results = [
        FakeRetrieverResult(
            id=20,
            chunk="Semantically irrelevant text.",
            rrf_score=0.01,
            metadata={},
        )
    ]
    integrator, _, _, _ = create_integrator(
        retrieval_results=raw_results,
        generator_response=INSUFFICIENT_CONTEXT_TOKEN,
    )

    response = integrator.answer(
        user_prompt="An unrelated question",
        retrieval_view="wiki_embedding",
    )

    assert response.sources == [
        RetrievedSource(
            id=20,
            chunk="Semantically irrelevant text.",
            score=0.01,
            metadata={},
        )
    ]


# ============================================================
# Rendering
# ============================================================

def test_render_answer_returns_answer_and_sources():
    integrator, _, _, _ = create_integrator()
    sources = [
        RetrievedSource(
            id=1,
            chunk="Source text",
            score=0.03,
            metadata={"title": "Source"},
        )
    ]

    response = integrator.render_answer(
        generated_answer="Rendered answer",
        retrieved_sources=sources,
    )

    assert response == IntegratorResponse(
        answer="Rendered answer",
        sources=sources,
        needs_new_search_query=False,
        insufficient_context=False,
    )


def test_ask_for_new_search_query_sets_control_flags():
    integrator, _, _, _ = create_integrator()

    response = integrator.ask_for_new_search_query()

    assert response.needs_new_search_query is True
    assert response.insufficient_context is True
    assert response.sources == []
    assert response.answer.strip()
    assert "another search" in response.answer.lower()


# ============================================================
# Validation
# ============================================================

@pytest.mark.parametrize("user_prompt", ["", " ", "\n\t"])
def test_answer_rejects_empty_user_prompt(user_prompt):
    integrator, embedding, retriever, generator = create_integrator()

    with pytest.raises(EmptyUserPromptError):
        integrator.answer(
            user_prompt=user_prompt,
            retrieval_view="wiki_embedding",
        )

    assert embedding.calls == []
    assert retriever.calls == []
    assert generator.calls == []


@pytest.mark.parametrize(
    "retrieval_view",
    [
        "",
        " ",
        "wiki embedding",
        "wiki_embedding;",
        "wiki_embedding; DROP TABLE users;",
    ],
)
def test_answer_rejects_invalid_retrieval_view(retrieval_view):
    integrator, embedding, retriever, generator = create_integrator()

    with pytest.raises(InvalidRetrievalViewError):
        integrator.answer(
            user_prompt="Valid question",
            retrieval_view=retrieval_view,
        )

    assert embedding.calls == []
    assert retriever.calls == []
    assert generator.calls == []


@pytest.mark.parametrize(
    "retrieval_view",
    [
        "wiki_embedding",
        "public.wiki_embedding",
        "rag_chunks_v1",
    ],
)
def test_answer_accepts_valid_retrieval_view(retrieval_view):
    integrator, _, retriever, _ = create_integrator(retrieval_results=[])

    integrator.answer(
        user_prompt="Valid question",
        retrieval_view=retrieval_view,
    )

    assert retriever.calls[0]["relation"] == retrieval_view