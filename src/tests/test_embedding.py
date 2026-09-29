from unittest.mock import MagicMock, call, patch

import pytest
import requests

from rag_pipeline.embedding import OllamaEmbeddingModel
from rag_pipeline.integrator import EmbeddingModelError


@pytest.fixture
def response():
    response = MagicMock()
    response.json.return_value = {
        "embedding": [0.1, 0.2, 0.3],
    }
    return response


@pytest.fixture
def model():
    return OllamaEmbeddingModel(
        ollama_base_url="http://ollama.test:11434"
    )


def test_initializes_with_default_base_url():
    model = OllamaEmbeddingModel()

    assert model._ollama_base_url == "http://localhost:11434"
    assert model._requests is requests


def test_initializes_with_custom_base_url():
    model = OllamaEmbeddingModel(
        ollama_base_url="http://ollama.test:9999"
    )

    assert model._ollama_base_url == "http://ollama.test:9999"


def test_embed_posts_expected_request(model, response):
    model._requests.post = MagicMock(return_value=response)

    result = model.embed("How does pgai work?")

    assert result == [0.1, 0.2, 0.3]
    model._requests.post.assert_called_once_with(
        "http://ollama.test:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": "How does pgai work?",
        },
        timeout=30,
    )


def test_embed_checks_http_status(model, response):
    model._requests.post = MagicMock(return_value=response)

    model.embed("Question")

    response.raise_for_status.assert_called_once_with()


def test_embed_parses_response_json(model, response):
    model._requests.post = MagicMock(return_value=response)

    model.embed("Question")

    response.json.assert_called_once_with()


def test_embed_returns_exact_embedding_object(model):
    embedding = [0.4, 0.5, 0.6]
    response = MagicMock()
    response.json.return_value = {"embedding": embedding}
    model._requests.post = MagicMock(return_value=response)

    result = model.embed("Question")

    assert result is embedding


def test_embed_accepts_empty_string(model, response):
    model._requests.post = MagicMock(return_value=response)

    result = model.embed("")

    assert result == [0.1, 0.2, 0.3]
    model._requests.post.assert_called_once_with(
        "http://ollama.test:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": "",
        },
        timeout=30,
    )


@pytest.mark.parametrize(
    "invalid_text",
    [
        None,
        123,
        1.5,
        [],
        {},
        b"bytes",
    ],
)
def test_embed_rejects_non_string_input(model, invalid_text):
    model._requests.post = MagicMock()

    with pytest.raises(
        EmbeddingModelError,
        match="Input text must be a string",
    ):
        model.embed(invalid_text)

    model._requests.post.assert_not_called()


def test_embed_wraps_connection_error(model):
    original_error = requests.ConnectionError("connection refused")
    model._requests.post = MagicMock(side_effect=original_error)

    with pytest.raises(
        EmbeddingModelError,
        match="Failed to call Ollama API: connection refused",
    ) as error:
        model.embed("Question")

    assert error.value.__cause__ is original_error


def test_embed_wraps_timeout(model):
    original_error = requests.Timeout("request timed out")
    model._requests.post = MagicMock(side_effect=original_error)

    with pytest.raises(
        EmbeddingModelError,
        match="Failed to call Ollama API: request timed out",
    ) as error:
        model.embed("Question")

    assert error.value.__cause__ is original_error


def test_embed_wraps_http_error(model):
    original_error = requests.HTTPError("500 Server Error")
    response = MagicMock()
    response.raise_for_status.side_effect = original_error
    model._requests.post = MagicMock(return_value=response)

    with pytest.raises(
        EmbeddingModelError,
        match="Failed to call Ollama API: 500 Server Error",
    ) as error:
        model.embed("Question")

    assert error.value.__cause__ is original_error
    response.json.assert_not_called()


def test_embed_wraps_invalid_json_type_error(model):
    original_error = TypeError("invalid JSON value")
    response = MagicMock()
    response.json.side_effect = original_error
    model._requests.post = MagicMock(return_value=response)

    with pytest.raises(
        EmbeddingModelError,
        match="Invalid response from Ollama API: invalid JSON value",
    ) as error:
        model.embed("Question")

    assert error.value.__cause__ is original_error


def test_embed_wraps_invalid_json_key_error(model):
    original_error = KeyError("invalid key")
    response = MagicMock()
    response.json.side_effect = original_error
    model._requests.post = MagicMock(return_value=response)

    with pytest.raises(
        EmbeddingModelError,
        match="Invalid response from Ollama API",
    ) as error:
        model.embed("Question")

    assert error.value.__cause__ is original_error


def test_embed_reports_missing_embedding_field(model):
    response = MagicMock()
    response.json.return_value = {
        "model": "nomic-embed-text",
    }
    model._requests.post = MagicMock(return_value=response)

    with pytest.raises(
        EmbeddingModelError,
        match=(
            "Unexpected error during embedding: "
            "Ollama API did not return expected embedding field"
        ),
    ) as error:
        model.embed("Question")

    assert isinstance(error.value.__cause__, EmbeddingModelError)
    assert str(error.value.__cause__) == (
        "Ollama API did not return expected embedding field"
    )


def test_embed_wraps_unexpected_response_error(model):
    original_error = ValueError("malformed response")
    response = MagicMock()
    response.json.side_effect = original_error
    model._requests.post = MagicMock(return_value=response)

    with pytest.raises(
        EmbeddingModelError,
        match=(
            "Unexpected error during embedding: malformed response"
        ),
    ) as error:
        model.embed("Question")

    assert error.value.__cause__ is original_error


def test_embed_preserves_trailing_slash_behavior(response):
    model = OllamaEmbeddingModel(
        ollama_base_url="http://ollama.test/"
    )
    model._requests.post = MagicMock(return_value=response)

    model.embed("Question")

    model._requests.post.assert_called_once_with(
        "http://ollama.test//api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": "Question",
        },
        timeout=30,
    )


def test_embed_executes_status_check_before_json_parsing(model):
    events = []

    response = MagicMock()
    response.raise_for_status.side_effect = lambda: events.append(
        "raise_for_status"
    )
    response.json.side_effect = lambda: (
        events.append("json"),
        {"embedding": [0.1]},
    )[1]
    model._requests.post = MagicMock(return_value=response)

    model.embed("Question")

    assert events == ["raise_for_status", "json"]


def test_missing_requests_dependency_raises_import_error():
    real_import = __import__

    def import_without_requests(
        name,
        globals=None,
        locals=None,
        fromlist=(),
        level=0,
    ):
        if name == "requests":
            raise ImportError("requests is unavailable")

        return real_import(
            name,
            globals,
            locals,
            fromlist,
            level,
        )

    with patch("builtins.__import__", side_effect=import_without_requests):
        with pytest.raises(
            ImportError,
            match=(
                "The 'requests' package is required for "
                "OllamaEmbeddingModel"
            ),
        ):
            OllamaEmbeddingModel()