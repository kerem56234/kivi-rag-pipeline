# src/tests/test_generator.py
"""Tests for the Ollama-backed generator model.

We follow the ZOMBIES testing heuristic:

Z - Zero: empty or whitespace-only prompts should be rejected.
O - One: one valid prompt should produce one generated answer.
M - Many: less relevant here because generate() handles one prompt at a time.
B - Boundary: invalid configuration values should be rejected.
I - Interface: the class should expose the generate(prompt) interface expected
    by RAGIntegrator.
E - Exception: HTTP and malformed-response failures should be translated.
S - Simple: verify the normal Ollama request/response flow.

All tests mock HTTP calls so they stay deterministic and do not require a
running Ollama server.
"""

import requests
import pytest

from rag_pipeline.generator import OllamaGeneratorModel
from rag_pipeline.integrator import GeneratorModelError


# ============================================================
# Test helpers
# ============================================================

class FakeSuccessfulResponse:
    """Small fake response object for successful Ollama calls."""

    def __init__(self, payload):
        # Store the JSON payload that json() should later return.
        self._payload = payload

    def raise_for_status(self):
        # Successful HTTP responses do not raise status errors.
        return None

    def json(self):
        # Return the fake JSON body supplied by the test.
        return self._payload


class FakeFailingStatusResponse:
    """Small fake response object whose status check fails."""

    def raise_for_status(self):
        # Simulate requests' behavior for 4xx/5xx responses.
        raise requests.HTTPError("500 Server Error")

    def json(self):
        # This should not matter because raise_for_status fails first.
        return {"response": "should not be used"}


def install_fake_post(monkeypatch, response):
    """Replace requests.post and record all outgoing request arguments."""

    recorded_request = {}

    def fake_post(url, *, json, timeout):
        # Capture all transport details so tests can assert the API contract.
        recorded_request["url"] = url
        recorded_request["json"] = json
        recorded_request["timeout"] = timeout
        return response

    monkeypatch.setattr(requests, "post", fake_post)
    return recorded_request


# ============================================================
# Z - Zero
# ============================================================

@pytest.mark.parametrize("prompt", ["", " ", "\n\t"])
def test_generate_rejects_empty_prompt(prompt):
    """Zero usable prompt content should fail before any HTTP request."""

    generator = OllamaGeneratorModel()

    with pytest.raises(GeneratorModelError, match="prompt must be a non-empty string"):
        generator.generate(prompt)


# ============================================================
# O / S - One simple successful generation
# ============================================================

def test_generate_sends_one_prompt_to_ollama_and_returns_response(monkeypatch):
    """One valid prompt should produce the generated answer text."""

    response = FakeSuccessfulResponse(
        {"response": "PostgreSQL stores the embeddings."}
    )
    recorded_request = install_fake_post(monkeypatch, response)

    generator = OllamaGeneratorModel(
        ollama_base_url="http://ollama.test:11434",
        model="test-generator",
        timeout=12.5,
    )

    answer = generator.generate("Question and retrieved context")

    assert answer == "PostgreSQL stores the embeddings."
    assert recorded_request == {
        "url": "http://ollama.test:11434/api/generate",
        "json": {
            "model": "test-generator",
            "prompt": "Question and retrieved context",
            "stream": False,
        },
        "timeout": 12.5,
    }


# ============================================================
# B - Boundary / configuration validation
# ============================================================

@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"ollama_base_url": ""},
            "ollama_base_url must be a non-empty string",
        ),
        (
            {"ollama_base_url": "   "},
            "ollama_base_url must be a non-empty string",
        ),
        (
            {"model": ""},
            "model must be a non-empty string",
        ),
        (
            {"model": "   "},
            "model must be a non-empty string",
        ),
        (
            {"timeout": 0},
            "timeout must be a positive number",
        ),
        (
            {"timeout": -1},
            "timeout must be a positive number",
        ),
    ],
)
def test_constructor_rejects_invalid_configuration(arguments, message):
    """Invalid adapter configuration should fail immediately."""

    with pytest.raises(ValueError, match=message):
        OllamaGeneratorModel(**arguments)


def test_generate_accepts_prompt_with_surrounding_whitespace(monkeypatch):
    """Boundary case: whitespace around meaningful text is still valid."""

    response = FakeSuccessfulResponse({"response": "A valid answer"})
    recorded_request = install_fake_post(monkeypatch, response)

    generator = OllamaGeneratorModel()

    answer = generator.generate("  meaningful prompt  ")

    assert answer == "A valid answer"
    assert recorded_request["json"]["prompt"] == "  meaningful prompt  "


# ============================================================
# I - Interface expected by RAGIntegrator
# ============================================================

def test_generator_exposes_generate_method():
    """The object should satisfy the GeneratorModel protocol structurally."""

    generator = OllamaGeneratorModel()

    assert callable(generator.generate)


# ============================================================
# E - Exceptions
# ============================================================

def test_generate_wraps_network_errors(monkeypatch):
    """Transport-level request failures should become GeneratorModelError."""

    def fake_post(url, *, json, timeout):
        raise requests.ConnectionError("Ollama is unavailable")

    monkeypatch.setattr(requests, "post", fake_post)

    generator = OllamaGeneratorModel()

    with pytest.raises(
        GeneratorModelError,
        match="Failed to call Ollama API",
    ) as error:
        generator.generate("Valid prompt")

    assert isinstance(error.value.__cause__, requests.ConnectionError)


def test_generate_wraps_http_status_errors(monkeypatch):
    """HTTP error responses should become GeneratorModelError."""

    response = FakeFailingStatusResponse()
    install_fake_post(monkeypatch, response)

    generator = OllamaGeneratorModel()

    with pytest.raises(
        GeneratorModelError,
        match="Failed to call Ollama API",
    ) as error:
        generator.generate("Valid prompt")

    assert isinstance(error.value.__cause__, requests.HTTPError)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"response": None},
        {"response": 123},
    ],
)
def test_generate_rejects_missing_or_non_string_response(monkeypatch, payload):
    """Malformed Ollama JSON should not leak as a successful generation."""

    response = FakeSuccessfulResponse(payload)
    install_fake_post(monkeypatch, response)

    generator = OllamaGeneratorModel()

    with pytest.raises(
        GeneratorModelError,
        match="Ollama API did not return a valid response field",
    ):
        generator.generate("Valid prompt")


def test_generate_wraps_invalid_json_response(monkeypatch):
    """Invalid JSON parsing should become GeneratorModelError."""

    class InvalidJsonResponse:
        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("invalid json")

    install_fake_post(monkeypatch, InvalidJsonResponse())

    generator = OllamaGeneratorModel()

    with pytest.raises(
        GeneratorModelError,
        match="Invalid response from Ollama API",
    ) as error:
        generator.generate("Valid prompt")

    assert isinstance(error.value.__cause__, ValueError)