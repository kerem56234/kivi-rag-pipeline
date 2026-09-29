from __future__ import annotations

from typing import Sequence

from rag_pipeline.integrator import EmbeddingModelError

class OllamaEmbeddingModel:
    """
    Implementation of EmbeddingModel that uses Ollama API with nomic-embed-text model.
    """

    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama embedding model.

        Args:
            ollama_base_url: Base URL for the Ollama API (default: http://localhost:11434)
        """
        self._ollama_base_url = ollama_base_url
        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError(
                "The 'requests' package is required for OllamaEmbeddingModel. Install it with 'pip install requests'")

    def embed(self, text: str) -> Sequence[float]:
        """
        Generate embedding for the given text using Ollama's nomic-embed-text model.

        Args:
            text: Text to embed

        Returns:
            List of float values representing the embedding vector

        Raises:
            EmbeddingModelError: If the embedding fails
        """
        if not isinstance(text, str):
            raise EmbeddingModelError("Input text must be a string")

        try:
            # Prepare the request payload
            payload = {
                "model": "nomic-embed-text",
                "prompt": text
            }

            # Make the API call to Ollama
            response = self._requests.post(
                f"{self._ollama_base_url}/api/embeddings",
                json=payload,
                timeout=30  # 30 second timeout
            )

            # Check if the request was successful
            response.raise_for_status()

            # Parse the response
            result = response.json()

            # Extract and return the embedding vector
            if "embedding" not in result:
                raise EmbeddingModelError("Ollama API did not return expected embedding field")

            return result["embedding"]

        except self._requests.exceptions.RequestException as exc:
            raise EmbeddingModelError(f"Failed to call Ollama API: {str(exc)}") from exc
        except (KeyError, TypeError) as exc:
            raise EmbeddingModelError(f"Invalid response from Ollama API: {str(exc)}") from exc
        except Exception as exc:
            raise EmbeddingModelError(f"Unexpected error during embedding: {str(exc)}") from exc