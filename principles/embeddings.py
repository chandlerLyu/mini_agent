"""Embedding adapters for raw corpus and principle retrieval."""

from __future__ import annotations

from hashlib import sha1
from typing import Protocol

import numpy as np


class EmbeddingModel(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedding:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence-transformers is required for local embeddings") from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")


class HashEmbedding:
    """Small deterministic embedding model for tests."""

    def __init__(self, dimensions: int = 16) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype="float32")
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = sha1(token.encode("utf-8")).hexdigest()
                vectors[row, int(digest[:8], 16) % self.dimensions] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms
