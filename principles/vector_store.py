"""FAISS-backed vector index for raw corpus chunks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from principles.embeddings import EmbeddingModel
from principles.schema import Chunk


@dataclass(frozen=True)
class ChunkSearchResult:
    chunk: Chunk
    score: float


class FaissChunkIndex:
    def __init__(self, *, index, chunks: list[Chunk], embedding_model: EmbeddingModel) -> None:
        self.index = index
        self.chunks = chunks
        self.embedding_model = embedding_model

    @classmethod
    def build(cls, chunks: list[Chunk], embedding_model: EmbeddingModel) -> "FaissChunkIndex":
        if not chunks:
            raise ValueError("cannot build FAISS index without chunks")
        import faiss

        vectors = embedding_model.embed([chunk.text for chunk in chunks])
        _validate_vectors(vectors, len(chunks))
        vectors = _normalize_vectors(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index=index, chunks=chunks, embedding_model=embedding_model)

    def search(self, query: str, *, top_k: int = 5) -> list[ChunkSearchResult]:
        if top_k <= 0:
            return []
        query_vector = self.embedding_model.embed([query])
        _validate_vectors(query_vector, 1)
        query_vector = _normalize_vectors(query_vector)
        scores, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))
        results: list[ChunkSearchResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append(ChunkSearchResult(chunk=self.chunks[int(index)], score=float(score)))
        return results

    def save(self, index_path: Path, metadata_path: Path) -> None:
        import faiss

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        payload = [chunk.model_dump() for chunk in self.chunks]
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path, embedding_model: EmbeddingModel) -> "FaissChunkIndex":
        import faiss

        index = faiss.read_index(str(index_path))
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        chunks = [Chunk.model_validate(item) for item in payload]
        return cls(index=index, chunks=chunks, embedding_model=embedding_model)


def _validate_vectors(vectors: np.ndarray, expected_rows: int) -> None:
    if vectors.dtype != np.float32:
        raise ValueError("embedding vectors must be float32")
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
        raise ValueError("embedding vectors have invalid shape")


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (vectors / norms).astype("float32")
