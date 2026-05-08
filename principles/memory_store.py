"""SQLite and FAISS storage for verified principle memory."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from principles.embeddings import EmbeddingModel
from principles.schema import Evidence, Principle


@dataclass(frozen=True)
class PrincipleSearchResult:
    principle: Principle
    score: float


class PrincipleMemoryStore:
    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        index,
        principle_ids: list[str],
        embedding_model: EmbeddingModel,
    ) -> None:
        self.connection = connection
        self.index = index
        self.principle_ids = principle_ids
        self.embedding_model = embedding_model
        self.connection.row_factory = sqlite3.Row

    @classmethod
    def build(cls, principles: list[Principle], embedding_model: EmbeddingModel) -> "PrincipleMemoryStore":
        if not principles:
            raise ValueError("cannot build principle memory without principles")

        import faiss

        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        _create_schema(connection)
        _replace_principles(connection, principles)

        vectors = embedding_model.embed([principle.summary for principle in principles])
        _validate_vectors(vectors, len(principles))
        vectors = _normalize_vectors(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(
            connection=connection,
            index=index,
            principle_ids=[principle.principle_id for principle in principles],
            embedding_model=embedding_model,
        )

    @classmethod
    def load(
        cls,
        *,
        sqlite_path: Path,
        index_path: Path,
        metadata_path: Path,
        embedding_model: EmbeddingModel,
    ) -> "PrincipleMemoryStore":
        import faiss

        connection = sqlite3.connect(sqlite_path)
        connection.row_factory = sqlite3.Row
        index = faiss.read_index(str(index_path))
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        principle_ids = list(payload["principle_ids"])
        return cls(connection=connection, index=index, principle_ids=principle_ids, embedding_model=embedding_model)

    def save(self, *, sqlite_path: Path, index_path: Path, metadata_path: Path) -> None:
        import faiss

        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        destination = sqlite3.connect(sqlite_path)
        try:
            self.connection.backup(destination)
        finally:
            destination.close()

        faiss.write_index(self.index, str(index_path))
        metadata_path.write_text(
            json.dumps({"principle_ids": self.principle_ids}, indent=2),
            encoding="utf-8",
        )

    def retrieve_principles(
        self,
        query: str,
        *,
        domain: str | None = None,
        top_k: int = 5,
        min_confidence: float = 0.6,
        status: str = "verified",
    ) -> list[PrincipleSearchResult]:
        if top_k <= 0 or not self.principle_ids:
            return []

        query_vector = self.embedding_model.embed([query])
        _validate_vectors(query_vector, 1)
        query_vector = _normalize_vectors(query_vector)
        limit = len(self.principle_ids)
        scores, indices = self.index.search(query_vector, limit)

        results: list[PrincipleSearchResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            principle_id = self.principle_ids[int(index)]
            principle = self.get_principle(principle_id)
            if principle is None:
                continue
            if principle.status != status:
                continue
            if principle.confidence < min_confidence:
                continue
            if domain is not None and principle.domain != domain:
                continue
            results.append(PrincipleSearchResult(principle=principle, score=float(score)))
            if len(results) >= top_k:
                break
        return results

    def get_principle(self, principle_id: str) -> Principle | None:
        row = self.connection.execute(
            """
            SELECT
              p.principle_id,
              p.name,
              p.domain,
              p.summary,
              p.confidence,
              p.status,
              p.created_at,
              p.updated_at,
              p.usage_count,
              p.success_count,
              p.failure_count,
              d.when_to_apply,
              d.how_to_apply,
              d.failure_modes,
              d.evidence
            FROM principles p
            JOIN principle_details d ON d.principle_id = p.principle_id
            WHERE p.principle_id = ?
            """,
            (principle_id,),
        ).fetchone()
        if row is None:
            return None
        return _principle_from_row(row)

    def record_usage(
        self,
        *,
        principle_id: str,
        task_id: str,
        query: str,
        agent_answer: str = "",
        verifier_feedback: str = "",
        outcome_score: float | None = None,
        created_at: str | None = None,
    ) -> str:
        if self.get_principle(principle_id) is None:
            raise ValueError(f"unknown principle_id: {principle_id}")
        usage_id = f"U{uuid4().hex}"
        self.connection.execute(
            """
            INSERT INTO principle_usage (
              usage_id,
              principle_id,
              task_id,
              query,
              agent_answer,
              verifier_feedback,
              outcome_score,
              created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, datetime('now')))
            """,
            (usage_id, principle_id, task_id, query, agent_answer, verifier_feedback, outcome_score, created_at),
        )
        self.connection.execute(
            "UPDATE principles SET usage_count = usage_count + 1 WHERE principle_id = ?",
            (principle_id,),
        )
        self.connection.commit()
        return usage_id

    def increment_success(self, principle_id: str) -> None:
        self._increment_counter(principle_id, "success_count")

    def increment_failure(self, principle_id: str) -> None:
        self._increment_counter(principle_id, "failure_count")

    def _increment_counter(self, principle_id: str, column: str) -> None:
        if column not in {"success_count", "failure_count"}:
            raise ValueError(f"unsupported counter: {column}")
        cursor = self.connection.execute(
            f"UPDATE principles SET {column} = {column} + 1 WHERE principle_id = ?",
            (principle_id,),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"unknown principle_id: {principle_id}")
        self.connection.commit()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS principles (
          principle_id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          domain TEXT NOT NULL,
          summary TEXT NOT NULL,
          confidence REAL NOT NULL,
          status TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          usage_count INTEGER NOT NULL DEFAULT 0,
          success_count INTEGER NOT NULL DEFAULT 0,
          failure_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS principle_details (
          principle_id TEXT PRIMARY KEY,
          when_to_apply TEXT NOT NULL,
          how_to_apply TEXT NOT NULL,
          failure_modes TEXT NOT NULL,
          evidence TEXT NOT NULL,
          FOREIGN KEY(principle_id) REFERENCES principles(principle_id)
        );

        CREATE TABLE IF NOT EXISTS principle_usage (
          usage_id TEXT PRIMARY KEY,
          principle_id TEXT NOT NULL,
          task_id TEXT NOT NULL,
          query TEXT NOT NULL,
          agent_answer TEXT,
          verifier_feedback TEXT,
          outcome_score REAL,
          created_at TEXT NOT NULL,
          FOREIGN KEY(principle_id) REFERENCES principles(principle_id)
        );
        """
    )
    connection.commit()


def _replace_principles(connection: sqlite3.Connection, principles: list[Principle]) -> None:
    connection.execute("DELETE FROM principle_usage")
    connection.execute("DELETE FROM principle_details")
    connection.execute("DELETE FROM principles")
    for principle in principles:
        connection.execute(
            """
            INSERT INTO principles (
              principle_id,
              name,
              domain,
              summary,
              confidence,
              status,
              created_at,
              updated_at,
              usage_count,
              success_count,
              failure_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                principle.principle_id,
                principle.name,
                principle.domain,
                principle.summary,
                principle.confidence,
                principle.status,
                principle.created_at,
                principle.updated_at,
                principle.usage_count,
                principle.success_count,
                principle.failure_count,
            ),
        )
        connection.execute(
            """
            INSERT INTO principle_details (
              principle_id,
              when_to_apply,
              how_to_apply,
              failure_modes,
              evidence
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                principle.principle_id,
                _json_dumps(principle.when_to_apply),
                _json_dumps(principle.how_to_apply),
                _json_dumps(principle.failure_modes),
                _json_dumps([item.model_dump() for item in principle.evidence]),
            ),
        )
    connection.commit()


def _principle_from_row(row: sqlite3.Row) -> Principle:
    payload = {
        "principle_id": row["principle_id"],
        "name": row["name"],
        "domain": row["domain"],
        "summary": row["summary"],
        "when_to_apply": json.loads(row["when_to_apply"]),
        "how_to_apply": json.loads(row["how_to_apply"]),
        "failure_modes": json.loads(row["failure_modes"]),
        "evidence": [Evidence.model_validate(item).model_dump() for item in json.loads(row["evidence"])],
        "confidence": row["confidence"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "usage_count": row["usage_count"],
        "success_count": row["success_count"],
        "failure_count": row["failure_count"],
    }
    return Principle.model_validate(payload)


def _json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=True)


def _validate_vectors(vectors: np.ndarray, expected_rows: int) -> None:
    if vectors.dtype != np.float32:
        raise ValueError("embedding vectors must be float32")
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
        raise ValueError("embedding vectors have invalid shape")


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (vectors / norms).astype("float32")
