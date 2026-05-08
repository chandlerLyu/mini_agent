from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from principles.embeddings import HashEmbedding
from principles.jsonl import write_jsonl
from principles.memory_store import PrincipleMemoryStore
from principles.schema import Principle
from scripts.build_principle_memory import main as build_memory_main


def principle(
    principle_id: str,
    *,
    summary: str,
    domain: str = "financial_reasoning",
    confidence: float = 0.8,
    status: str = "verified",
) -> Principle:
    return Principle.model_validate(
        {
            "principle_id": principle_id,
            "name": f"Principle {principle_id}",
            "domain": domain,
            "summary": summary,
            "when_to_apply": ["when the situation matches the summary"],
            "how_to_apply": ["use the principle as a reasoning constraint"],
            "failure_modes": ["using the principle outside its scope"],
            "evidence": [{"source_file": "book.pdf", "chunk_id": f"chunk_{principle_id}", "page": 1}],
            "confidence": confidence,
            "status": status,
            "created_at": "2026-05-08T00:00:00+00:00",
            "updated_at": "2026-05-08T00:00:00+00:00",
            "usage_count": 0,
            "success_count": 0,
            "failure_count": 0,
        }
    )


class PrincipleMemoryStoreTests(unittest.TestCase):
    def test_builds_sqlite_rows_and_retrieves_by_semantic_query(self) -> None:
        principles = [
            principle("P0001", summary="market liquidity feedback can reinforce prices"),
            principle("P0002", summary="transparent teams should write narrow tests", domain="coding"),
        ]
        store = PrincipleMemoryStore.build(principles, HashEmbedding(dimensions=32))

        results = store.retrieve_principles("liquidity prices", top_k=1, min_confidence=0.0)

        self.assertEqual(results[0].principle.principle_id, "P0001")
        self.assertGreaterEqual(results[0].score, 0.0)
        self.assertEqual(store.get_principle("P0001").evidence[0].chunk_id, "chunk_P0001")

    def test_retrieval_filters_status_confidence_and_domain(self) -> None:
        principles = [
            principle("P0001", summary="market liquidity feedback", confidence=0.9),
            principle("P0002", summary="market liquidity feedback", confidence=0.4),
            principle("P0003", summary="market liquidity feedback", status="needs_revision"),
            principle("P0004", summary="market liquidity feedback", domain="coding"),
        ]
        store = PrincipleMemoryStore.build(principles, HashEmbedding(dimensions=32))

        results = store.retrieve_principles(
            "market liquidity",
            domain="financial_reasoning",
            top_k=10,
            min_confidence=0.6,
            status="verified",
        )

        self.assertEqual([result.principle.principle_id for result in results], ["P0001"])

    def test_save_and_load_roundtrip_preserves_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings = HashEmbedding(dimensions=24)
            store = PrincipleMemoryStore.build(
                [principle("P0001", summary="beliefs affect financing feedback")],
                embeddings,
            )
            store.save(
                sqlite_path=root / "principles.sqlite",
                index_path=root / "principles.faiss",
                metadata_path=root / "principles_metadata.json",
            )

            loaded = PrincipleMemoryStore.load(
                sqlite_path=root / "principles.sqlite",
                index_path=root / "principles.faiss",
                metadata_path=root / "principles_metadata.json",
                embedding_model=embeddings,
            )

            self.assertEqual(
                loaded.retrieve_principles("financing feedback", min_confidence=0.0)[0].principle.principle_id,
                "P0001",
            )

    def test_usage_records_and_counters_update_sqlite(self) -> None:
        store = PrincipleMemoryStore.build(
            [principle("P0001", summary="feedback requires boundary checks")],
            HashEmbedding(dimensions=16),
        )

        usage_id = store.record_usage(principle_id="P0001", task_id="T1", query="analyze this case")
        store.increment_success("P0001")
        store.increment_failure("P0001")

        updated = store.get_principle("P0001")
        usage_count = store.connection.execute("SELECT COUNT(*) FROM principle_usage").fetchone()[0]
        self.assertTrue(usage_id.startswith("U"))
        self.assertEqual(usage_count, 1)
        self.assertEqual(updated.usage_count, 1)
        self.assertEqual(updated.success_count, 1)
        self.assertEqual(updated.failure_count, 1)

    def test_build_cli_writes_sqlite_faiss_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "principles.jsonl"
            sqlite_path = root / "principles.sqlite"
            index_path = root / "principles.faiss"
            metadata_path = root / "principles_metadata.json"
            write_jsonl(input_path, [principle("P0001", summary="market liquidity feedback")])

            with patch(
                "scripts.build_principle_memory.SentenceTransformerEmbedding",
                lambda _model_name: HashEmbedding(dimensions=16),
            ):
                code = build_memory_main(
                    [
                        "--input",
                        str(input_path),
                        "--sqlite",
                        str(sqlite_path),
                        "--index",
                        str(index_path),
                        "--metadata",
                        str(metadata_path),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertTrue(sqlite_path.exists())
            self.assertTrue(index_path.exists())
            self.assertTrue(metadata_path.exists())
            connection = sqlite3.connect(sqlite_path)
            try:
                count = connection.execute("SELECT COUNT(*) FROM principles").fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
