from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import ModelConfig
from interfaces import Message
from models.deterministic import DeterministicModel
from principles.chunker import chunk_documents
from principles.embeddings import HashEmbedding
from principles.extractor import extract_principles_from_chunks
from principles.jsonl import read_jsonl, write_jsonl
from principles.loaders import load_corpus, load_document
from principles.schema import Chunk, PrincipleCandidate
from principles.vector_store import FaissChunkIndex


class PrinciplesPipelineTests(unittest.TestCase):
    def test_loads_text_and_markdown_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("plain text", encoding="utf-8")
            (root / "guide.md").write_text("# Guide\n\nmarkdown text", encoding="utf-8")
            (root / "skip.bin").write_bytes(b"ignored")

            documents = load_corpus(root)

            self.assertEqual([document.source_file for document in documents], ["guide.md", "note.txt"])
            self.assertEqual({document.file_type for document in documents}, {"md", "txt"})

    def test_pdf_loader_reports_malformed_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pdf"
            path.write_text("not a real pdf", encoding="utf-8")

            with self.assertRaises(Exception):
                load_document(path)

    def test_chunker_preserves_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "note.txt"
            source.write_text("alpha beta gamma delta epsilon", encoding="utf-8")
            documents = load_corpus(root)

            chunks = chunk_documents(documents, chunk_size=3, chunk_overlap=1)

            self.assertGreaterEqual(len(chunks), 2)
            self.assertEqual(chunks[0].source_file, "note.txt")
            self.assertEqual(chunks[0].metadata["file_type"], "txt")
            self.assertTrue(chunks[0].chunk_id.startswith("chunk_"))

    def test_jsonl_roundtrip_validates_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.jsonl"
            chunk = Chunk(chunk_id="c1", source_file="a.txt", text="hello", metadata={"file_type": "txt"})

            write_jsonl(path, [chunk])
            loaded = read_jsonl(path, Chunk)

            self.assertEqual(loaded, [chunk])

    def test_faiss_chunk_index_search_returns_metadata(self) -> None:
        chunks = [
            Chunk(chunk_id="c1", source_file="finance.txt", text="market feedback liquidity cycle"),
            Chunk(chunk_id="c2", source_file="code.txt", text="unit tests catch regressions"),
        ]
        embeddings = HashEmbedding(dimensions=24)
        index = FaissChunkIndex.build(chunks, embeddings)

        results = index.search("liquidity market", top_k=1)

        self.assertEqual(results[0].chunk.chunk_id, "c1")
        self.assertEqual(results[0].chunk.source_file, "finance.txt")
        self.assertGreaterEqual(results[0].score, 0.0)

    def test_faiss_chunk_index_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks = [Chunk(chunk_id="c1", source_file="a.txt", text="alpha beta")]
            embeddings = HashEmbedding(dimensions=8)
            index = FaissChunkIndex.build(chunks, embeddings)

            index.save(root / "raw.faiss", root / "raw.json")
            loaded = FaissChunkIndex.load(root / "raw.faiss", root / "raw.json", embeddings)

            self.assertEqual(loaded.search("alpha", top_k=1)[0].chunk.chunk_id, "c1")

    def test_principle_candidate_requires_actionable_fields(self) -> None:
        with self.assertRaises(Exception):
            PrincipleCandidate.model_validate(
                {
                    "principle_id": "P0001",
                    "name": "Too thin",
                    "domain": "general",
                    "summary": "A summary without operational fields.",
                    "when_to_apply": [],
                    "how_to_apply": ["do something"],
                    "failure_modes": ["may fail"],
                    "evidence": [{"source_file": "a.txt", "chunk_id": "c1"}],
                    "confidence": 0.5,
                }
            )

    def test_deterministic_extractor_outputs_candidate_jsonl_ready_objects(self) -> None:
        chunks = [Chunk(chunk_id="c1", source_file="a.txt", text="When feedback loops exist, check limits.")]
        model = DeterministicModel(
            [
                Message(
                    role="assistant",
                    content=(
                        '{"principles":[{"name":"Check feedback limits","domain":"general_reasoning",'
                        '"summary":"When a feedback loop reinforces behavior, identify the constraint that can stop it.",'
                        '"when_to_apply":["reinforcing feedback loops"],'
                        '"how_to_apply":["name the loop","find the limiting constraint"],'
                        '"failure_modes":["assuming reinforcement continues forever"],'
                        '"confidence":0.82}]}'
                    ),
                )
            ]
        )

        candidates = extract_principles_from_chunks(chunks=chunks, model=model, config=ModelConfig())

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].principle_id, "P0001")
        self.assertEqual(candidates[0].status, "candidate")
        self.assertEqual(candidates[0].evidence[0].chunk_id, "c1")


if __name__ == "__main__":
    unittest.main()
