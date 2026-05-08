from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import EnvironmentConfig
from environment.local import LocalEnvironment
from interfaces import ToolCall
from principles.embeddings import HashEmbedding
from principles.memory_store import PrincipleMemoryStore
from principles.schema import Principle


class EnvironmentToolTests(unittest.TestCase):
    def test_file_tools_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = LocalEnvironment(EnvironmentConfig(cwd=root))

            write = env.execute(ToolCall(id="1", name="write_file", arguments={"path": "a.txt", "content": "alpha"}))
            self.assertTrue(write.success)

            read = env.execute(ToolCall(id="2", name="read_file", arguments={"path": "a.txt"}))
            self.assertEqual(read.output, "alpha")

            search = env.execute(ToolCall(id="3", name="search", arguments={"query": "alp"}))
            self.assertIn("a.txt", search.output)

    def test_demo_repo_relative_paths_work_with_demo_repo_cwd(self) -> None:
        demo_root = Path(__file__).resolve().parent.parent / "demo_repo"
        env = LocalEnvironment(EnvironmentConfig(cwd=demo_root))
        result = env.execute(ToolCall(id="1", name="read_file", arguments={"path": "calculator.py"}))
        self.assertTrue(result.success)
        self.assertIn("def divide", result.output)

    def test_bash_failure_becomes_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = LocalEnvironment(EnvironmentConfig(cwd=Path(tmp)))
            result = env.execute(ToolCall(id="1", name="bash", arguments={"command": "exit 7"}))
            self.assertFalse(result.success)
            self.assertEqual(result.return_code, 7)

    def test_default_registry_includes_principle_augmented_tool(self) -> None:
        env = LocalEnvironment(EnvironmentConfig(cwd=Path(".")))

        definitions = {definition.name: definition for definition in env.tool_definitions()}

        self.assertIn("principleAugmented", definitions)
        self.assertEqual(definitions["principleAugmented"].parameters["required"], ["query"])
        self.assertIn("planning", definitions["principleAugmented"].description)

    def test_principle_augmented_returns_missing_memory_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = EnvironmentConfig(
                cwd=root,
                principle_sqlite_path=root / "missing.sqlite",
                principle_index_path=root / "missing.faiss",
                principle_metadata_path=root / "missing.json",
            )
            env = LocalEnvironment(config)

            result = env.execute(
                ToolCall(id="1", name="principleAugmented", arguments={"query": "plan a code change"})
            )

            self.assertFalse(result.success)
            self.assertIn("build_principle_memory.py", result.error)
            self.assertEqual(result.metadata["error_type"], "MissingPrincipleMemory")

    def test_principle_augmented_returns_formatted_principle_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _principle_memory_config(root)
            _write_principle_memory(config, [_principle("P0001", "market liquidity feedback can reinforce prices")])
            env = LocalEnvironment(config)

            with patch("principles.embeddings.SentenceTransformerEmbedding", lambda _name: HashEmbedding(32)):
                result = env.execute(
                    ToolCall(id="1", name="principleAugmented", arguments={"query": "analyze liquidity prices"})
                )

            self.assertTrue(result.success)
            self.assertIn("User query:", result.output)
            self.assertIn("P0001", result.output)
            self.assertIn("market liquidity feedback", result.output)
            self.assertIn("Failure modes:", result.output)
            self.assertIn("chunk_P0001", result.output)
            self.assertIn("ignore it and say it was ignored", result.output)

    def test_principle_augmented_reports_no_relevant_principles_after_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _principle_memory_config(root)
            config.principle_status = "verified"
            _write_principle_memory(
                config,
                [_principle("P0001", "market liquidity feedback can reinforce prices", status="needs_revision")],
            )
            env = LocalEnvironment(config)

            with patch("principles.embeddings.SentenceTransformerEmbedding", lambda _name: HashEmbedding(32)):
                result = env.execute(
                    ToolCall(id="1", name="principleAugmented", arguments={"query": "analyze liquidity prices"})
                )

            self.assertTrue(result.success)
            self.assertIn("No relevant principles found.", result.output)


def _principle_memory_config(root: Path) -> EnvironmentConfig:
    return EnvironmentConfig(
        cwd=root,
        principle_sqlite_path=root / "principles.sqlite",
        principle_index_path=root / "principles.faiss",
        principle_metadata_path=root / "principles_metadata.json",
        principle_embedding_model="hash-test",
        principle_min_confidence=0.0,
    )


def _write_principle_memory(config: EnvironmentConfig, principles: list[Principle]) -> None:
    store = PrincipleMemoryStore.build(principles, HashEmbedding(dimensions=32))
    store.save(
        sqlite_path=config.principle_sqlite_path,
        index_path=config.principle_index_path,
        metadata_path=config.principle_metadata_path,
    )


def _principle(principle_id: str, summary: str, *, status: str = "verified") -> Principle:
    return Principle.model_validate(
        {
            "principle_id": principle_id,
            "name": f"Principle {principle_id}",
            "domain": "financial_reasoning",
            "summary": summary,
            "when_to_apply": ["when analyzing similar market dynamics"],
            "how_to_apply": ["trace the feedback loop"],
            "failure_modes": ["assuming the loop continues without constraints"],
            "evidence": [{"source_file": "book.pdf", "chunk_id": f"chunk_{principle_id}", "page": 7}],
            "confidence": 0.8,
            "status": status,
            "created_at": "2026-05-08T00:00:00+00:00",
            "updated_at": "2026-05-08T00:00:00+00:00",
        }
    )


if __name__ == "__main__":
    unittest.main()
