from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import EnvironmentConfig
from environment.local import LocalEnvironment
from interfaces import ToolCall


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


if __name__ == "__main__":
    unittest.main()
