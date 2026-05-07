from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.loop import AgentLoop
from config import AppConfig
from environment.local import LocalEnvironment
from interfaces import Message, ToolCall
from models.deterministic import DeterministicModel


class AgentLoopTests(unittest.TestCase):
    def test_loop_appends_observations_and_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("hello", encoding="utf-8")
            config = AppConfig()
            config.environment.cwd = root
            config.agent.trajectory_path = root / "traj.json"

            model = DeterministicModel(
                [
                    Message(
                        role="assistant",
                        content="Read the file first.",
                        tool_calls=[ToolCall(id="1", name="read_file", arguments={"path": "note.txt"})],
                    ),
                    Message(role="assistant", content="<final_answer>Done.</final_answer>"),
                ]
            )

            result = AgentLoop(
                model=model,
                environment=LocalEnvironment(config.environment),
                config=config,
            ).run("Inspect the note")

            self.assertEqual(result.status, "submitted")
            self.assertEqual(result.final_answer, "Done.")
            payload = config.agent.trajectory_path.read_text(encoding="utf-8")
            self.assertIn("hello", payload)

    def test_missing_tool_calls_generates_correction_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig()
            config.environment.cwd = root
            config.agent.trajectory_path = root / "traj.json"
            config.agent.step_limit = 2

            model = DeterministicModel(
                [
                    Message(role="assistant", content="I forgot the tool."),
                    Message(role="assistant", content="<final_answer>Recovered.</final_answer>"),
                ]
            )

            result = AgentLoop(
                model=model,
                environment=LocalEnvironment(config.environment),
                config=config,
            ).run("Recover from formatting.")

            self.assertEqual(result.status, "submitted")
            payload = config.agent.trajectory_path.read_text(encoding="utf-8")
            self.assertIn("Format error", payload)

    def test_pre_tool_hook_blocks_rm_rf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "keep.txt").write_text("safe", encoding="utf-8")
            config = AppConfig()
            config.environment.cwd = root
            config.agent.trajectory_path = root / "traj.json"

            model = DeterministicModel(
                [
                    Message(
                        role="assistant",
                        content="Delete everything.",
                        tool_calls=[ToolCall(id="1", name="bash", arguments={"command": "rm -rf ./*"})],
                    ),
                    Message(role="assistant", content="<final_answer>Blocked and continued.</final_answer>"),
                ]
            )

            result = AgentLoop(
                model=model,
                environment=LocalEnvironment(config.environment),
                config=config,
            ).run("Try something destructive.")

            self.assertEqual(result.status, "submitted")
            self.assertTrue((root / "keep.txt").exists())
            payload = config.agent.trajectory_path.read_text(encoding="utf-8")
            self.assertIn("Blocked destructive bash command", payload)
            self.assertIn("blocked_by_hook", payload)


if __name__ == "__main__":
    unittest.main()
