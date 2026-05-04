from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from interfaces import Message, ToolCall, ToolDefinition
from models.deterministic import DeterministicModel
from trajectory.store import TrajectoryStore


class ModelAndTrajectoryTests(unittest.TestCase):
    def test_deterministic_model_isolated_behind_interface(self) -> None:
        model = DeterministicModel(
            [Message(role="assistant", content="demo", tool_calls=[ToolCall(id="1", name="list_files", arguments={})])]
        )
        message = model.query(messages=[], tools=[ToolDefinition(name="x", description="y", parameters={})], config=None)
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.tool_calls[0].name, "list_files")

    def test_trajectory_contains_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TrajectoryStore()
            config = AppConfig()
            path = Path(tmp) / "traj.json"
            state = type(
                "DummyState",
                (),
                {
                    "exit_status": "submitted",
                    "final_answer": "answer",
                    "step_count": 1,
                    "model_calls": 1,
                    "total_cost": 0.5,
                    "to_dict": lambda self: {"messages": []},
                },
            )()
            store.save(path, state, config)
            payload = store.load(path)
            self.assertEqual(payload["info"]["status"], "submitted")
            self.assertIn("config", payload)


if __name__ == "__main__":
    unittest.main()
