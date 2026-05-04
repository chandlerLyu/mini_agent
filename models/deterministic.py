"""Deterministic model for tests and demos."""

from __future__ import annotations

from dataclasses import dataclass, field

from interfaces import Message, ToolDefinition


@dataclass
class DeterministicModel:
    scripted_messages: list[Message] = field(default_factory=list)

    def query(self, *, messages: list[Message], tools: list[ToolDefinition], config) -> Message:
        if not self.scripted_messages:
            return Message(role="assistant", content="<final_answer>No scripted response remaining.</final_answer>")
        return self.scripted_messages.pop(0)
