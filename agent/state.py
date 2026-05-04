"""Agent state and transcript handling."""

from __future__ import annotations

from dataclasses import dataclass, field

from interfaces import Message, ToolCall, ToolResult


@dataclass
class AgentState:
    task: str
    messages: list[Message]
    step_count: int = 0
    model_calls: int = 0
    total_cost: float = 0.0
    latest_actions: list[ToolCall] = field(default_factory=list)
    latest_observations: list[Message] = field(default_factory=list)
    exit_status: str = ""
    final_answer: str = ""

    @classmethod
    def from_task(cls, *, task: str, config) -> "AgentState":
        system = Message(role="system", content=config.agent.system_template)
        user = Message(role="user", content=config.agent.task_template.format(task=task))
        return cls(task=task, messages=[system, user])

    @property
    def is_finished(self) -> bool:
        return bool(self.exit_status)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def record_model_message(self, message: Message) -> None:
        self.model_calls += 1
        self.total_cost += float(message.metadata.get("cost", 0.0))
        self.add_message(message)

    def record_tool_result(self, result: ToolResult, observation: Message) -> None:
        self.latest_observations.append(observation)
        self.add_message(observation)

    def mark_finished(self, *, status: str, final_answer: str) -> None:
        self.exit_status = status
        self.final_answer = final_answer

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "step_count": self.step_count,
            "model_calls": self.model_calls,
            "total_cost": self.total_cost,
            "latest_actions": [call.to_dict() for call in self.latest_actions],
            "latest_observations": [message.to_dict() for message in self.latest_observations],
            "exit_status": self.exit_status,
            "final_answer": self.final_answer,
            "messages": [message.to_dict() for message in self.messages],
        }
