"""Agent-specific exceptions."""

from __future__ import annotations

from interfaces import Message


class AgentInterrupt(Exception):
    def __init__(self, *messages: Message) -> None:
        self.messages = list(messages)
        super().__init__()


class FormatError(AgentInterrupt):
    @classmethod
    def from_text(cls, text: str) -> "FormatError":
        return cls(Message(role="user", content=text, metadata={"interrupt_type": "format_error"}))


class LimitsExceeded(Exception):
    pass


class Submitted(Exception):
    def __init__(self, final_answer: str) -> None:
        self.final_answer = final_answer
        super().__init__(final_answer)


class ExecutionError(Exception):
    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(message)


class UserInterrupt(AgentInterrupt):
    pass
