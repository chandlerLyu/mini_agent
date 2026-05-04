"""Core public interfaces and typed structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class Message:
    role: str
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def tool_observation(
        cls, *, tool_name: str, tool_call_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> "Message":
        return cls(role="tool", name=tool_name, tool_call_id=tool_call_id, content=content, metadata=metadata or {})

    def to_dict(self) -> dict[str, Any]:
        data = {
            "role": self.role,
            "content": self.content,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "metadata": self.metadata,
        }
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            data["name"] = self.name
        return data

    def to_model_message(self) -> dict[str, Any]:
        payload = {"role": self.role, "content": self.content}
        if self.role == "tool" and self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
            payload["name"] = self.name
        return payload

    def extract_final_answer(self) -> str:
        start_tag = "<final_answer>"
        end_tag = "</final_answer>"
        if start_tag in self.content and end_tag in self.content:
            return self.content.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
        return ""


@dataclass
class ToolResult:
    tool_name: str
    tool_call_id: str
    success: bool
    output: str
    return_code: int | None = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "success": self.success,
            "output": self.output,
            "return_code": self.return_code,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class AgentRunResult:
    task: str
    status: str
    final_answer: str
    steps: int
    cost: float
    trajectory_path: str


class ModelClient(Protocol):
    def query(self, *, messages: list[Message], tools: list[ToolDefinition], config) -> Message: ...


class Tool(Protocol):
    definition: ToolDefinition

    def validate_args(self, arguments: dict[str, Any]) -> None: ...

    def execute(self, action: ToolCall, environment_config) -> ToolResult: ...


class ToolExecutor(Protocol):
    def execute(self, action: ToolCall) -> ToolResult: ...

    def tool_definitions(self) -> list[ToolDefinition]: ...
