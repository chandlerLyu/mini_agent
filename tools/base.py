"""Base tool helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from exceptions import ExecutionError, FormatError
from interfaces import Tool, ToolCall, ToolDefinition, ToolResult


class BaseTool(Tool):
    definition: ToolDefinition

    def validate_args(self, arguments: dict[str, Any]) -> None:
        required = self.definition.parameters.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing:
            raise FormatError.from_text(
                f"Tool '{self.definition.name}' missing required arguments: {', '.join(missing)}."
            )

    def resolve_path(self, cwd: Path, raw_path: str) -> Path:
        root = cwd.resolve()
        candidate = (root / raw_path).resolve()
        if root not in candidate.parents and candidate != root:
            raise ExecutionError(self.definition.name, f"Path '{raw_path}' escapes cwd '{root}'.")
        return candidate

    def result(self, *, action: ToolCall, output: str, success: bool, return_code: int | None = None, error: str = ""):
        return ToolResult(
            tool_name=self.definition.name,
            tool_call_id=action.id,
            success=success,
            output=output,
            return_code=return_code,
            error=error,
        )
