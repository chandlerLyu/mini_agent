"""Local tool execution environment."""

from __future__ import annotations

from dataclasses import dataclass

from config import EnvironmentConfig
from exceptions import ExecutionError, FormatError
from interfaces import ToolCall, ToolDefinition, ToolExecutor, ToolResult
from tools.registry import ToolRegistry, build_default_registry


@dataclass
class LocalEnvironment(ToolExecutor):
    config: EnvironmentConfig
    registry: ToolRegistry

    def __init__(self, config: EnvironmentConfig, registry: ToolRegistry | None = None) -> None:
        self.config = config
        self.registry = registry or build_default_registry()

    def tool_definitions(self) -> list[ToolDefinition]:
        return self.registry.definitions()

    def execute(self, action: ToolCall) -> ToolResult:
        tool = self.registry.get(action.name)
        if tool is None:
            raise FormatError.from_text(f"Unknown tool '{action.name}'.")
        try:
            tool.validate_args(action.arguments)
            return tool.execute(action, self.config)
        except FormatError:
            raise
        except ExecutionError as exc:
            return ToolResult(
                tool_name=action.name,
                tool_call_id=action.id,
                success=False,
                output="",
                return_code=-1,
                error=str(exc),
                metadata={"error_type": type(exc).__name__},
            )
        except Exception as exc:  # pragma: no cover - safety path
            return ToolResult(
                tool_name=action.name,
                tool_call_id=action.id,
                success=False,
                output="",
                return_code=-1,
                error=str(exc),
                metadata={"error_type": type(exc).__name__},
            )
