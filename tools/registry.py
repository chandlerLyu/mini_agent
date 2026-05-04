"""Tool registry utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

from interfaces import Tool, ToolDefinition
from tools.builtins import BashTool, ListFilesTool, ReadFileTool, SearchTool, WriteFileTool


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.definition.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition for tool in self.tools.values()]


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in [BashTool(), ReadFileTool(), WriteFileTool(), SearchTool(), ListFilesTool()]:
        registry.register(tool)
    return registry
