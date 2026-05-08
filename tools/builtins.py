"""Built-in tool implementations."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from config import EnvironmentConfig
from exceptions import ExecutionError
from interfaces import ToolCall, ToolDefinition, ToolResult
from principles.memory_store import PrincipleMemoryStore, PrincipleSearchResult
from tools.base import BaseTool


class BashTool(BaseTool):
    definition = ToolDefinition(
        name="bash",
        description="Execute a bash command in the local environment.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command to execute."}},
            "required": ["command"],
        },
    )

    def execute(self, action: ToolCall, environment_config: EnvironmentConfig):
        command = action.arguments["command"]
        cwd = environment_config.cwd.resolve()
        env = os.environ | environment_config.env
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
                timeout=environment_config.timeout,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:
            raise ExecutionError(self.definition.name, str(exc)) from exc
        output = result.stdout + (result.stderr or "")
        return self.result(
            action=action,
            output=output.strip(),
            success=result.returncode == 0,
            return_code=result.returncode,
            error="" if result.returncode == 0 else f"Command exited with {result.returncode}",
        )


class ReadFileTool(BaseTool):
    definition = ToolDefinition(
        name="read_file",
        description="Read a UTF-8 text file inside the working directory.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative path to the file."}},
            "required": ["path"],
        },
    )

    def execute(self, action: ToolCall, environment_config: EnvironmentConfig):
        path = self.resolve_path(environment_config.cwd, action.arguments["path"])
        return self.result(action=action, output=path.read_text(encoding="utf-8"), success=True, return_code=0)


class WriteFileTool(BaseTool):
    definition = ToolDefinition(
        name="write_file",
        description="Write a UTF-8 text file inside the working directory.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file."},
                "content": {"type": "string", "description": "Replacement file content."},
            },
            "required": ["path", "content"],
        },
    )

    def execute(self, action: ToolCall, environment_config: EnvironmentConfig):
        path = self.resolve_path(environment_config.cwd, action.arguments["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(action.arguments["content"], encoding="utf-8")
        return self.result(action=action, output=f"Wrote {path.relative_to(environment_config.cwd.resolve())}", success=True, return_code=0)


class SearchTool(BaseTool):
    definition = ToolDefinition(
        name="search",
        description="Search text inside files under the working directory.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to search for."},
                "path": {"type": "string", "description": "Relative directory or file path.", "default": "."},
            },
            "required": ["query"],
        },
    )

    def execute(self, action: ToolCall, environment_config: EnvironmentConfig):
        query = action.arguments["query"]
        path = self.resolve_path(environment_config.cwd, action.arguments.get("path", "."))
        matches: list[str] = []
        if path.is_file():
            matches.extend(self._search_file(path, query, environment_config.cwd.resolve()))
        else:
            for candidate in path.rglob("*"):
                if candidate.is_file():
                    matches.extend(self._search_file(candidate, query, environment_config.cwd.resolve()))
        output = "\n".join(matches[:200]) if matches else "No matches found."
        return self.result(action=action, output=output, success=True, return_code=0)

    def _search_file(self, path: Path, query: str, root: Path) -> list[str]:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return []
        lines = []
        for index, line in enumerate(text.splitlines(), start=1):
            if query in line:
                lines.append(f"{path.relative_to(root)}:{index}:{line}")
        return lines


class ListFilesTool(BaseTool):
    definition = ToolDefinition(
        name="list_files",
        description="List files under a directory in the working tree.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative directory path.", "default": "."},
                "recursive": {"type": "boolean", "description": "Whether to recurse.", "default": False},
            },
        },
    )

    def execute(self, action: ToolCall, environment_config: EnvironmentConfig):
        base = self.resolve_path(environment_config.cwd, action.arguments.get("path", "."))
        recursive = bool(action.arguments.get("recursive", False))
        if recursive:
            items = sorted(
                str(path.relative_to(environment_config.cwd.resolve()))
                for path in base.rglob("*")
                if path != base
            )
        else:
            items = sorted(str(path.relative_to(environment_config.cwd.resolve())) for path in base.iterdir())
        output = "\n".join(items) if items else "(empty)"
        return self.result(action=action, output=output, success=True, return_code=0)


class PrincipleAugmentedTool(BaseTool):
    definition = ToolDefinition(
        name="principleAugmented",
        description=(
            "Retrieve verified reusable principles to guide reasoning. Strongly consider using this tool near the "
            "start of a task, while analyzing a problem, planning an implementation, making a design decision, or "
            "debugging complex behavior. Use the returned principles as reasoning constraints only when they are "
            "strongly correlated with the current situation; ignore weakly related principles."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of the current task, situation, plan, or decision.",
                }
            },
            "required": ["query"],
        },
    )

    def execute(self, action: ToolCall, environment_config: EnvironmentConfig):
        missing_paths = [
            path
            for path in [
                environment_config.principle_sqlite_path,
                environment_config.principle_index_path,
                environment_config.principle_metadata_path,
            ]
            if not path.exists()
        ]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            return ToolResult(
                tool_name=self.definition.name,
                tool_call_id=action.id,
                success=False,
                output="",
                return_code=-1,
                error=(
                    "Principle memory artifacts are missing. Run scripts/build_principle_memory.py first. "
                    f"Missing: {missing}"
                ),
                metadata={"error_type": "MissingPrincipleMemory"},
            )

        from principles.embeddings import SentenceTransformerEmbedding

        query = action.arguments["query"]
        embeddings = SentenceTransformerEmbedding(environment_config.principle_embedding_model)
        store = PrincipleMemoryStore.load(
            sqlite_path=environment_config.principle_sqlite_path,
            index_path=environment_config.principle_index_path,
            metadata_path=environment_config.principle_metadata_path,
            embedding_model=embeddings,
        )
        results = store.retrieve_principles(
            query,
            top_k=environment_config.principle_top_k,
            min_confidence=environment_config.principle_min_confidence,
            status=environment_config.principle_status,
        )
        output = _format_principle_augmented_prompt(query, results)
        return self.result(action=action, output=output, success=True, return_code=0)


def _format_principle_augmented_prompt(query: str, results: list[PrincipleSearchResult]) -> str:
    if not results:
        return (
            f"User query:\n{query}\n\n"
            "Relevant principles:\nNo relevant principles found.\n\n"
            "Instruction:\nProceed without principle constraints from memory for this query."
        )

    lines = [
        "User query:",
        query,
        "",
        "Relevant principles:",
    ]
    for result in results:
        principle = result.principle
        lines.extend(
            [
                f"- {principle.principle_id}: {principle.name}",
                f"  Domain: {principle.domain}",
                f"  Confidence: {principle.confidence:.2f}",
                f"  Retrieval score: {result.score:.4f}",
                f"  Summary: {principle.summary}",
                f"  When to apply: {_format_list(principle.when_to_apply)}",
                f"  How to apply: {_format_list(principle.how_to_apply)}",
                f"  Failure modes: {_format_list(principle.failure_modes)}",
                f"  Evidence: {_format_evidence(principle.evidence)}",
            ]
        )
    lines.extend(
        [
            "",
            "Instruction:",
            "Use these principles as reasoning constraints, not as decorations.",
            "First decide whether each principle is strongly correlated with the current situation.",
            "If a principle is weakly related, ignore it and say it was ignored because relevance was weak.",
            "For each major claim or decision, explain which strongly relevant principle supports it.",
            "Mention when the principle may not apply.",
        ]
    )
    return "\n".join(lines)


def _format_list(items: list[str]) -> str:
    return "; ".join(items)


def _format_evidence(items) -> str:
    return "; ".join(
        f"source_file={item.source_file}, chunk_id={item.chunk_id}, page={item.page}" for item in items
    )
