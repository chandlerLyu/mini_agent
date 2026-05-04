"""Default prompt templates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplates:
    system_template: str
    task_template: str
    observation_template: str
    format_error_template: str


DEFAULT_PROMPTS = PromptTemplates(
    system_template=(
        "You are my_agent, a small coding agent. Work step by step, keep context linear, "
        "and either call one or more tools or return <final_answer>...</final_answer> when done."
    ),
    task_template=(
        "Solve the following task inside the current working directory.\n\n"
        "Task: {task}\n\n"
        "Prefer targeted inspection before edits. Use tools when you need information or changes."
    ),
    observation_template=(
        "tool={tool_name}\n"
        "success={success}\n"
        "return_code={return_code}\n"
        "error={error}\n"
        "output:\n{output}"
    ),
    format_error_template=(
        "Format error: {error}\n"
        "You must either call one of the available tools with valid JSON arguments or return "
        "<final_answer>...</final_answer>."
    ),
)
