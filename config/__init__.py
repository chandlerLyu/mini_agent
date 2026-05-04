"""Typed application configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from prompts.templates import DEFAULT_PROMPTS


@dataclass
class AgentConfig:
    system_template: str = DEFAULT_PROMPTS.system_template
    task_template: str = DEFAULT_PROMPTS.task_template
    observation_template: str = DEFAULT_PROMPTS.observation_template
    format_error_template: str = DEFAULT_PROMPTS.format_error_template
    step_limit: int = 12
    cost_limit: float = 0.0
    trajectory_path: Path = Path("data/last_run.json")


@dataclass
class ModelConfig:
    model_name: str = ""
    max_retries: int = 3
    retry_base_delay: float = 1.0
    temperature: float = 0.0
    model_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    cwd: Path = Path(".")
    timeout: int = 30
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class AppConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_paths(asdict(self))


def _serialize_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_paths(item) for item in value]
    return value
