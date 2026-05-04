"""Hook registration and sample hooks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from utils.logging import get_logger

Hook = Callable[..., None]
LOGGER = get_logger("hooks")


class HookManager:
    def __init__(self) -> None:
        self._hooks: dict[str, list[Hook]] = defaultdict(list)

    @classmethod
    def with_defaults(cls) -> "HookManager":
        manager = cls()
        manager.register("post_step", logging_hook)
        manager.register("pre_run", noop_hook)
        return manager

    def register(self, event: str, hook: Hook) -> None:
        self._hooks[event].append(hook)

    def emit(self, event: str, **kwargs: Any) -> None:
        for hook in self._hooks.get(event, []):
            hook(event=event, **kwargs)


def noop_hook(**_: Any) -> None:
    """Placeholder hook proving extension points exist."""


def logging_hook(*, event: str, state, **_: Any) -> None:
    LOGGER.debug("hook=%s step=%s messages=%s", event, state.step_count, len(state.messages))
