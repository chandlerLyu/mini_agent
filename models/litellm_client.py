"""LiteLLM-backed model adapter."""

from __future__ import annotations

import time
from typing import Any

from interfaces import Message, ModelClient, ToolCall, ToolDefinition


class LiteLLMModel(ModelClient):
    def query(self, *, messages: list[Message], tools: list[ToolDefinition], config) -> Message:
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise RuntimeError("litellm is required for LiteLLMModel. Install it with `pip install litellm`.") from exc

        payload = [message.to_model_message() for message in messages]
        tool_payload = [tool.to_openai_tool() for tool in tools]
        attempts = max(config.max_retries, 1)
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                response = litellm.completion(
                    model=config.model_name,
                    messages=payload,
                    tools=tool_payload,
                    temperature=config.temperature,
                    **config.model_kwargs,
                )
                return self._response_to_message(response)
            except Exception as exc:  # pragma: no cover - network/provider dependent
                last_error = exc
                if attempt == attempts - 1:
                    raise
                time.sleep(config.retry_base_delay * (2**attempt))

        assert last_error is not None
        raise last_error

    def _response_to_message(self, response: Any) -> Message:
        choice = response.choices[0].message
        raw_tool_calls = getattr(choice, "tool_calls", None) or []
        tool_calls = [self._convert_tool_call(tool_call) for tool_call in raw_tool_calls]
        content = getattr(choice, "content", "") or ""
        cost = self._extract_cost(response)
        return Message(role="assistant", content=content, tool_calls=tool_calls, metadata={"cost": cost})

    def _convert_tool_call(self, tool_call: Any) -> ToolCall:
        function = getattr(tool_call, "function", None)
        arguments = getattr(function, "arguments", "{}")
        if isinstance(arguments, str):
            import json

            arguments = json.loads(arguments)
        return ToolCall(
            id=getattr(tool_call, "id", "tool-call"),
            name=getattr(function, "name", ""),
            arguments=arguments,
        )

    def _extract_cost(self, response: Any) -> float:
        hidden = getattr(response, "_hidden_params", {}) or {}
        response_cost = hidden.get("response_cost")
        return float(response_cost or 0.0)
