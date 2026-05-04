"""Main agent loop orchestration."""

from __future__ import annotations

import traceback
from pathlib import Path

from agent.hooks import HookManager
from agent.state import AgentState
from config import AppConfig
from exceptions import AgentInterrupt, FormatError, LimitsExceeded, Submitted
from interfaces import AgentRunResult, Message, ModelClient
from trajectory.store import TrajectoryStore


class AgentLoop:
    """Linear agent loop with explicit model, tool, and trajectory layers."""

    def __init__(
        self,
        *,
        model: ModelClient,
        environment,
        config: AppConfig,
        trajectory_store: TrajectoryStore | None = None,
        hooks: HookManager | None = None,
    ) -> None:
        self.model = model
        self.environment = environment
        self.config = config
        self.trajectory_store = trajectory_store or TrajectoryStore()
        self.hooks = hooks or HookManager.with_defaults()

    def run(self, task: str) -> AgentRunResult:
        state = AgentState.from_task(task=task, config=self.config)
        self.hooks.emit("pre_run", state=state, loop=self)
        self._save(state)

        try:
            while not state.is_finished:
                self.step(state)
        except Submitted as exc:
            state.mark_finished(status="submitted", final_answer=exc.final_answer)
            state.add_message(Message(role="assistant", content=exc.final_answer, metadata={"final": True}))
        except AgentInterrupt as exc:
            for message in exc.messages:
                state.add_message(message)
            state.mark_finished(status="interrupted", final_answer="")
        except LimitsExceeded as exc:
            state.mark_finished(status="limits_exceeded", final_answer="")
            state.add_message(Message(role="system", content=str(exc), metadata={"error_type": type(exc).__name__}))
        except Exception as exc:  # pragma: no cover - defensive safety path
            state.mark_finished(status="failed", final_answer="")
            state.add_message(
                Message(
                    role="system",
                    content=str(exc),
                    metadata={
                        "error_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    },
                )
            )
            self.hooks.emit("on_error", state=state, loop=self, error=exc)
            raise
        finally:
            self._save(state)
            self.hooks.emit("on_finish", state=state, loop=self)

        return AgentRunResult(
            task=state.task,
            status=state.exit_status or "submitted",
            final_answer=state.final_answer,
            steps=state.step_count,
            cost=state.total_cost,
            trajectory_path=str(self.config.agent.trajectory_path),
        )

    def step(self, state: AgentState) -> None:
        self._check_limits(state)
        self.hooks.emit("pre_model_call", state=state, loop=self)
        state.latest_actions = []
        state.latest_observations = []

        try:
            assistant_message = self.model.query(
                messages=state.messages,
                tools=self.environment.tool_definitions(),
                config=self.config.model,
            )
        except AgentInterrupt as exc:
            for message in exc.messages:
                state.add_message(message)
            self._finalize_step(state)
            return

        state.record_model_message(assistant_message)
        self.hooks.emit("post_model_call", state=state, loop=self, message=assistant_message)

        final_answer = assistant_message.extract_final_answer()
        if final_answer:
            raise Submitted(final_answer)

        if not assistant_message.tool_calls:
            correction = self.config.agent.format_error_template.format(
                error="No tool calls were produced. Use a tool or provide <final_answer>...</final_answer>."
            )
            state.add_message(Message(role="user", content=correction, metadata={"interrupt_type": "format_error"}))
            self._finalize_step(state)
            return

        state.latest_actions = list(assistant_message.tool_calls)

        for tool_call in assistant_message.tool_calls:
            self.hooks.emit("pre_tool_execution", state=state, loop=self, tool_call=tool_call)
            result = self.environment.execute(tool_call)
            observation = Message.tool_observation(
                tool_name=result.tool_name,
                tool_call_id=result.tool_call_id,
                content=self.config.agent.observation_template.format(
                    tool_name=result.tool_name,
                    success=result.success,
                    output=result.output,
                    return_code=result.return_code,
                    error=result.error,
                ),
                metadata=result.to_dict(),
            )
            state.record_tool_result(result, observation)
            self.hooks.emit("post_tool_execution", state=state, loop=self, result=result, observation=observation)

        self._finalize_step(state)

    def _check_limits(self, state: AgentState) -> None:
        if 0 < self.config.agent.step_limit <= state.step_count:
            raise LimitsExceeded("Agent step limit reached.")
        if 0 < self.config.agent.cost_limit <= state.total_cost:
            raise LimitsExceeded("Agent cost limit reached.")

    def _finalize_step(self, state: AgentState) -> None:
        state.step_count += 1
        self._save(state)
        self.hooks.emit("post_step", state=state, loop=self)

    def _save(self, state: AgentState) -> Path:
        path = self.config.agent.trajectory_path
        self.trajectory_store.save(path, state, self.config)
        return path
