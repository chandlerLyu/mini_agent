# Architecture

## Model layer

`models/litellm_client.py` isolates provider-specific details behind a `ModelClient` interface. The agent loop only deals with typed `Message` and `ToolDefinition` objects.

## Tool layer

Tool definitions live separately from execution. The model sees tool schemas from the registry, while `environment/local.py` validates and executes `ToolCall` objects through concrete tool implementations.

## Environment layer

V1 only supports local execution. `bash` uses `subprocess.run`, while file-oriented tools use direct filesystem operations scoped to the configured working directory.

## State and trajectory

`AgentState` stores the full linear transcript, counters, latest actions, latest observations, exit status, and final answer. `TrajectoryStore` serializes the full run after each step.

## Loop lifecycle

1. Create initial system and task messages.
2. Call the model with the current transcript and tool definitions.
3. Parse tool calls or a `<final_answer>...</final_answer>` response.
4. Execute tool calls and append observation messages.
5. Save the trajectory and continue until completion or limits.

## Hooks and future skills

`HookManager` provides lifecycle events such as `pre_run`, `pre_model_call`, `post_tool_execution`, and `on_finish`. Skills are intentionally not implemented yet, but the hook and prompt boundaries are where they should attach later.
