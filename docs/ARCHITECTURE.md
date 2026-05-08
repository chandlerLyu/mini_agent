# Architecture

## Model layer

`models/litellm_client.py` isolates provider-specific details behind a `ModelClient` interface. The agent loop only deals with typed `Message` and `ToolDefinition` objects.

## Tool layer

Tool definitions live separately from execution. The model sees tool schemas from the registry, while `environment/local.py` validates and executes `ToolCall` objects through concrete tool implementations.

The default registry includes local coding tools (`bash`, `read_file`, `write_file`, `search`, and `list_files`) plus `principleAugmented`, which retrieves verified principles from principle memory and returns an augmented reasoning prompt for planning, analysis, and design decisions.

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

## PrincipleRAG memory pipeline

`principles/` contains the offline PrincipleRAG pipeline and principle retrieval layer. It loads `.txt`, `.md`, and text-based `.pdf` files, chunks them with source metadata, writes JSONL records, builds a FAISS raw chunk index with local sentence-transformer embeddings, extracts candidate principles through the existing `ModelClient` interface, and verifies candidates with a type-aware verifier.

Verified principles are stored in `memory/principles.jsonl` for manual inspection and can be transformed into structured principle memory:

- SQLite stores principle metadata, details, and usage records.
- FAISS stores semantic embeddings over principle summaries.
- `PrincipleMemoryStore` retrieves principles with semantic search plus structured filters such as status and confidence.

The current guided-agent bridge is tool-based rather than automatic. `principleAugmented` loads the built principle memory, retrieves relevant verified principles for a query, and formats them as reasoning constraints. The returned prompt tells the model to use only strongly correlated principles and ignore weakly related ones. Answer verification and conservative memory updates are still later milestones.
