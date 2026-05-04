# my_agent

`my_agent` is a small coding-agent project built to understand and demonstrate the core architecture behind mini-SWE-agent style systems.

## What it includes

- A linear agent loop in [agent/loop.py](/Users/qchd/coding_project/mini_agent/my_agent/agent/loop.py)
- Typed transcript and state tracking in [agent/state.py](/Users/qchd/coding_project/mini_agent/my_agent/agent/state.py)
- A LiteLLM-backed model adapter plus a deterministic test model
- A local execution environment with `bash`, `read_file`, `write_file`, `search`, and `list_files`
- Trajectory persistence after every step
- Hook extension points for future skills and plugins
- A tiny demo repo and toy evaluation tasks

## Quick start

Install `litellm` if you want to run against a real model:

```bash
pip install litellm
python -m run.cli local --model openai/gpt-4.1-mini --task "Read README.md and summarize the project"
```

When you use the default `--cwd demo_repo`, refer to files inside that directory directly, for example
`calculator.py` rather than `demo_repo/calculator.py`.

List bundled demo tasks:

```bash
python -m run.cli demo --list
```

## Architecture

See [docs/ARCHITECTURE.md](/Users/qchd/coding_project/mini_agent/my_agent/docs/ARCHITECTURE.md).
