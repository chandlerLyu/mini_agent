# my_agent

`my_agent` is a small coding-agent project built to understand and demonstrate the core architecture behind mini-SWE-agent style systems.

## What it includes

- A linear agent loop in [agent/loop.py](/Users/qchd/coding_project/mini_agent/my_agent/agent/loop.py)
- Typed transcript and state tracking in [agent/state.py](/Users/qchd/coding_project/mini_agent/my_agent/agent/state.py)
- A LiteLLM-backed model adapter plus a deterministic test model
- A local execution environment with `bash`, `read_file`, `write_file`, `search`, `list_files`, and `principleAugmented`
- Trajectory persistence after every step
- Hook extension points for future skills and plugins
- A tiny demo repo and toy evaluation tasks
- An offline PrincipleRAG pipeline for ingestion, extraction, verification, storage, and agent-time principle retrieval

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

## PrincipleRAG milestone

The current PrincipleRAG milestone turns a source corpus into verified, searchable principle memory:

```bash
conda activate other
python scripts/ingest_corpus.py \
  --input_dir corpus \
  --output corpus_embedding/chunks.jsonl \
  --index corpus_embedding/raw_chunks.faiss \
  --metadata corpus_embedding/raw_chunks_metadata.json \
  --embedding-model semantic_model/all-MiniLM-L6-v2

python scripts/extract_principles.py \
  --chunks corpus_embedding/chunks.jsonl \
  --output memory/principles_candidates.jsonl \
  --model openai/gpt-4.1-mini \
  --verbose

python scripts/verify_principles.py \
  --input memory/principles_candidates.jsonl \
  --chunks corpus_embedding/chunks.jsonl \
  --output memory/principles.jsonl \
  --results-output memory/verification_results.jsonl \
  --rejected-output memory/rejected_principles.jsonl \
  --errors-output memory/verification_errors.jsonl \
  --model deepseek/deepseek-v4-pro \
  --verbose

python scripts/build_principle_memory.py \
  --input memory/principles.jsonl \
  --sqlite memory/principles.sqlite \
  --index memory/principles.faiss \
  --metadata memory/principles_metadata.json \
  --embedding-model semantic_model/all-MiniLM-L6-v2
```

Ingestion stores raw chunks as JSONL and builds a FAISS raw-corpus index. Extraction uses the existing `ModelClient` abstraction to produce candidate principles with source evidence. Verification filters and revises candidates with type-aware structured critique. `build_principle_memory.py` then turns verified principles into SQLite metadata plus a FAISS principle index.

At agent time, the `principleAugmented` tool retrieves relevant verified principles for a query and returns an augmented reasoning prompt. The model is instructed to use strongly correlated principles as reasoning constraints and ignore weakly related principles.

Run the full test suite with:

```bash
conda run -n other python -m unittest discover -q
```

## Architecture

See [docs/ARCHITECTURE.md](/Users/qchd/coding_project/mini_agent/my_agent/docs/ARCHITECTURE.md).
