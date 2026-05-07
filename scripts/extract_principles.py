"""Extract candidate principles from processed chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ModelConfig
from models.litellm_client import LiteLLMModel
from principles.extractor import extract_principles_from_chunks
from principles.jsonl import read_jsonl, write_jsonl
from principles.schema import Chunk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract candidate principles from chunk JSONL.")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl", help="Input chunk JSONL path.")
    parser.add_argument("--output", default="memory/principles_candidates.jsonl", help="Output candidate JSONL path.")
    parser.add_argument("--model", required=True, help="LiteLLM model name, e.g. openai/gpt-4.1-mini.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-chunks", "--max_chunks", type=int, default=0, help="Optional cap for demo/debug extraction.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chunks = read_jsonl(Path(args.chunks), Chunk)
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]

    config = ModelConfig(model_name=args.model, temperature=args.temperature)
    candidates = extract_principles_from_chunks(chunks=chunks, model=LiteLLMModel(), config=config)
    write_jsonl(Path(args.output), candidates)

    print(f"chunks={len(chunks)}")
    print(f"candidates={len(candidates)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
