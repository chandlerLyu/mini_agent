"""Build SQLite and FAISS principle memory from verified principle JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from principles.embeddings import SentenceTransformerEmbedding
from principles.jsonl import read_jsonl
from principles.memory_store import PrincipleMemoryStore
from principles.schema import Principle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build searchable principle memory from verified principles.")
    parser.add_argument("--input", default="memory/principles.jsonl", help="Input verified principle JSONL path.")
    parser.add_argument("--sqlite", default="memory/principles.sqlite", help="Output SQLite database path.")
    parser.add_argument("--index", default="memory/principles.faiss", help="Output FAISS index path.")
    parser.add_argument("--metadata", default="memory/principles_metadata.json", help="Output FAISS metadata path.")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--status", default="", help="Optional status filter before building memory.")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Optional confidence floor before indexing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    principles = read_jsonl(Path(args.input), Principle)
    filtered = [
        principle
        for principle in principles
        if (not args.status or principle.status == args.status) and principle.confidence >= args.min_confidence
    ]
    if not filtered:
        raise ValueError("no principles matched the requested filters")

    embeddings = SentenceTransformerEmbedding(args.embedding_model)
    store = PrincipleMemoryStore.build(filtered, embeddings)
    store.save(sqlite_path=Path(args.sqlite), index_path=Path(args.index), metadata_path=Path(args.metadata))

    print(f"principles={len(principles)}")
    print(f"indexed={len(filtered)}")
    print(f"sqlite={args.sqlite}")
    print(f"index={args.index}")
    print(f"metadata={args.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
