"""Ingest a corpus into chunks and a FAISS raw chunk index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from principles.chunker import chunk_documents
from principles.embeddings import SentenceTransformerEmbedding
from principles.jsonl import write_jsonl
from principles.loaders import load_corpus
from principles.vector_store import FaissChunkIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest .pdf, .md, and .txt files into raw corpus memory.")
    parser.add_argument("--input-dir", "--input_dir", required=True, help="Directory containing corpus files.")
    parser.add_argument("--output", default="data/processed/chunks.jsonl", help="Output chunk JSONL path.")
    parser.add_argument("--index", default="memory/raw_chunks.faiss", help="Output FAISS index path.")
    parser.add_argument("--metadata", default="memory/raw_chunks_metadata.json", help="Output chunk metadata path.")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunk-size", "--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", "--chunk_overlap", type=int, default=150)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    documents = load_corpus(Path(args.input_dir))
    chunks = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    write_jsonl(Path(args.output), chunks)

    if chunks:
        embeddings = SentenceTransformerEmbedding(args.embedding_model)
        index = FaissChunkIndex.build(chunks, embeddings)
        index.save(Path(args.index), Path(args.metadata))

    print(f"documents={len(documents)}")
    print(f"chunks={len(chunks)}")
    print(f"chunks_path={args.output}")
    if chunks:
        print(f"index_path={args.index}")
        print(f"metadata_path={args.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
