"""Extract candidate principles from processed chunks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ModelConfig
from models.litellm_client import LiteLLMModel
from principles.extractor import extract_principles_from_chunk
from principles.jsonl import read_jsonl, write_jsonl
from principles.schema import Chunk, PrincipleCandidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract candidate principles from chunk JSONL.")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl", help="Input chunk JSONL path.")
    parser.add_argument("--output", default="memory/principles_candidates.jsonl", help="Output candidate JSONL path.")
    parser.add_argument("--model", required=True, help="LiteLLM model name, e.g. openai/gpt-4.1-mini.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-chunks", "--max_chunks", type=int, default=0, help="Optional cap for demo/debug extraction.")
    parser.add_argument("--verbose", action="store_true", help="Print per-chunk extraction progress to stderr.")
    parser.add_argument("--progress-every", type=int, default=1, help="Verbose progress interval in chunks.")
    parser.add_argument("--errors-output", default="", help="Chunk-level extraction error JSONL path.")
    parser.add_argument("--start-chunk", type=int, default=1, help="1-based chunk number to start from for resume runs.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Write output every N processed chunks.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output instead of appending to it.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chunks = read_jsonl(Path(args.chunks), Chunk)
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
    if args.start_chunk < 1:
        parser.error("--start-chunk must be >= 1")
    if args.start_chunk > len(chunks) + 1:
        parser.error(f"--start-chunk must be <= {len(chunks) + 1}")

    output_path = Path(args.output)
    existing_candidates = [] if args.overwrite else _load_existing_candidates(output_path)
    if args.verbose and existing_candidates:
        _log_progress(f"loaded existing candidates={len(existing_candidates)} from {output_path}")
    config = ModelConfig(model_name=args.model, temperature=args.temperature)
    candidates = _extract_with_progress(
        chunks=chunks[args.start_chunk - 1 :],
        model=LiteLLMModel(),
        config=config,
        verbose=args.verbose,
        progress_every=max(args.progress_every, 1),
        errors_output=_error_path(args.output, args.errors_output),
        output_path=output_path,
        initial_candidates=existing_candidates,
        start_chunk_number=args.start_chunk,
        total_chunks=len(chunks),
        checkpoint_every=max(args.checkpoint_every, 1),
        overwrite_errors=args.overwrite,
    )
    write_jsonl(output_path, candidates)

    print(f"chunks={len(chunks)}")
    print(f"start_chunk={args.start_chunk}")
    print(f"existing_candidates={len(existing_candidates)}")
    print(f"candidates={len(candidates)}")
    print(f"output={args.output}")
    return 0


def _extract_with_progress(
    *,
    chunks: list[Chunk],
    model: LiteLLMModel,
    config: ModelConfig,
    verbose: bool,
    progress_every: int,
    errors_output: Path,
    output_path: Path,
    initial_candidates: list[PrincipleCandidate] | None = None,
    start_chunk_number: int = 1,
    total_chunks: int | None = None,
    checkpoint_every: int = 1,
    overwrite_errors: bool = False,
) -> list[PrincipleCandidate]:
    candidates: list[PrincipleCandidate] = list(initial_candidates or [])
    seen: set[str] = {_dedupe_key(candidate) for candidate in candidates}
    total = total_chunks if total_chunks is not None else len(chunks)
    started_at = time.monotonic()
    errors_output.parent.mkdir(parents=True, exist_ok=True)
    if overwrite_errors or not errors_output.exists():
        errors_output.write_text("", encoding="utf-8")

    for processed, chunk in enumerate(chunks, start=1):
        index = start_chunk_number + processed - 1
        if verbose:
            _log_progress(
                f"extracting chunk {index}/{total} remaining={total - index} "
                f"source={chunk.source_file} page={chunk.page} chunk_id={chunk.chunk_id}"
            )

        try:
            extracted = extract_principles_from_chunk(
                chunk=chunk,
                model=model,
                config=config,
                start_index=len(candidates) + 1,
            )
        except ValueError as exc:
            _record_chunk_error(errors_output, index, chunk, exc)
            if verbose:
                _log_progress(f"skipped chunk {index}/{total} error={exc}")
            continue
        added = 0
        for candidate in extracted:
            key = _dedupe_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(_replace_candidate_id(candidate, f"P{len(candidates) + 1:04d}"))
            added += 1

        if verbose and (index % progress_every == 0 or index == total):
            elapsed = time.monotonic() - started_at
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta = (total - index) / rate if rate > 0 else 0.0
            _log_progress(
                f"done chunk {index}/{total} added={added} total_candidates={len(candidates)} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )

        if processed % checkpoint_every == 0 or index == total:
            write_jsonl(output_path, candidates)

    return candidates


def _replace_candidate_id(candidate: PrincipleCandidate, principle_id: str) -> PrincipleCandidate:
    payload = candidate.model_dump()
    payload["principle_id"] = principle_id
    return PrincipleCandidate.model_validate(payload)


def _dedupe_key(candidate: PrincipleCandidate) -> str:
    return " ".join(f"{candidate.name} {candidate.summary}".lower().split())


def _log_progress(message: str) -> None:
    print(f"[extract_principles] {message}", file=sys.stderr, flush=True)


def _error_path(output: str, errors_output: str) -> Path:
    if errors_output:
        return Path(errors_output)
    return Path(output).with_suffix(".errors.jsonl")


def _record_chunk_error(path: Path, index: int, chunk: Chunk, error: Exception) -> None:
    payload = {
        "chunk_number": index,
        "chunk_id": chunk.chunk_id,
        "source_file": chunk.source_file,
        "page": chunk.page,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _load_existing_candidates(path: Path) -> list[PrincipleCandidate]:
    if not path.exists():
        return []
    return read_jsonl(path, PrincipleCandidate)


if __name__ == "__main__":
    raise SystemExit(main())
