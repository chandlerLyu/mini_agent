"""Verify candidate principles with type-aware structured critique."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ModelConfig
from models.litellm_client import LiteLLMModel
from principles.jsonl import read_jsonl, write_jsonl
from principles.schema import Chunk, Principle, PrincipleCandidate, VerificationRecord
from principles.verifier import build_verified_principle, resolve_evidence_chunks, verify_principle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify extracted candidate principles.")
    parser.add_argument("--input", default="memory/principles_candidates.jsonl", help="Input candidate JSONL path.")
    parser.add_argument("--chunks", default="corpus_embedding/chunks.jsonl", help="Chunk JSONL path for evidence text.")
    parser.add_argument("--output", default="memory/principles.jsonl", help="Verified principle JSONL path.")
    parser.add_argument("--results-output", default="memory/verification_results.jsonl")
    parser.add_argument("--rejected-output", default="memory/rejected_principles.jsonl")
    parser.add_argument("--errors-output", default="memory/verification_errors.jsonl")
    parser.add_argument("--model", required=True, help="LiteLLM model name, e.g. deepseek/deepseek-v4-pro.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Per-LLM-call timeout in seconds.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum verifier response tokens.")
    parser.add_argument("--max-retries", type=int, default=1, help="LiteLLM retry attempts per principle.")
    parser.add_argument("--api-base", default="", help="Optional provider API base URL passed to LiteLLM.")
    parser.add_argument("--api-key-env", default="", help="Optional env var name whose value is passed as api_key.")
    parser.add_argument("--start-principle", type=int, default=1, help="1-based candidate number to start from.")
    parser.add_argument("--max-principles", type=int, default=0, help="Optional cap for demo/debug verification.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Write outputs every N processed candidates.")
    parser.add_argument("--progress-every", type=int, default=1, help="Verbose progress interval in candidates.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs instead of appending.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    candidates = read_jsonl(Path(args.input), PrincipleCandidate)
    if args.max_principles > 0:
        candidates = candidates[: args.max_principles]
    if args.start_principle < 1:
        parser.error("--start-principle must be >= 1")
    if args.start_principle > len(candidates) + 1:
        parser.error(f"--start-principle must be <= {len(candidates) + 1}")

    chunks_by_id = {chunk.chunk_id: chunk for chunk in read_jsonl(Path(args.chunks), Chunk)}
    output_path = Path(args.output)
    results_path = Path(args.results_output)
    rejected_path = Path(args.rejected_output)
    errors_path = Path(args.errors_output)

    verified = [] if args.overwrite else _load_jsonl_if_exists(output_path, Principle)
    results = [] if args.overwrite else _load_jsonl_if_exists(results_path, VerificationRecord)
    rejected = [] if args.overwrite else _load_jsonl_if_exists(rejected_path, VerificationRecord)
    if args.verbose:
        _log_progress(
            f"loaded existing verified={len(verified)} results={len(results)} rejected={len(rejected)}"
        )

    model_kwargs = {"timeout": args.request_timeout, "max_tokens": args.max_tokens}
    if args.api_base:
        model_kwargs["api_base"] = args.api_base
    if args.api_key_env:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            parser.error(f"--api-key-env {args.api_key_env} is not set")
        model_kwargs["api_key"] = api_key

    config = ModelConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
        model_kwargs=model_kwargs,
    )
    state = _verify_with_progress(
        candidates=candidates[args.start_principle - 1 :],
        chunks_by_id=chunks_by_id,
        model=LiteLLMModel(),
        config=config,
        output_path=output_path,
        results_path=results_path,
        rejected_path=rejected_path,
        errors_path=errors_path,
        initial_verified=verified,
        initial_results=results,
        initial_rejected=rejected,
        start_principle_number=args.start_principle,
        total_principles=len(candidates),
        checkpoint_every=max(args.checkpoint_every, 1),
        progress_every=max(args.progress_every, 1),
        verbose=args.verbose,
        overwrite=args.overwrite,
    )

    _write_outputs(output_path, results_path, rejected_path, state)
    print(f"candidates={len(candidates)}")
    print(f"start_principle={args.start_principle}")
    print(f"verified={len(state['verified'])}")
    print(f"results={len(state['results'])}")
    print(f"rejected={len(state['rejected'])}")
    print(f"errors={state['errors']}")
    print(f"output={args.output}")
    return 0


def _verify_with_progress(
    *,
    candidates: list[PrincipleCandidate],
    chunks_by_id: dict[str, Chunk],
    model,
    config: ModelConfig,
    output_path: Path,
    results_path: Path,
    rejected_path: Path,
    errors_path: Path,
    initial_verified: list[Principle],
    initial_results: list[VerificationRecord],
    initial_rejected: list[VerificationRecord],
    start_principle_number: int,
    total_principles: int,
    checkpoint_every: int,
    progress_every: int,
    verbose: bool,
    overwrite: bool,
) -> dict:
    state = {
        "verified": list(initial_verified),
        "results": list(initial_results),
        "rejected": list(initial_rejected),
        "errors": 0,
    }
    seen_verified_ids = {principle.principle_id for principle in state["verified"]}
    seen_result_ids = {record.candidate.principle_id for record in state["results"]}
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not errors_path.exists():
        errors_path.write_text("", encoding="utf-8")
    started_at = time.monotonic()

    for processed, candidate in enumerate(candidates, start=1):
        index = start_principle_number + processed - 1
        if verbose:
            _log_progress(
                f"verifying principle {index}/{total_principles} remaining={total_principles - index} "
                f"principle_id={candidate.principle_id}"
            )

        try:
            evidence_chunks = resolve_evidence_chunks(candidate, chunks_by_id)
            if verbose:
                _log_progress(
                    f"calling model principle_id={candidate.principle_id} evidence_chunks={len(evidence_chunks)}"
                )
            result = verify_principle(candidate=candidate, evidence_chunks=evidence_chunks, model=model, config=config)
            if verbose:
                _log_progress(f"model returned principle_id={candidate.principle_id}")
        except Exception as exc:
            state["errors"] += 1
            _record_error(errors_path, index, candidate, exc)
            if verbose:
                _log_progress(f"skipped principle {index}/{total_principles} error={exc}")
            continue

        record = VerificationRecord(candidate=candidate, result=result)
        if candidate.principle_id not in seen_result_ids:
            state["results"].append(record)
            seen_result_ids.add(candidate.principle_id)

        if result.decision in {"verified", "needs_revision"} and candidate.principle_id not in seen_verified_ids:
            state["verified"].append(build_verified_principle(candidate, result))
            seen_verified_ids.add(candidate.principle_id)
        elif result.decision in {
            "rejected",
            "reclassify_as_observation",
            "reclassify_as_definition",
            "merge_duplicate",
            "requires_more_evidence",
        }:
            state["rejected"].append(record)

        if verbose and (processed % progress_every == 0 or index == total_principles):
            elapsed = time.monotonic() - started_at
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta = (total_principles - index) / rate if rate > 0 else 0.0
            _log_progress(
                f"done principle {index}/{total_principles} decision={result.decision} "
                f"verified={len(state['verified'])} rejected={len(state['rejected'])} "
                f"errors={state['errors']} elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )

        if processed % checkpoint_every == 0 or index == total_principles:
            _write_outputs(output_path, results_path, rejected_path, state)

    return state


def _write_outputs(output_path: Path, results_path: Path, rejected_path: Path, state: dict) -> None:
    write_jsonl(output_path, state["verified"])
    write_jsonl(results_path, state["results"])
    write_jsonl(rejected_path, state["rejected"])


def _load_jsonl_if_exists(path: Path, model):
    if not path.exists():
        return []
    return read_jsonl(path, model)


def _record_error(path: Path, index: int, candidate: PrincipleCandidate, error: Exception) -> None:
    payload = {
        "principle_number": index,
        "principle_id": candidate.principle_id,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _log_progress(message: str) -> None:
    print(f"[verify_principles] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
