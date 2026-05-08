"""LLM-backed candidate principle extraction."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from config import ModelConfig
from interfaces import Message, ModelClient
from principles.schema import Evidence, PrincipleCandidate


EXTRACT_SYSTEM_PROMPT = """You extract reusable principles from source text.

A principle is not a summary. It must be a general, actionable rule that can guide future reasoning beyond the passage.

Consider the following 6 questions when extracting:

1. What reusable principle is implied by this text?
2. Is it general enough to apply beyond this passage?
3. When should it be applied?
4. How should it be applied?
5. When might it fail?
6. What source evidence supports it?

Return strict JSON only with this shape:
{"principles":[{"name":"...","domain":"...","summary":"...","when_to_apply":["..."],"how_to_apply":["..."],"failure_modes":["..."],"confidence":0.0}]}
Do not include markdown fences.
"""


def extract_principles_from_chunk(
    *,
    chunk,
    model: ModelClient,
    config: ModelConfig,
    id_prefix: str = "P",
    start_index: int = 1,
) -> list[PrincipleCandidate]:
    messages = [
        Message(role="system", content=EXTRACT_SYSTEM_PROMPT),
        Message(
            role="user",
            content=(
                "Extract candidate principles from this chunk. "
                "Each candidate must be reusable, actionable, evidence-grounded, and include failure modes. "
                'If the chunk does not contain any reusable candidate principle, return {"principles":[]}.\n\n'
                f"source_file: {chunk.source_file}\n"
                f"chunk_id: {chunk.chunk_id}\n"
                f"page: {chunk.page}\n\n"
                f"{chunk.text}"
            ),
        ),
    ]
    response = model.query(messages=messages, tools=[], config=config)
    payload = _parse_json_object(response.content)
    raw_principles = payload.get("principles", [])
    if not isinstance(raw_principles, list):
        raise ValueError("extractor response field 'principles' must be a list")

    evidence = Evidence(source_file=chunk.source_file, chunk_id=chunk.chunk_id, page=chunk.page)
    candidates: list[PrincipleCandidate] = []
    for offset, item in enumerate(raw_principles, start=start_index):
        if not isinstance(item, dict):
            continue
        candidate_payload = _normalize_candidate_payload(item)
        candidate_payload["principle_id"] = f"{id_prefix}{offset:04d}"
        candidate_payload["status"] = "candidate"
        candidate_payload["evidence"] = [evidence.model_dump()]
        try:
            candidates.append(PrincipleCandidate.model_validate(candidate_payload))
        except ValidationError:
            continue
    return candidates


def extract_principles_from_chunks(
    *,
    chunks,
    model: ModelClient,
    config: ModelConfig,
    id_prefix: str = "P",
) -> list[PrincipleCandidate]:
    candidates: list[PrincipleCandidate] = []
    next_index = 1
    seen: set[str] = set()
    for chunk in chunks:
        extracted = extract_principles_from_chunk(
            chunk=chunk,
            model=model,
            config=config,
            id_prefix=id_prefix,
            start_index=next_index,
        )
        for candidate in extracted:
            key = _dedupe_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(replace_candidate_id(candidate, f"{id_prefix}{next_index:04d}"))
            next_index += 1
    return candidates


def replace_candidate_id(candidate: PrincipleCandidate, principle_id: str) -> PrincipleCandidate:
    payload = candidate.model_dump()
    payload["principle_id"] = principle_id
    return PrincipleCandidate.model_validate(payload)


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("extractor response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("extractor response must be a JSON object")
    return payload


def _normalize_candidate_payload(item: dict) -> dict:
    allowed_keys = {
        "name",
        "domain",
        "summary",
        "when_to_apply",
        "how_to_apply",
        "failure_modes",
        "confidence",
    }
    payload = {key: item[key] for key in allowed_keys if key in item}
    for key in ["when_to_apply", "how_to_apply", "failure_modes"]:
        if isinstance(payload.get(key), str):
            payload[key] = [payload[key]]
    return payload


def _dedupe_key(candidate: PrincipleCandidate) -> str:
    value = f"{candidate.name} {candidate.summary}"
    return " ".join(value.lower().split())
