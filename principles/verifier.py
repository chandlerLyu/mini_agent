"""Type-aware LLM verifier for candidate principles."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from config import ModelConfig
from interfaces import Message, ModelClient
from principles.schema import Chunk, Principle, PrincipleCandidate, VerificationResult, utc_now_iso


VERIFY_SYSTEM_PROMPT = """You verify extracted principle candidates for PrincipleRAG memory.

Do not use numeric scalar scores. Use typed judgment, structured critique, evidence binding, generalization audit, counterexample checks, and conservative revision.

Pipeline:
1. classify principle_type
2. decompose the principle according to type
3. bind exact evidence chunks to principle parts
4. check whether the evidence warrants the principle
5. audit generalization scope
6. identify counterexamples or boundary conditions
7. decide: verified, needs_revision, rejected, reclassify_as_observation, reclassify_as_definition, merge_duplicate, or requires_more_evidence

Principle types:
mechanism, normative_value, descriptive_phenomenon, procedural, constraint_invariant, failure_pattern, tradeoff, conceptual_definition.

Output contract:
- Return exactly one valid JSON object.
- Do not include markdown fences.
- Do not include explanatory text before or after the JSON.
- Use double quotes for every JSON key and string.
- Use null, true, and false, not None/True/False.
- Include every required top-level key shown in the expected shape.
- Keep comments concise so the full object fits within the token budget.
"""


def verify_principle(
    *,
    candidate: PrincipleCandidate,
    evidence_chunks: list[Chunk],
    model: ModelClient,
    config: ModelConfig,
) -> VerificationResult:
    response = model.query(
        messages=[
            Message(role="system", content=VERIFY_SYSTEM_PROMPT),
            Message(role="user", content=_build_user_prompt(candidate, evidence_chunks)),
        ],
        tools=[],
        config=config,
    )
    payload = _parse_json_object(response.content)
    payload = _normalize_verification_payload(payload, candidate.principle_id)
    try:
        return VerificationResult.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"verifier response failed schema validation: {exc}") from exc


def build_verified_principle(candidate: PrincipleCandidate, result: VerificationResult) -> Principle:
    payload = candidate.model_dump()
    if result.revision_needed and result.revised_principle:
        payload.update(_normalize_revised_principle(result.revised_principle))
    payload["principle_id"] = candidate.principle_id
    payload["evidence"] = candidate.model_dump()["evidence"]
    payload["status"] = "needs_revision" if result.decision == "needs_revision" else "verified"
    payload["updated_at"] = utc_now_iso()
    return Principle.model_validate(payload)


def resolve_evidence_chunks(candidate: PrincipleCandidate, chunks_by_id: dict[str, Chunk]) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for evidence in candidate.evidence:
        if evidence.chunk_id in seen:
            continue
        chunk = chunks_by_id.get(evidence.chunk_id)
        if chunk is not None:
            chunks.append(chunk)
            seen.add(evidence.chunk_id)
    return chunks


def _build_user_prompt(candidate: PrincipleCandidate, evidence_chunks: list[Chunk]) -> str:
    evidence_payload = [
        {
            "chunk_id": chunk.chunk_id,
            "source_file": chunk.source_file,
            "page": chunk.page,
            "text": chunk.text,
        }
        for chunk in evidence_chunks
    ]
    return (
        "Verify this candidate principle against its source evidence.\n"
        "If evidence is missing, use decision requires_more_evidence or rejected.\n"
        "Revise only by weakening, narrowing, clarifying, adding mechanisms, or adding boundaries.\n\n"
        "Expected JSON shape:\n"
        "{\n"
        '  "principle_id": "P0000",\n'
        '  "principle_type": "mechanism",\n'
        '  "type_confidence": "high",\n'
        '  "decision": "verified",\n'
        '  "decomposition": {"core_claim": "...", "scope": "...", "boundary_conditions": ["..."]},\n'
        '  "evidence_bindings": [{"principle_part": "core_claim", "source_chunk_id": "chunk_id", "support_type": "direct", "explanation": "..."}],\n'
        '  "clarity": {"label": "pass", "comment": "..."},\n'
        '  "reasoning_link": {"label": "strong", "comment": "..."},\n'
        '  "generalization_audit": {"label": "appropriate", "comment": "...", "suggested_scope": null},\n'
        '  "actionability": {"label": "actionable", "comment": "..."},\n'
        '  "boundary_awareness": {"label": "sufficient", "comment": "..."},\n'
        '  "counterexample_check": {"has_counterexample": true, "counterexample": "...", "needed_boundary_condition": "..."},\n'
        '  "revision_needed": false,\n'
        '  "revision_instruction": null,\n'
        '  "revised_principle": null,\n'
        '  "verifier_rationale": "..."\n'
        "}\n\n"
        "Allowed labels:\n"
        "- type_confidence: high | medium | low\n"
        "- decision: verified | needs_revision | rejected | reclassify_as_observation | reclassify_as_definition | merge_duplicate | requires_more_evidence\n"
        "- support_type: direct | indirect | missing | contradicted\n"
        "- generalization_audit.label: appropriate | overgeneralized | undergeneralized | not_a_generalization\n\n"
        f"Candidate principle:\n{candidate.model_dump_json()}\n\n"
        f"Evidence chunks:\n{json.dumps(evidence_payload, ensure_ascii=True)}"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        excerpt = text[:500].replace("\n", "\\n")
        raise ValueError(f"verifier response was not valid JSON; excerpt={excerpt!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError("verifier response must be a JSON object")
    return payload


def _normalize_verification_payload(payload: dict[str, Any], principle_id: str) -> dict[str, Any]:
    allowed = {
        "principle_id",
        "principle_type",
        "type_confidence",
        "decision",
        "decomposition",
        "evidence_bindings",
        "clarity",
        "reasoning_link",
        "generalization_audit",
        "actionability",
        "boundary_awareness",
        "counterexample_check",
        "revision_needed",
        "revision_instruction",
        "revised_principle",
        "verifier_rationale",
    }
    normalized = {key: payload[key] for key in allowed if key in payload}
    normalized.setdefault("principle_id", principle_id)
    normalized["principle_id"] = principle_id
    normalized.setdefault("type_confidence", "medium")
    normalized.setdefault("decomposition", {})
    normalized.setdefault("evidence_bindings", [])
    normalized.setdefault("revision_needed", bool(normalized.get("revised_principle")))
    normalized.setdefault("revision_instruction", None)
    normalized.setdefault("revised_principle", None)
    if isinstance(normalized.get("revised_principle"), str):
        if not normalized.get("revision_instruction"):
            normalized["revision_instruction"] = normalized["revised_principle"]
        normalized["revised_principle"] = None
    return normalized


def _normalize_revised_principle(revised: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "name",
        "domain",
        "summary",
        "when_to_apply",
        "how_to_apply",
        "failure_modes",
        "confidence",
    }
    payload = {key: revised[key] for key in allowed if key in revised}
    for key in ["when_to_apply", "how_to_apply", "failure_modes"]:
        if isinstance(payload.get(key), str):
            payload[key] = [payload[key]]
    return payload
