"""Typed schemas for raw corpus chunks and candidate principles."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_file: str
    page: int | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk text must not be empty")
        return value


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_file: str
    chunk_id: str
    page: int | None = None
    quote: str | None = None


class PrincipleCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principle_id: str
    name: str
    domain: str
    summary: str
    when_to_apply: list[str]
    how_to_apply: list[str]
    failure_modes: list[str]
    evidence: list[Evidence]
    confidence: float = Field(ge=0.0, le=1.0)
    status: Literal["candidate"] = "candidate"
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    @field_validator("name", "domain", "summary")
    @classmethod
    def text_fields_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must not be empty")
        return value.strip()

    @field_validator("when_to_apply", "how_to_apply", "failure_modes", "evidence")
    @classmethod
    def lists_must_not_be_empty(cls, value: list[Any]) -> list[Any]:
        if not value:
            raise ValueError("list field must not be empty")
        return value


class ExtractedPrinciples(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principles: list[PrincipleCandidate] = Field(default_factory=list)


class Principle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principle_id: str
    name: str
    domain: str
    summary: str
    when_to_apply: list[str]
    how_to_apply: list[str]
    failure_modes: list[str]
    evidence: list[Evidence]
    confidence: float = Field(ge=0.0, le=1.0)
    status: Literal["candidate", "verified", "rejected", "needs_revision"]
    created_at: str
    updated_at: str
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    @field_validator("name", "domain", "summary")
    @classmethod
    def principle_text_fields_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must not be empty")
        return value.strip()

    @field_validator("when_to_apply", "how_to_apply", "failure_modes", "evidence")
    @classmethod
    def principle_lists_must_not_be_empty(cls, value: list[Any]) -> list[Any]:
        if not value:
            raise ValueError("list field must not be empty")
        return value


PrincipleType = Literal[
    "mechanism",
    "normative_value",
    "descriptive_phenomenon",
    "procedural",
    "constraint_invariant",
    "failure_pattern",
    "tradeoff",
    "conceptual_definition",
]

VerificationDecision = Literal[
    "verified",
    "needs_revision",
    "rejected",
    "reclassify_as_observation",
    "reclassify_as_definition",
    "merge_duplicate",
    "requires_more_evidence",
]


class VerificationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    comment: str


class EvidenceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principle_part: str
    source_chunk_id: str
    support_type: Literal["direct", "indirect", "missing", "contradicted"]
    explanation: str


class GeneralizationAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["appropriate", "overgeneralized", "undergeneralized", "not_a_generalization"]
    comment: str
    suggested_scope: str | None = None


class CounterexampleCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_counterexample: bool
    counterexample: str | None = None
    needed_boundary_condition: str | None = None


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principle_id: str
    principle_type: PrincipleType
    type_confidence: Literal["high", "medium", "low"]
    decision: VerificationDecision
    decomposition: dict[str, Any] = Field(default_factory=dict)
    evidence_bindings: list[EvidenceBinding] = Field(default_factory=list)
    clarity: VerificationCheck
    reasoning_link: VerificationCheck
    generalization_audit: GeneralizationAudit
    actionability: VerificationCheck
    boundary_awareness: VerificationCheck
    counterexample_check: CounterexampleCheck
    revision_needed: bool = False
    revision_instruction: str | None = None
    revised_principle: dict[str, Any] | None = None
    verifier_rationale: str


class VerificationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: PrincipleCandidate
    result: VerificationResult
