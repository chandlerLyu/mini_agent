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
