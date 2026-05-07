"""Corpus file loading for text, markdown, and PDF documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedDocument:
    source_file: str
    text: str
    file_type: str
    page: int | None = None


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def load_corpus(input_dir: Path) -> list[LoadedDocument]:
    documents: list[LoadedDocument] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        documents.extend(load_document(path, root=input_dir))
    return documents


def load_document(path: Path, *, root: Path | None = None) -> list[LoadedDocument]:
    suffix = path.suffix.lower()
    source_file = str(path.relative_to(root)) if root else path.name
    if suffix in {".txt", ".md"}:
        return [
            LoadedDocument(
                source_file=source_file,
                text=path.read_text(encoding="utf-8"),
                file_type=suffix.removeprefix("."),
            )
        ]
    if suffix == ".pdf":
        return _load_pdf(path, source_file=source_file)
    raise ValueError(f"Unsupported corpus file type: {path.suffix}")


def _load_pdf(path: Path, *, source_file: str) -> list[LoadedDocument]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pypdf is required to load PDF files") from exc

    reader = PdfReader(str(path))
    documents: list[LoadedDocument] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                LoadedDocument(
                    source_file=source_file,
                    text=text,
                    file_type="pdf",
                    page=page_index,
                )
            )
    return documents
