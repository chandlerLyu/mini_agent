"""Token-aware chunking for raw corpus memory."""

from __future__ import annotations

from hashlib import sha1

from principles.loaders import LoadedDocument
from principles.schema import Chunk, utc_now_iso


class Tokenizer:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:  # pragma: no cover - fallback for minimal envs
            self._encoding = None

    def encode(self, text: str) -> list[int] | list[str]:
        if self._encoding is not None:
            return self._encoding.encode(text)
        return text.split()

    def decode(self, tokens: list[int] | list[str]) -> str:
        if self._encoding is not None:
            return self._encoding.decode(tokens)
        return " ".join(str(token) for token in tokens)


def chunk_documents(
    documents: list[LoadedDocument],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    tokenizer = tokenizer or Tokenizer()
    chunks: list[Chunk] = []
    for document in documents:
        tokens = tokenizer.encode(document.text)
        if not tokens:
            continue
        start = 0
        local_index = 1
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            text = tokenizer.decode(tokens[start:end]).strip()
            if text:
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(document.source_file, document.page, local_index),
                        source_file=document.source_file,
                        page=document.page,
                        text=text,
                        metadata={
                            "file_type": document.file_type,
                            "created_at": utc_now_iso(),
                            "chunk_index": local_index,
                        },
                    )
                )
                local_index += 1
            if end == len(tokens):
                break
            start = end - chunk_overlap
    return chunks


def _chunk_id(source_file: str, page: int | None, index: int) -> str:
    stable = sha1(f"{source_file}:{page}:{index}".encode("utf-8")).hexdigest()[:10]
    return f"chunk_{stable}_{index:04d}"
