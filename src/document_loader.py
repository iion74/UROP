from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import fitz


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".csv"}


@dataclass(frozen=True)
class SourceDocument:
    source: str
    text: str
    # ``source`` is the stable document identifier used by retrieval metrics.
    # Benchmark corpora need a separate human/Wikipedia title because two
    # different passages can legitimately share the same title.
    title: str | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    title: str


def supported_files(data_dir: Path) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    return sorted(path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def load_documents(data_dir: Path) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for path in supported_files(data_dir):
        try:
            text = _read_file(path)
        except Exception:
            continue
        if text.strip():
            documents.append(SourceDocument(source=str(path.relative_to(data_dir)), text=text))
    return documents


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with fitz.open(path) as pdf:
            return "\n".join(page.get_text("text") for page in pdf)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", errors="ignore", newline="") as file:
            return "\n".join(" | ".join(row) for row in csv.reader(file))
    return ""


def chunk_documents(documents: list[SourceDocument], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        normalized = re.sub(r"[ \t]+", " ", document.text)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
        start = 0
        sequence = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            if end < len(normalized):
                boundary = max(normalized.rfind(". ", start, end), normalized.rfind("\n", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary + 1
            text = normalized[start:end].strip()
            if text:
                chunks.append(Chunk(
                    chunk_id=f"{document.source}:{sequence}",
                    source=document.source,
                    title=document.title or Path(document.source).stem,
                    text=text,
                ))
                sequence += 1
            # The final chunk has no following chunk to overlap with.  Moving
            # back by ``overlap`` here would repeatedly index its trailing
            # characters one-by-one, which is especially destructive for a
            # benchmark corpus containing many short documents.
            if end >= len(normalized):
                break
            start = max(end - overlap, start + 1)
    return chunks
