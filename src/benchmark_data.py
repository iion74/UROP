"""Canonical preprocessing adapters for multi-hop QA benchmarks.

The raw releases use different evidence identifiers: HotpotQA and 2Wiki use
article titles plus sentence indices, while MuSiQue identifies a particular
paragraph by its per-question index.  This module preserves those semantics
and exposes one JSON-serialisable schema for retrieval-policy experiments.

Gold decompositions and relation triples are deliberately stored separately
from documents.  They may supervise or evaluate a policy, but must never be
used to construct an evaluation-time retrieval graph.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path


SCHEMA_VERSION = 1


def normalise_hotpot_record(record: Mapping, split: str) -> dict:
    """Convert a Hugging Face or official HotpotQA row."""
    example_id = str(record.get("id", record.get("_id", ""))).strip()
    contexts = _context_pairs(record.get("context", []))
    documents = [
        _document("hotpotqa", example_id, position, title, sentences)
        for position, (title, sentences) in enumerate(contexts)
        if _text_from_sentences(sentences)
    ]
    supports = _title_supports(
        "hotpotqa",
        example_id,
        documents,
        _supporting_fact_pairs(record.get("supporting_facts", [])),
    )
    has_labels = bool(record.get("answer")) and bool(supports)
    return _example(
        dataset="hotpotqa",
        split=split,
        example_id=example_id,
        question=record.get("question"),
        answer=record.get("answer") if has_labels else None,
        answer_aliases=(),
        answerable=True if has_labels else None,
        reasoning_type=record.get("type"),
        difficulty=record.get("level"),
        hop_count=2 if supports else None,
        documents=documents,
        gold_supports=supports,
        reasoning_chain=(),
        metadata={},
    )


def normalise_musique_record(record: Mapping, split: str) -> dict:
    """Convert a MuSiQue answerable/full row without losing paragraph IDs."""
    example_id = str(record.get("id", "")).strip()
    documents: list[dict] = []
    document_by_paragraph_idx: dict[int, dict] = {}
    for position, paragraph in enumerate(record.get("paragraphs", [])):
        if not isinstance(paragraph, Mapping):
            continue
        paragraph_idx = _optional_int(paragraph.get("idx"))
        text = _clean_text(paragraph.get("paragraph_text"))
        if paragraph_idx is None or not text:
            continue
        document = _document(
            "musique",
            example_id,
            position,
            paragraph.get("title"),
            (),
            text=text,
            source_key=f"p{paragraph_idx}",
        )
        documents.append(document)
        document_by_paragraph_idx[paragraph_idx] = document

    reasoning_chain: list[dict] = []
    gold_supports: list[dict] = []
    seen_support_indices: set[int] = set()
    decomposition = record.get("question_decomposition", [])
    if not isinstance(decomposition, Sequence) or isinstance(decomposition, (str, bytes)):
        decomposition = []
    for hop, step in enumerate(decomposition, start=1):
        if not isinstance(step, Mapping):
            continue
        paragraph_idx = _optional_int(step.get("paragraph_support_idx"))
        document = document_by_paragraph_idx.get(paragraph_idx) if paragraph_idx is not None else None
        reasoning_chain.append(
            {
                "step": hop,
                "question": _clean_text(step.get("question")),
                "answer": _clean_text(step.get("answer")),
                "support_doc_ids": [document["doc_id"]] if document else [],
                "source_step_id": step.get("id"),
            }
        )
        if document and paragraph_idx not in seen_support_indices:
            gold_supports.append(
                _support(
                    "musique",
                    example_id,
                    len(gold_supports),
                    document["title"],
                    (document["doc_id"],),
                    (),
                    paragraph_idx=paragraph_idx,
                    hop=hop,
                )
            )
            seen_support_indices.add(paragraph_idx)

    # Keep a gold paragraph even if a release row omits it from the explicit
    # decomposition.  In MuSiQue-Ans these two annotations normally agree.
    for paragraph in record.get("paragraphs", []):
        if not isinstance(paragraph, Mapping) or not paragraph.get("is_supporting"):
            continue
        paragraph_idx = _optional_int(paragraph.get("idx"))
        if paragraph_idx is None or paragraph_idx in seen_support_indices:
            continue
        document = document_by_paragraph_idx.get(paragraph_idx)
        if document:
            gold_supports.append(
                _support(
                    "musique",
                    example_id,
                    len(gold_supports),
                    document["title"],
                    (document["doc_id"],),
                    (),
                    paragraph_idx=paragraph_idx,
                    hop=None,
                )
            )
            seen_support_indices.add(paragraph_idx)

    answer_present = "answer" in record
    answerable = record.get("answerable")
    if not isinstance(answerable, bool):
        answerable = True if answer_present and gold_supports else None
    hop_count = len(reasoning_chain) if reasoning_chain else None
    return _example(
        dataset="musique",
        split=split,
        example_id=example_id,
        question=record.get("question"),
        answer=record.get("answer") if answer_present else None,
        answer_aliases=record.get("answer_aliases", ()),
        answerable=answerable,
        reasoning_type=f"{hop_count}hop" if hop_count else None,
        difficulty=None,
        hop_count=hop_count,
        documents=documents,
        gold_supports=gold_supports,
        reasoning_chain=reasoning_chain,
        metadata={},
    )


def normalise_2wiki_record(
    record: Mapping,
    split: str,
    *,
    answer_aliases_by_id: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    """Convert a 2WikiMultiHopQA row and retain its relation evidence."""
    example_id = str(record.get("_id", record.get("id", ""))).strip()
    contexts = _context_pairs(record.get("context", []))
    documents = [
        _document("2wiki", example_id, position, title, sentences)
        for position, (title, sentences) in enumerate(contexts)
        if _text_from_sentences(sentences)
    ]
    supports = _title_supports(
        "2wiki",
        example_id,
        documents,
        _supporting_fact_pairs(record.get("supporting_facts", [])),
    )

    evidence_ids = record.get("evidences_id", [])
    reasoning_chain: list[dict] = []
    evidences = record.get("evidences", [])
    if not isinstance(evidences, Sequence) or isinstance(evidences, (str, bytes)):
        evidences = []
    for position, evidence in enumerate(evidences):
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)) or len(evidence) != 3:
            continue
        relation_ids = (
            evidence_ids[position]
            if isinstance(evidence_ids, Sequence) and position < len(evidence_ids)
            else ()
        )
        reasoning_chain.append(
            {
                "step": position + 1,
                "subject": _clean_text(evidence[0]),
                "relation": _clean_text(evidence[1]),
                "object": _clean_text(evidence[2]),
                "entity_ids": list(relation_ids) if isinstance(relation_ids, Sequence) else [],
            }
        )

    has_labels = bool(record.get("answer")) and bool(supports)
    answer_id = _clean_text(record.get("answer_id"))
    answer_aliases = (
        answer_aliases_by_id.get(answer_id, ())
        if answer_aliases_by_id is not None and answer_id
        else ()
    )
    return _example(
        dataset="2wiki",
        split=split,
        example_id=example_id,
        question=record.get("question"),
        answer=record.get("answer") if has_labels else None,
        answer_aliases=answer_aliases,
        answerable=True if has_labels else None,
        reasoning_type=record.get("type"),
        difficulty=None,
        # 2Wiki questions contain two-hop paths.  bridge_comparison has two
        # such paths and therefore four supporting documents, not four depth.
        hop_count=2 if supports else None,
        documents=documents,
        gold_supports=supports,
        reasoning_chain=reasoning_chain,
        metadata={
            "answer_id": record.get("answer_id"),
            "entity_ids": record.get("entity_ids"),
            "reasoning_step_count": len(reasoning_chain),
        },
    )


def validate_example(example: Mapping, *, require_gold: bool = True) -> None:
    """Fail early when an adapter would silently corrupt evidence labels."""
    prefix = f"{example.get('dataset')}:{example.get('id')}"
    if example.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{prefix}: 지원하지 않는 공통 스키마 버전입니다.")
    if not _clean_text(example.get("question")):
        raise ValueError(f"{prefix}: 질문이 비어 있습니다.")
    documents = example.get("documents", [])
    if not documents:
        raise ValueError(f"{prefix}: 후보 문서가 없습니다.")
    document_ids = [document.get("doc_id") for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError(f"{prefix}: 문서 ID가 중복됩니다.")
    known_ids = set(document_ids)

    supports = example.get("gold_supports", [])
    if require_gold and example.get("answerable") is not False and not supports:
        raise ValueError(f"{prefix}: 정답 근거가 없습니다.")
    by_id = {document["doc_id"]: document for document in documents}
    for support in supports:
        candidate_ids = support.get("doc_ids", [])
        if not candidate_ids or not set(candidate_ids).issubset(known_ids):
            raise ValueError(f"{prefix}: 정답 근거가 후보 문서와 연결되지 않습니다.")
        sentence_ids = support.get("sentence_ids", [])
        if sentence_ids and not any(
            all(0 <= sentence_id < len(by_id[doc_id].get("sentences", [])) for sentence_id in sentence_ids)
            for doc_id in candidate_ids
        ):
            raise ValueError(f"{prefix}: 정답 문장 번호가 문서 범위를 벗어납니다.")


def iter_json_array(path: Path, *, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Stream a top-level JSON array without loading 2Wiki train into RAM."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8-sig") as source:
        buffer = ""
        position = 0
        eof = False
        while True:
            if position > chunk_size:
                buffer = buffer[position:]
                position = 0
            if not eof and len(buffer) - position < chunk_size:
                block = source.read(chunk_size)
                if block:
                    buffer += block
                else:
                    eof = True

            while position < len(buffer) and (buffer[position].isspace() or buffer[position] in "[," ):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            if position >= len(buffer):
                if eof:
                    return
                continue

            try:
                record, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                block = source.read(chunk_size)
                if block:
                    buffer += block
                else:
                    eof = True
                continue
            if not isinstance(record, dict):
                raise ValueError(f"JSON 배열 항목이 객체가 아닙니다: {path}")
            yield record
            position = end


def summarise_examples(examples: Iterable[Mapping]) -> dict:
    """Return compact validation statistics for a processed JSONL stream."""
    rows = 0
    documents = 0
    gold_supports = 0
    duplicate_title_rows = 0
    ambiguous_gold_rows = 0
    answerability = {"answerable": 0, "unanswerable": 0, "unlabelled": 0}
    hop_counts: dict[str, int] = {}
    reasoning_types: dict[str, int] = {}
    for example in examples:
        rows += 1
        docs = example.get("documents", [])
        documents += len(docs)
        supports = example.get("gold_supports", [])
        gold_supports += len(supports)
        title_counts: dict[str, int] = {}
        for document in docs:
            title = str(document.get("normalised_title", ""))
            title_counts[title] = title_counts.get(title, 0) + 1
        duplicate_title_rows += int(any(count > 1 for count in title_counts.values()))
        ambiguous_gold_rows += int(any(len(support.get("doc_ids", [])) > 1 for support in supports))
        if example.get("answerable") is True:
            answerability["answerable"] += 1
        elif example.get("answerable") is False:
            answerability["unanswerable"] += 1
        else:
            answerability["unlabelled"] += 1
        hop = example.get("hop_count")
        if hop is not None:
            hop_counts[str(hop)] = hop_counts.get(str(hop), 0) + 1
        reasoning_type = example.get("reasoning_type")
        if reasoning_type:
            reasoning_types[str(reasoning_type)] = reasoning_types.get(str(reasoning_type), 0) + 1
    return {
        "rows": rows,
        "documents": documents,
        "gold_supports": gold_supports,
        "duplicate_title_rows": duplicate_title_rows,
        "ambiguous_gold_rows": ambiguous_gold_rows,
        "answerability": answerability,
        "hop_counts": dict(sorted(hop_counts.items())),
        "reasoning_types": dict(sorted(reasoning_types.items())),
    }


def _example(
    *,
    dataset: str,
    split: str,
    example_id: str,
    question,
    answer,
    answer_aliases,
    answerable: bool | None,
    reasoning_type,
    difficulty,
    hop_count: int | None,
    documents: list[dict],
    gold_supports: list[dict],
    reasoning_chain: Iterable[dict],
    metadata: Mapping,
) -> dict:
    aliases = answer_aliases if isinstance(answer_aliases, Sequence) and not isinstance(answer_aliases, (str, bytes)) else ()
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "split": split,
        "id": example_id,
        "question": _clean_text(question),
        "answer": _clean_text(answer) if answer is not None else None,
        "answer_aliases": [_clean_text(alias) for alias in aliases if _clean_text(alias)],
        "answerable": answerable,
        "reasoning_type": _clean_text(reasoning_type) or None,
        "difficulty": _clean_text(difficulty) or None,
        "hop_count": hop_count,
        "support_document_count": len(gold_supports),
        "documents": documents,
        "gold_supports": gold_supports,
        "reasoning_chain": list(reasoning_chain),
        "metadata": dict(metadata),
    }


def _document(
    dataset: str,
    example_id: str,
    position: int,
    title,
    sentences: Iterable,
    *,
    text: str | None = None,
    source_key: str | None = None,
) -> dict:
    clean_title = _clean_text(title)
    clean_sentences = [_clean_text(sentence) for sentence in sentences if _clean_text(sentence)]
    clean_text = _clean_text(text) if text is not None else " ".join(clean_sentences)
    source_key = source_key or f"c{position}"
    fingerprint = hashlib.sha1(
        f"{normalise_title(clean_title)}\0{clean_text}".encode("utf-8")
    ).hexdigest()
    return {
        "doc_id": f"{dataset}:{example_id}:{source_key}",
        "content_id": f"{dataset}:{fingerprint}",
        "source_index": position,
        "title": clean_title,
        "normalised_title": normalise_title(clean_title),
        "text": clean_text,
        "sentences": clean_sentences,
    }


def _support(
    dataset: str,
    example_id: str,
    position: int,
    title: str,
    document_ids: Iterable[str],
    sentence_ids: Iterable[int],
    *,
    paragraph_idx: int | None,
    hop: int | None,
    invalid_sentence_ids: Iterable[int] = (),
) -> dict:
    return {
        "support_id": f"{dataset}:{example_id}:s{position}",
        "title": _clean_text(title),
        "doc_ids": list(document_ids),
        "sentence_ids": sorted(set(sentence_ids)),
        "invalid_sentence_ids": sorted(set(invalid_sentence_ids)),
        "paragraph_idx": paragraph_idx,
        "hop": hop,
    }


def _title_supports(
    dataset: str,
    example_id: str,
    documents: list[dict],
    supporting_facts: Iterable[tuple[str, int]],
) -> list[dict]:
    grouped: OrderedDict[str, dict] = OrderedDict()
    for title, sentence_id in supporting_facts:
        key = normalise_title(title)
        group = grouped.setdefault(key, {"title": _clean_text(title), "sentence_ids": []})
        if sentence_id >= 0:
            group["sentence_ids"].append(sentence_id)

    document_ids_by_title: dict[str, list[str]] = {}
    for document in documents:
        document_ids_by_title.setdefault(document["normalised_title"], []).append(document["doc_id"])

    documents_by_id = {document["doc_id"]: document for document in documents}
    supports: list[dict] = []
    for position, (title_key, group) in enumerate(grouped.items()):
        document_ids = document_ids_by_title.get(title_key, ())
        valid_sentence_ids: list[int] = []
        invalid_sentence_ids: list[int] = []
        for sentence_id in set(group["sentence_ids"]):
            if any(sentence_id < len(documents_by_id[doc_id]["sentences"]) for doc_id in document_ids):
                valid_sentence_ids.append(sentence_id)
            else:
                invalid_sentence_ids.append(sentence_id)
        supports.append(
            _support(
                dataset,
                example_id,
                position,
                group["title"],
                document_ids,
                valid_sentence_ids,
                paragraph_idx=None,
                hop=None,
                invalid_sentence_ids=invalid_sentence_ids,
            )
        )
    return supports


def _context_pairs(raw_context) -> list[tuple[str, list[str]]]:
    if isinstance(raw_context, Mapping):
        raw_context = zip(raw_context.get("title", []), raw_context.get("sentences", []), strict=False)
    if not isinstance(raw_context, Iterable) or isinstance(raw_context, (str, bytes)):
        return []
    pairs: list[tuple[str, list[str]]] = []
    for item in raw_context:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            continue
        title, sentences = item
        if not isinstance(sentences, Sequence) or isinstance(sentences, (str, bytes)):
            continue
        pairs.append((_clean_text(title), [str(sentence) for sentence in sentences if isinstance(sentence, str)]))
    return pairs


def _supporting_fact_pairs(raw_facts) -> list[tuple[str, int]]:
    if isinstance(raw_facts, Mapping):
        raw_facts = zip(raw_facts.get("title", []), raw_facts.get("sent_id", []), strict=False)
    if not isinstance(raw_facts, Iterable) or isinstance(raw_facts, (str, bytes)):
        return []
    pairs: list[tuple[str, int]] = []
    for item in raw_facts:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            continue
        title, sentence_id = item
        parsed_sentence_id = _optional_int(sentence_id)
        if isinstance(title, str) and parsed_sentence_id is not None:
            pairs.append((title, parsed_sentence_id))
    return pairs


def _text_from_sentences(sentences: Iterable) -> str:
    return " ".join(_clean_text(sentence) for sentence in sentences if _clean_text(sentence))


def normalise_title(title) -> str:
    """Normalise only for matching; the original display title is retained."""
    return re.sub(r"[^\w]+", " ", _clean_text(title).casefold(), flags=re.UNICODE).strip()


def _clean_text(value) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", value).strip()


def _optional_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
