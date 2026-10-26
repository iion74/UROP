"""Preprocess HotpotQA, MuSiQue, and 2Wiki into one validated schema.

Examples:
    python -m experiments.preprocess_multihop --dataset all --split all
    python -m experiments.preprocess_multihop --dataset musique --split train --max-examples 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from functools import lru_cache, partial
from pathlib import Path
from time import perf_counter

from src.benchmark_data import (
    SCHEMA_VERSION,
    iter_json_array,
    normalise_2wiki_record,
    normalise_hotpot_record,
    normalise_musique_record,
    validate_example,
)


Adapter = Callable[[Mapping, str], dict]


def main() -> None:
    args = _arguments()
    datasets = ("hotpotqa", "musique", "2wiki") if args.dataset == "all" else (args.dataset,)
    splits = ("train", "validation") if args.split == "all" else (args.split,)
    jobs = [(dataset, split) for dataset in datasets for split in splits]
    print(
        f"멀티홉 공통 전처리를 시작합니다: 데이터셋 {len(datasets)}개 · 분할 {len(splits)}개 "
        f"· 스키마 v{SCHEMA_VERSION}",
        flush=True,
    )
    total_started = perf_counter()
    for position, (dataset, split) in enumerate(jobs, start=1):
        _process_job(dataset, split, args, position, len(jobs))
    print(f"전체 전처리 완료 · 총 소요 {_duration(perf_counter() - total_started)}", flush=True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="세 멀티홉 QA 데이터셋의 공통 스키마 전처리")
    parser.add_argument("--dataset", choices=("all", "hotpotqa", "musique", "2wiki"), default="all")
    parser.add_argument("--split", choices=("all", "train", "validation"), default="all")
    parser.add_argument("--hotpot", default="data/benchmarks/hotpotqa/distractor")
    parser.add_argument("--musique", default="data/benchmarks/musique")
    parser.add_argument("--two-wiki", default="data/benchmarks/2wikimultihopqa")
    parser.add_argument("--output-root", default="data/benchmarks/processed")
    parser.add_argument("--max-examples", type=int, default=0, help="분할별 최대 문항 수, 0이면 전체")
    parser.add_argument("--seed", type=int, default=42, help="일부 문항만 변환할 때의 재현 가능한 표본 seed")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_examples < 0:
        parser.error("--max-examples는 0 이상이어야 합니다.")
    if args.log_every < 1:
        parser.error("--log-every는 1 이상이어야 합니다.")
    return args


def _process_job(dataset: str, split: str, args: argparse.Namespace, job: int, jobs: int) -> None:
    output = Path(args.output_root) / dataset / f"{split}.jsonl"
    manifest = output.with_suffix(".manifest.json")
    if not args.overwrite and (output.exists() or manifest.exists()):
        raise SystemExit(f"출력 파일이 이미 존재합니다. 덮어쓰려면 --overwrite를 사용하세요: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jsonl.tmp")
    source, raw_split, raw_records, adapter = _records_for(dataset, split, args)
    sample_seed = _dataset_seed(args.seed, dataset, split)
    records = _sample_records(raw_records, args.max_examples, sample_seed)
    print(f"[{job}/{jobs}] {dataset}/{split} 시작 · 원본 {source} ({raw_split})", flush=True)
    started = perf_counter()
    stats = _empty_stats()
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            for record in records:
                example = adapter(record, split)
                validate_example(example, require_gold=True)
                destination.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")
                _update_stats(stats, example)
                if stats["rows"] % args.log_every == 0:
                    elapsed = perf_counter() - started
                    print(
                        f"      {stats['rows']:,}문항 · {stats['documents']:,}문서 "
                        f"· {stats['rows'] / max(elapsed, 1e-9):.1f}문항/초",
                        flush=True,
                    )
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "split": split,
        "source_split": raw_split,
        "source": str(source),
        "output": str(output),
        "max_examples": args.max_examples,
        "sampling": "reservoir" if args.max_examples else "all_in_source_order",
        "seed": args.seed,
        "stats": _finalise_stats(stats),
    }
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[{job}/{jobs}] 완료: {stats['rows']:,}문항 · {stats['documents']:,}문서 "
        f"· 정답 근거 {stats['gold_supports']:,}개 · {_duration(perf_counter() - started)}\n"
        f"      출력 {output}\n"
        f"      hop 분포 {dict(sorted(stats['hop_counts'].items()))} "
        f"· 유형 분포 {dict(sorted(stats['reasoning_types'].items()))}",
        flush=True,
    )


def _records_for(
    dataset: str,
    split: str,
    args: argparse.Namespace,
) -> tuple[Path, str, Iterator[Mapping], Adapter]:
    if dataset == "hotpotqa":
        source = Path(args.hotpot)
        raw_split = "train" if split == "train" else "validation"
        from datasets import load_from_disk

        saved = load_from_disk(str(source))
        if not hasattr(saved, "keys") or raw_split not in saved:
            raise ValueError(f"HotpotQA에 {raw_split} 분할이 없습니다: {source}")
        return source, raw_split, iter(saved[raw_split]), normalise_hotpot_record

    if dataset == "musique":
        raw_split = "train" if split == "train" else "dev"
        source = Path(args.musique) / f"musique_ans_v1.0_{raw_split}.jsonl"
        return source, raw_split, _iter_jsonl(source), normalise_musique_record

    raw_split = "train" if split == "train" else "dev"
    source = Path(args.two_wiki) / f"{raw_split}.json"
    aliases = _load_2wiki_aliases(Path(args.two_wiki) / "id_aliases.json")
    adapter = partial(normalise_2wiki_record, answer_aliases_by_id=aliases)
    return source, raw_split, iter_json_array(source), adapter


@lru_cache(maxsize=4)
def _load_2wiki_aliases(path: Path) -> dict[str, tuple[str, ...]]:
    """Load the official 2Wiki answer aliases and demonyms JSONL file."""
    aliases: dict[str, tuple[str, ...]] = {}
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping) or not record.get("Q_id"):
                raise ValueError(f"2Wiki alias {line_number}행 형식이 잘못됐습니다: {path}")
            values = (*record.get("aliases", ()), *record.get("demonyms", ()))
            aliases[str(record["Q_id"])] = tuple(
                dict.fromkeys(str(value).strip() for value in values if str(value).strip())
            )
    return aliases


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL {line_number}행이 객체가 아닙니다: {path}")
            yield record


def _sample_records(records: Iterator[Mapping], maximum: int, seed: int) -> Iterator[Mapping]:
    """Uniformly sample a streaming split instead of taking its biased prefix."""
    if maximum == 0:
        return records
    rng = random.Random(seed)
    reservoir: list[Mapping] = []
    for position, record in enumerate(records):
        if position < maximum:
            reservoir.append(record)
            continue
        replacement = rng.randrange(position + 1)
        if replacement < maximum:
            reservoir[replacement] = record
    rng.shuffle(reservoir)
    return iter(reservoir)


def _dataset_seed(seed: int, dataset: str, split: str) -> int:
    suffix = int(hashlib.sha256(f"{dataset}/{split}".encode("utf-8")).hexdigest()[:8], 16)
    return seed ^ suffix


def _empty_stats() -> dict:
    return {
        "rows": 0,
        "documents": 0,
        "gold_supports": 0,
        "duplicate_title_rows": 0,
        "ambiguous_gold_rows": 0,
        "invalid_sentence_labels": 0,
        "answerability": Counter(),
        "hop_counts": Counter(),
        "reasoning_types": Counter(),
    }


def _update_stats(stats: dict, example: Mapping) -> None:
    stats["rows"] += 1
    documents = example["documents"]
    supports = example["gold_supports"]
    stats["documents"] += len(documents)
    stats["gold_supports"] += len(supports)
    title_counts = Counter(document["normalised_title"] for document in documents)
    stats["duplicate_title_rows"] += int(any(count > 1 for count in title_counts.values()))
    stats["ambiguous_gold_rows"] += int(any(len(support["doc_ids"]) > 1 for support in supports))
    stats["invalid_sentence_labels"] += sum(len(support.get("invalid_sentence_ids", [])) for support in supports)
    answerability = example.get("answerable")
    key = "answerable" if answerability is True else "unanswerable" if answerability is False else "unlabelled"
    stats["answerability"][key] += 1
    if example.get("hop_count") is not None:
        stats["hop_counts"][int(example["hop_count"])] += 1
    if example.get("reasoning_type"):
        stats["reasoning_types"][str(example["reasoning_type"])] += 1


def _finalise_stats(stats: dict) -> dict:
    return {
        key: dict(sorted(value.items())) if isinstance(value, Counter) else value
        for key, value in stats.items()
    }


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}시간 {minutes:02d}분 {seconds:02d}초"
    if minutes:
        return f"{minutes}분 {seconds:02d}초"
    return f"{seconds}초"


if __name__ == "__main__":
    main()
