"""Train one adaptive retrieval policy from three multi-hop QA benchmarks.

The datasets keep separate retrieval indexes so that corpus size and duplicate
titles from one benchmark cannot contaminate another.  Only their policy rows
(query-time evidence features and oracle action utilities) are combined.

Example:
    python -m experiments.train_multihop_policy \
      --processed-root data/benchmarks/processed \
      --datasets hotpotqa musique 2wiki \
      --index-root data/benchmarks/multihop/fixed_corpus/index \
      --rows-root data/benchmarks/multihop/fixed_corpus/policy_rows \
      --model-out data/models/multihop_action_policy_fixed_corpus.npz \
      --corpus-examples-per-dataset 2000 \
      --max-examples-per-dataset 2000

The corpus cohort and the labelled query cohort are deliberately separate.
Reducing ``--max-examples-per-dataset`` for a pilot therefore does not make
retrieval artificially easier by shrinking the index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from src.document_loader import SourceDocument, chunk_documents
from src.policy import LinearUtilityPolicy, POLICY_FEATURE_NAMES, PolicyTrainingProgress, profile_features
from src.retriever import GRAPH_FORMAT_VERSION, SemanticRetriever
from src.routing import POLICY_ACTION_IDS, EvidenceProfile, QueryRouter
from src.settings import settings


SUPPORTED_DATASETS = ("hotpotqa", "musique", "2wiki")
LABEL_FORMAT_VERSION = 3
QUERY_SAMPLING_VERSION = "stable_hash_v1"


@dataclass(frozen=True)
class CorpusDocument:
    content_id: str
    title: str
    text: str


@dataclass(frozen=True)
class MultiHopExample:
    dataset: str
    example_id: str
    question: str
    documents: tuple[CorpusDocument, ...]
    supporting_groups: tuple[frozenset[str], ...]
    hop_count: int | None
    reasoning_type: str | None
    answer: str | None = None
    answer_aliases: tuple[str, ...] = ()

    @property
    def policy_id(self) -> str:
        return f"{self.dataset}:{self.example_id}"


def main() -> None:
    args = _arguments()
    started = perf_counter()
    if args.prepare_index_only:
        _log("멀티홉 QA 고정 corpus 인덱스 준비를 시작합니다. (정책 라벨과 생성 LLM은 사용하지 않습니다.)")
    else:
        _log("세 데이터셋 통합 검색 행동 정책 학습을 시작합니다. (생성 LLM은 사용하지 않습니다.)")
    query_count = "미사용" if args.prepare_index_only else str(args.max_examples_per_dataset or "corpus 전체")
    _log(
        "설정: "
        f"datasets={','.join(args.datasets)}, split={args.split}, "
        f"corpus_examples_per_dataset={args.corpus_examples_per_dataset or '전체'}, "
        f"max_queries_per_dataset={query_count}, top_k={args.top_k}, "
        f"joint_weight={args.joint_recall_weight:.2f}, epochs={args.epochs}, seed={args.seed}"
    )

    examples_by_dataset: dict[str, list[MultiHopExample]] = {}
    corpus_by_dataset: dict[str, list[MultiHopExample]] = {}
    corpus_fingerprints: dict[str, str] = {}
    query_fingerprints: dict[str, str] = {}
    stage_started = perf_counter()
    for dataset in args.datasets:
        path = Path(args.processed_root) / dataset / f"{args.split}.jsonl"
        corpus_examples = load_processed_examples(
            path,
            args.corpus_examples_per_dataset,
            _dataset_seed(args.corpus_seed, dataset),
        )
        if not corpus_examples:
            raise SystemExit(f"유효한 전처리 문항이 없습니다: {path}")
        if (
            not args.prepare_index_only
            and args.max_examples_per_dataset
            and args.max_examples_per_dataset > len(corpus_examples)
        ):
            raise SystemExit(
                f"{dataset}: 요청한 query {args.max_examples_per_dataset:,}개가 고정 corpus cohort "
                f"{len(corpus_examples):,}개보다 많습니다. --corpus-examples-per-dataset을 늘리세요."
            )
        examples = (
            corpus_examples
            if args.prepare_index_only
            else select_query_examples(
                corpus_examples,
                args.max_examples_per_dataset,
                _dataset_seed(args.query_seed, dataset),
            )
        )
        examples_by_dataset[dataset] = examples
        corpus_by_dataset[dataset] = corpus_examples
        corpus_fingerprints[dataset] = _corpus_fingerprint(corpus_examples)
        query_fingerprints[dataset] = _example_fingerprint(examples)
        hops = Counter(example.hop_count for example in examples)
        types = Counter(example.reasoning_type for example in examples)
        _log(
            f"[1/5] {dataset}: 고정 corpus cohort {len(corpus_examples):,}문항 · "
            f"{'인덱스만 준비' if args.prepare_index_only else f'정책 query {len(examples):,}문항'} "
            f"· hop {dict(sorted(hops.items(), key=lambda x: str(x[0])))} "
            f"· 유형 {dict(sorted(types.items(), key=lambda x: str(x[0])))}"
        )
    _log(f"[1/5] 공통 데이터 준비 완료 · {_duration(perf_counter() - stage_started)}")

    all_rows: list[dict] = []
    shared_embedding_model = None
    shared_reranker = None
    for dataset in args.datasets:
        examples = examples_by_dataset[dataset]
        corpus_examples = corpus_by_dataset[dataset]
        stage_started = perf_counter()
        index_dir = Path(args.index_root) / args.split / dataset
        retriever = _load_or_build_retriever(
            corpus_examples,
            index_dir,
            args.rebuild_index,
            split=args.split,
            embedding_model=shared_embedding_model,
            reranker=shared_reranker,
        )
        shared_embedding_model = retriever.model or shared_embedding_model
        _log(f"[2/5] {dataset} 검색 인덱스 준비 완료 · {_duration(perf_counter() - stage_started)}")

        if args.prepare_index_only:
            continue

        rows_path = Path(args.rows_root) / f"{dataset}_{args.split}.jsonl"
        label_config = _label_config(dataset, corpus_fingerprints[dataset], args)
        rows = _load_or_label_actions(
            retriever,
            examples,
            rows_path,
            label_config,
            top_k=args.top_k,
            penalty=args.extra_candidate_penalty,
            joint_weight=args.joint_recall_weight,
            log_every=args.label_log_every,
            relabel=args.relabel,
        )
        shared_embedding_model = retriever.model or shared_embedding_model
        shared_reranker = retriever.reranker or shared_reranker
        all_rows.extend(rows)

    if args.prepare_index_only:
        _log(f"고정 corpus 인덱스 준비 완료 · 총 소요 {_duration(perf_counter() - started)}")
        return

    train_rows, dev_rows = _split_rows(all_rows, args.dev_ratio)
    _log(f"[4/5] 통합 정책 학습 시작: train={len(train_rows):,}, dev={len(dev_rows):,}")
    train_features, train_labels = _arrays(train_rows)
    train_utilities = _utility_matrix(train_rows)
    training_weights = _training_weights(
        train_rows,
        graph_positive_weight=args.graph_positive_weight,
        balance_datasets=not args.no_dataset_balance,
    )
    graph_positive_rows = sum(_has_graph_advantage(row) for row in train_rows)
    _log(
        f"      graph 우위 학습 행 {graph_positive_rows:,}개 · 가중치 {args.graph_positive_weight:.1f}배 "
        f"· 데이터셋 균형={'끔' if args.no_dataset_balance else '켬'}"
    )
    stage_started = perf_counter()
    policy = LinearUtilityPolicy.fit(
        train_features,
        train_utilities,
        sample_weights=training_weights,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        progress_callback=_log_training_progress,
        report_every=args.epoch_log_every,
    )
    _log(f"[4/5] 정책 학습 완료 · {_duration(perf_counter() - stage_started)}")
    model_path = Path(args.model_out)
    policy.save(model_path)
    _write_model_manifest(
        model_path,
        args,
        train_rows,
        dev_rows,
        corpus_fingerprints=corpus_fingerprints,
        query_fingerprints=query_fingerprints,
    )
    _log(f"[5/5] 모델 저장 완료: {model_path}")
    _report(policy, train_rows, dev_rows)
    print(f"\n학습 행 {len(train_rows):,}개 · 내부 개발 행 {len(dev_rows):,}개", flush=True)
    print(f"학습 행동 라벨: {dict(sorted(Counter(train_labels).items()))}", flush=True)
    print(f"정책 모델: {model_path}", flush=True)
    _log(f"전체 완료 · 총 소요 {_duration(perf_counter() - started)}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HotpotQA·MuSiQue·2Wiki 통합 검색 행동 정책 학습")
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--rows-root", default="data/benchmarks/multihop/fixed_corpus/policy_rows")
    parser.add_argument("--model-out", default="data/models/multihop_action_policy_fixed_corpus.npz")
    parser.add_argument(
        "--max-examples-per-dataset",
        "--max-queries-per-dataset",
        dest="max_examples_per_dataset",
        type=int,
        default=2000,
        help="정책 라벨을 만들 query 수. 0이면 고정 corpus cohort의 모든 query",
    )
    parser.add_argument(
        "--corpus-examples-per-dataset",
        type=int,
        default=2000,
        help="고정 pooled corpus를 구성할 문항 수. 0이면 split 전체 문서 사용",
    )
    parser.add_argument("--corpus-seed", type=int, default=42, help="고정 corpus cohort 표본 seed")
    parser.add_argument("--query-seed", type=int, default=42, help="고정 corpus 안의 query 표본 seed")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--joint-recall-weight", type=float, default=0.70)
    parser.add_argument("--extra-candidate-penalty", type=float, default=0.02)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--graph-positive-weight", type=float, default=8.0)
    parser.add_argument("--label-log-every", type=int, default=25)
    parser.add_argument("--epoch-log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-dataset-balance", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--prepare-index-only",
        action="store_true",
        help="정책 라벨/학습 없이 고정 corpus 인덱스만 준비",
    )
    parser.add_argument("--relabel", action="store_true", help="기존 행동 라벨을 버리고 다시 생성")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.max_examples_per_dataset < 0 or args.corpus_examples_per_dataset < 0:
        parser.error("query 수와 corpus 문항 수는 0 이상이어야 합니다.")
    if (
        not args.prepare_index_only
        and args.max_examples_per_dataset
        and args.corpus_examples_per_dataset
        and args.max_examples_per_dataset > args.corpus_examples_per_dataset
    ):
        parser.error("--max-examples-per-dataset은 --corpus-examples-per-dataset보다 클 수 없습니다.")
    if args.top_k < 1 or args.label_log_every < 1 or args.epoch_log_every < 1:
        parser.error("top-k와 로그 간격은 1 이상이어야 합니다.")
    if not 0.0 <= args.joint_recall_weight <= 1.0:
        parser.error("--joint-recall-weight는 0과 1 사이여야 합니다.")
    if args.extra_candidate_penalty < 0.0:
        parser.error("--extra-candidate-penalty는 0 이상이어야 합니다.")
    if args.graph_positive_weight < 1.0:
        parser.error("--graph-positive-weight는 1 이상이어야 합니다.")
    return args


def load_processed_examples(path: Path, maximum: int, seed: int) -> list[MultiHopExample]:
    records = _reservoir_sample(_iter_jsonl(path), maximum, seed)
    examples = [_normalise_processed_record(record) for record in records]
    return sorted((example for example in examples if example is not None), key=lambda item: item.example_id)


def select_query_examples(
    corpus_examples: Iterable[MultiHopExample],
    maximum: int,
    seed: int,
) -> list[MultiHopExample]:
    """Select a deterministic, nested query cohort from a fixed corpus cohort.

    Hash ranking makes the 100-query cohort a strict subset of the 500-query
    cohort for the same corpus and seed.  Corpus construction is therefore
    invariant when only the query count changes.
    """
    examples = tuple(corpus_examples)
    if maximum < 0:
        raise ValueError("maximum은 0 이상이어야 합니다.")
    if maximum == 0 or maximum >= len(examples):
        return sorted(examples, key=lambda item: item.policy_id)
    ranked = sorted(
        examples,
        key=lambda item: (
            hashlib.sha256(f"{seed}\0{item.policy_id}".encode("utf-8")).digest(),
            item.policy_id,
        ),
    )
    return sorted(ranked[:maximum], key=lambda item: item.policy_id)


def _normalise_processed_record(record: Mapping) -> MultiHopExample | None:
    dataset = str(record.get("dataset", ""))
    example_id = str(record.get("id", ""))
    question = str(record.get("question", "")).strip()
    if dataset not in SUPPORTED_DATASETS or not example_id or not question:
        return None
    documents: list[CorpusDocument] = []
    local_to_content: dict[str, str] = {}
    for raw_document in record.get("documents", []):
        if not isinstance(raw_document, Mapping):
            continue
        doc_id = str(raw_document.get("doc_id", ""))
        content_id = str(raw_document.get("content_id", ""))
        title = str(raw_document.get("title", "")).strip()
        text = str(raw_document.get("text", "")).strip()
        if not doc_id or not content_id or not text:
            continue
        local_to_content[doc_id] = content_id
        documents.append(CorpusDocument(content_id, title, text))

    support_groups: list[frozenset[str]] = []
    for support in record.get("gold_supports", []):
        if not isinstance(support, Mapping):
            continue
        group = frozenset(
            local_to_content[doc_id]
            for doc_id in support.get("doc_ids", [])
            if doc_id in local_to_content
        )
        if group and group not in support_groups:
            support_groups.append(group)
    if not documents or not support_groups:
        return None
    return MultiHopExample(
        dataset=dataset,
        example_id=example_id,
        question=question,
        documents=tuple(documents),
        supporting_groups=tuple(support_groups),
        hop_count=_optional_int(record.get("hop_count")),
        reasoning_type=str(record["reasoning_type"]) if record.get("reasoning_type") else None,
        answer=str(record["answer"]).strip() if record.get("answer") is not None else None,
        answer_aliases=tuple(
            str(alias).strip()
            for alias in (record.get("answer_aliases") or [])
            if str(alias).strip()
        ),
    )


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"전처리 JSONL {line_number}행이 손상됐습니다: {path}") from error
            if not isinstance(record, dict):
                raise ValueError(f"전처리 JSONL {line_number}행이 객체가 아닙니다: {path}")
            yield record


def _reservoir_sample(records: Iterable[Mapping], maximum: int, seed: int) -> list[Mapping]:
    if maximum == 0:
        return list(records)
    rng = random.Random(seed)
    reservoir: list[Mapping] = []
    for position, record in enumerate(records):
        if position < maximum:
            reservoir.append(record)
            continue
        replacement = rng.randrange(position + 1)
        if replacement < maximum:
            reservoir[replacement] = record
    return reservoir


def _load_or_build_retriever(
    corpus_examples: list[MultiHopExample],
    index_dir: Path,
    rebuild: bool,
    *,
    split: str,
    embedding_model=None,
    reranker=None,
) -> SemanticRetriever:
    retriever = SemanticRetriever(index_dir)
    retriever.model = embedding_model
    retriever.reranker = reranker
    corpus_fingerprint = _corpus_fingerprint(corpus_examples)
    manifest_path = index_dir / "multihop_policy_manifest.json"
    if not rebuild and _matches_index_manifest(manifest_path, corpus_fingerprint) and retriever.load():
        _log(f"[2/5] 기존 {corpus_examples[0].dataset} 고정 corpus 인덱스 재사용: {index_dir}")
        return retriever

    documents = _unique_corpus_documents(corpus_examples)
    source_documents = [
        SourceDocument(source=document.content_id, title=document.title, text=document.text)
        for document in sorted(documents.values(), key=lambda item: item.content_id)
    ]
    chunks = chunk_documents(source_documents, settings.chunk_size, settings.chunk_overlap)
    _log(
        f"[2/5] {corpus_examples[0].dataset} 새 고정 corpus 인덱스: "
        f"문서 {len(source_documents):,}개 · 청크 {len(chunks):,}개\n"
        "      임베딩 진행률이 아래 표시됩니다."
    )
    retriever.build(chunks, show_embedding_progress=True)
    manifest_path.write_text(
        json.dumps(
            {
                "corpus_fingerprint": corpus_fingerprint,
                "dataset": corpus_examples[0].dataset,
                "split": split,
                "split_corpus_examples": len(corpus_examples),
                "documents": len(source_documents),
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "graph_format_version": GRAPH_FORMAT_VERSION,
                **_retrieval_config(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _log(
        f"[2/5] {corpus_examples[0].dataset} 고정 corpus 인덱스 저장: title bridge 간선 "
        f"{retriever.graph_title_bridge_edges:,}개"
    )
    return retriever


def _unique_corpus_documents(examples: Iterable[MultiHopExample]) -> dict[str, CorpusDocument]:
    documents: dict[str, CorpusDocument] = {}
    for example in examples:
        for document in example.documents:
            existing = documents.setdefault(document.content_id, document)
            if existing.title != document.title or existing.text != document.text:
                raise ValueError(f"동일 content_id에 서로 다른 문서가 연결됐습니다: {document.content_id}")
    return documents


def _corpus_fingerprint(examples: Iterable[MultiHopExample]) -> str:
    """Fingerprint only index-visible text, never the sampled query IDs."""
    digest = hashlib.sha256()
    documents = _unique_corpus_documents(examples)
    for document in sorted(documents.values(), key=lambda item: item.content_id):
        for value in (document.content_id, document.title, document.text):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _example_fingerprint(examples: Iterable[MultiHopExample]) -> str:
    digest = hashlib.sha256()
    for example in sorted(examples, key=lambda item: item.policy_id):
        encoded = example.policy_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _matches_index_manifest(path: Path, fingerprint: str) -> bool:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        return (
            saved.get("corpus_fingerprint") == fingerprint
            and saved.get("chunk_size") == settings.chunk_size
            and saved.get("chunk_overlap") == settings.chunk_overlap
            and saved.get("graph_format_version") == GRAPH_FORMAT_VERSION
            and all(saved.get(key) == value for key, value in _retrieval_config().items())
        )
    except (OSError, ValueError, TypeError):
        return False


def _label_config(dataset: str, corpus_fingerprint: str, args: argparse.Namespace) -> dict:
    return {
        "format_version": LABEL_FORMAT_VERSION,
        "dataset": dataset,
        "split": args.split,
        "corpus_fingerprint": corpus_fingerprint,
        "query_sampling_version": QUERY_SAMPLING_VERSION,
        "top_k": args.top_k,
        "extra_candidate_penalty": args.extra_candidate_penalty,
        "joint_recall_weight": args.joint_recall_weight,
        "actions": list(POLICY_ACTION_IDS),
        "feature_names": list(POLICY_FEATURE_NAMES),
        "graph_format_version": GRAPH_FORMAT_VERSION,
        "retrieval_config": _retrieval_config(),
    }


def _load_or_label_actions(
    retriever: SemanticRetriever,
    examples: list[MultiHopExample],
    rows_path: Path,
    config: dict,
    *,
    top_k: int,
    penalty: float,
    joint_weight: float,
    log_every: int,
    relabel: bool,
) -> list[dict]:
    manifest_path = rows_path.with_suffix(".manifest.json")
    cached_rows: dict[str, dict] = {}
    if rows_path.exists() or manifest_path.exists():
        if not relabel and _matching_label_manifest(manifest_path, config):
            cached_rows = {row["id"]: row for row in _iter_jsonl(rows_path)} if rows_path.exists() else {}
        elif not relabel:
            raise SystemExit(
                f"기존 행동 라벨 설정이 현재 실행과 다릅니다: {rows_path}\n"
                "다른 --rows-root를 사용하거나 --relabel로 다시 생성하세요."
            )

    expected_ids = {example.policy_id for example in examples}
    selected_rows = {row_id: row for row_id, row in cached_rows.items() if row_id in expected_ids}
    if len(selected_rows) == len(examples):
        _log(
            f"[3/5] {examples[0].dataset} 행동 라벨 {len(selected_rows):,}개 재사용 "
            f"(전체 cache {len(cached_rows):,}개): {rows_path}"
        )
        return [selected_rows[example.policy_id] for example in examples]

    rows_path.parent.mkdir(parents=True, exist_ok=True)
    if relabel or not cached_rows:
        cached_rows = {}
        selected_rows = {}
        mode = "w"
    else:
        mode = "a"
        _log(
            f"[3/5] {examples[0].dataset} 요청 cohort 라벨 {len(selected_rows):,}/{len(examples):,}개를 "
            f"전체 cache {len(cached_rows):,}개에서 재사용하고 이어서 생성합니다."
        )
    _write_label_manifest(
        manifest_path,
        config,
        status="in_progress",
        cached_rows=len(cached_rows),
        requested_rows=len(examples),
        requested_completed=len(selected_rows),
    )

    label_counts = Counter(str(row["label"]) for row in selected_rows.values())
    oracle_utility_total = sum(max(action["utility"] for action in row["actions"].values()) for row in selected_rows.values())
    oracle_joint_total = sum(
        max(action["all_support_recall"] for action in row["actions"].values()) for row in selected_rows.values()
    )
    started = perf_counter()
    newly_completed = 0
    with rows_path.open(mode, encoding="utf-8") as destination:
        for example in examples:
            if example.policy_id in selected_rows:
                continue
            initial = retriever.search(example.question, top_k=max(top_k * 3, 12))
            profile = retriever._evidence_profile(initial)
            actions = _evaluate_actions(retriever, example, initial, profile, top_k, penalty, joint_weight)
            label = _best_action(actions)
            row = {
                "id": example.policy_id,
                "example_id": example.example_id,
                "dataset": example.dataset,
                "hop_count": example.hop_count,
                "reasoning_type": example.reasoning_type,
                "question": example.question,
                "features": dict(zip(POLICY_FEATURE_NAMES, profile_features(profile).tolist(), strict=True)),
                "label": label,
                "actions": actions,
            }
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            destination.flush()
            cached_rows[example.policy_id] = row
            selected_rows[example.policy_id] = row
            label_counts[label] += 1
            oracle_utility_total += max(action["utility"] for action in actions.values())
            oracle_joint_total += max(action["all_support_recall"] for action in actions.values())
            newly_completed += 1
            completed = len(selected_rows)
            if completed % log_every == 0 or completed == len(examples):
                elapsed = perf_counter() - started
                eta = elapsed / max(1, newly_completed) * (len(examples) - completed)
                distribution = ", ".join(
                    f"{action}={count / completed:.0%}" for action, count in sorted(label_counts.items())
                )
                _log(
                    f"[3/5] {example.dataset} 라벨 {completed:,}/{len(examples):,} ({completed / len(examples):.1%}) "
                    f"| ETA {_duration(eta)} | Oracle 효용={oracle_utility_total / completed:.3f} "
                    f"| Oracle All-Support={oracle_joint_total / completed:.3f} | 행동: {distribution}"
                )
                _write_label_manifest(
                    manifest_path,
                    config,
                    status="in_progress",
                    cached_rows=len(cached_rows),
                    requested_rows=len(examples),
                    requested_completed=completed,
                )

    _write_label_manifest(
        manifest_path,
        config,
        status="ready",
        cached_rows=len(cached_rows),
        requested_rows=len(examples),
        requested_completed=len(selected_rows),
    )
    _log(
        f"[3/5] {examples[0].dataset} 요청 행동 라벨 {len(selected_rows):,}개 준비 완료 "
        f"(전체 cache {len(cached_rows):,}개): {rows_path}"
    )
    return [selected_rows[example.policy_id] for example in examples]


def _matching_label_manifest(path: Path, expected: Mapping) -> bool:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        return all(saved.get(key) == value for key, value in expected.items())
    except (OSError, ValueError, TypeError):
        return False


def _write_label_manifest(
    path: Path,
    config: Mapping,
    *,
    status: str,
    cached_rows: int,
    requested_rows: int,
    requested_completed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(config)
    payload.update(
        {
            "status": status,
            "cached_rows": cached_rows,
            "last_requested_rows": requested_rows,
            "last_requested_completed": requested_completed,
        }
    )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _evaluate_actions(
    retriever: SemanticRetriever,
    example: MultiHopExample,
    initial: list[tuple],
    profile: EvidenceProfile,
    top_k: int,
    penalty: float,
    joint_weight: float,
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    for action_id in POLICY_ACTION_IDS:
        sources, extra_candidates = retrieve_for_action(
            retriever,
            example.question,
            initial,
            profile,
            action_id,
            top_k,
        )
        retrieved_sources = {chunk.source for chunk, _ in sources}
        source_recall, all_support_recall = evidence_metrics(retrieved_sources, example.supporting_groups)
        utility = (
            joint_weight * all_support_recall
            + (1.0 - joint_weight) * source_recall
            - penalty * extra_candidates
        )
        results[action_id] = {
            "source_recall": source_recall,
            "all_support_recall": all_support_recall,
            "extra_candidates": extra_candidates,
            "utility": utility,
        }
    return results


def retrieve_for_action(
    retriever: SemanticRetriever,
    question: str,
    initial: list[tuple],
    profile: EvidenceProfile,
    action_id: str,
    top_k: int,
) -> tuple[list[tuple], int]:
    """Execute one bounded retrieval action for training and QA evaluation."""
    decision = retriever.rule_router.decision_for_action(profile, action_id, reason="oracle_action")
    if decision.graph_hops == 0:
        return initial[:top_k], 0
    expanded = retriever._expand_for_plan(initial, decision)
    sources = retriever._rerank_graph_expansion(question, initial, expanded, top_k)
    return sources, len(expanded) - len(initial)


def evidence_metrics(
    retrieved_sources: set[str],
    supporting_groups: tuple[frozenset[str], ...],
) -> tuple[float, float]:
    if not supporting_groups:
        return 0.0, 0.0
    recovered = sum(bool(retrieved_sources & group) for group in supporting_groups)
    recall = recovered / len(supporting_groups)
    return recall, float(recovered == len(supporting_groups))


def _best_action(actions: Mapping[str, Mapping]) -> str:
    return max(
        POLICY_ACTION_IDS,
        key=lambda action: (
            float(actions[action]["utility"]),
            float(actions[action]["all_support_recall"]),
            float(actions[action]["source_recall"]),
            -int(actions[action]["extra_candidates"]),
            -POLICY_ACTION_IDS.index(action),
        ),
    )


def _split_rows(rows: list[dict], dev_ratio: float) -> tuple[list[dict], list[dict]]:
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError("dev_ratio는 0과 1 사이여야 합니다.")
    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    for row in rows:
        bucket = int(hashlib.sha256(row["id"].encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        (dev_rows if bucket < dev_ratio else train_rows).append(row)
    if not train_rows or not dev_rows:
        raise ValueError("학습/개발 분할이 비었습니다. 문항 수를 늘리세요.")
    return train_rows, dev_rows


def _arrays(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    return np.asarray(
        [[row["features"][name] for name in POLICY_FEATURE_NAMES] for row in rows], dtype="float64"
    ), [str(row["label"]) for row in rows]


def _utility_matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[float(row["actions"][action]["utility"]) for action in POLICY_ACTION_IDS] for row in rows],
        dtype="float64",
    )


def _has_graph_advantage(row: Mapping) -> bool:
    vector = float(row["actions"]["vector"]["utility"])
    graph = max(float(row["actions"][action]["utility"]) for action in POLICY_ACTION_IDS if action != "vector")
    return graph > vector + 1e-9


def _training_weights(
    rows: list[dict],
    *,
    graph_positive_weight: float,
    balance_datasets: bool,
) -> np.ndarray:
    counts = Counter(str(row["dataset"]) for row in rows)
    dataset_count = len(counts)
    weights: list[float] = []
    for row in rows:
        dataset_weight = len(rows) / (dataset_count * counts[str(row["dataset"])]) if balance_datasets else 1.0
        graph_weight = graph_positive_weight if _has_graph_advantage(row) else 1.0
        weights.append(dataset_weight * graph_weight)
    return np.asarray(weights, dtype="float64")


def _profile_from_row(row: Mapping) -> EvidenceProfile:
    features = row["features"]
    return EvidenceProfile(
        score_concentration=features["score_concentration"],
        score_margin=features["score_margin"],
        source_dispersion=features["source_dispersion"],
        graph_connectivity=features["graph_connectivity"],
        redundancy=features["redundancy"],
        one_hop_gain=features["one_hop_gain"],
        two_hop_gain=features["two_hop_gain"],
    )


def _report(policy: LinearUtilityPolicy, train_rows: list[dict], dev_rows: list[dict]) -> None:
    del train_rows
    print("\n[내부 개발 분할]", flush=True)
    _report_group(policy, dev_rows, "전체")
    for dataset in sorted({row["dataset"] for row in dev_rows}):
        _report_group(policy, [row for row in dev_rows if row["dataset"] == dataset], dataset)


def _report_group(policy: LinearUtilityPolicy, rows: list[dict], name: str) -> None:
    features, labels = _arrays(rows)
    predicted_utilities = policy.predict_utilities(features)
    predicted = [policy.actions[int(index)] for index in np.argmax(predicted_utilities, axis=1)]
    oracle = [_best_action(row["actions"]) for row in rows]
    rule = []
    # All datasets share the same deterministic rule baseline parameters.
    rule_router = QueryRouter(
        small_budget=settings.graph_small_budget,
        large_budget=settings.graph_large_budget,
        utility_threshold=settings.graph_utility_threshold,
    )
    for row in rows:
        rule.append(rule_router.action_id_for(rule_router.decide(_profile_from_row(row))))

    def average(actions: Iterable[str], metric: str) -> float:
        return float(np.mean([row["actions"][action][metric] for row, action in zip(rows, actions, strict=True)]))

    accuracy = sum(prediction == label for prediction, label in zip(predicted, labels, strict=True)) / len(rows)
    print(
        f"{name}: n={len(rows):,} | oracle 행동 일치율={accuracy:.3f} "
        f"| 효용 policy/vector/rule/oracle="
        f"{average(predicted, 'utility'):.3f}/{average(['vector'] * len(rows), 'utility'):.3f}/"
        f"{average(rule, 'utility'):.3f}/{average(oracle, 'utility'):.3f} "
        f"| All-Support policy/vector/oracle="
        f"{average(predicted, 'all_support_recall'):.3f}/"
        f"{average(['vector'] * len(rows), 'all_support_recall'):.3f}/"
        f"{average(oracle, 'all_support_recall'):.3f}",
        flush=True,
    )


def _write_model_manifest(
    model_path: Path,
    args: argparse.Namespace,
    train_rows: list[dict],
    dev_rows: list[dict],
    *,
    corpus_fingerprints: Mapping[str, str],
    query_fingerprints: Mapping[str, str],
) -> None:
    manifest = model_path.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "model": str(model_path),
                "datasets": list(args.datasets),
                "split": args.split,
                "corpus_examples_per_dataset": args.corpus_examples_per_dataset,
                "max_queries_per_dataset": args.max_examples_per_dataset,
                "corpus_seed": args.corpus_seed,
                "query_seed": args.query_seed,
                "policy_seed": args.seed,
                "query_sampling_version": QUERY_SAMPLING_VERSION,
                "corpus_fingerprints": dict(sorted(corpus_fingerprints.items())),
                "query_fingerprints": dict(sorted(query_fingerprints.items())),
                "top_k": args.top_k,
                "joint_recall_weight": args.joint_recall_weight,
                "extra_candidate_penalty": args.extra_candidate_penalty,
                "graph_positive_weight": args.graph_positive_weight,
                "dataset_balance": not args.no_dataset_balance,
                "train_rows": len(train_rows),
                "dev_rows": len(dev_rows),
                "train_dataset_counts": dict(sorted(Counter(row["dataset"] for row in train_rows).items())),
                "dev_dataset_counts": dict(sorted(Counter(row["dataset"] for row in dev_rows).items())),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _log_training_progress(progress: PolicyTrainingProgress) -> None:
    selected_utility = f" | train 선택 효용={progress.selected_utility:.3f}" if progress.selected_utility is not None else ""
    _log(
        f"[4/5] epoch {progress.epoch:>3}/{progress.epochs} ({progress.epoch / progress.epochs:.0%}) "
        f"| loss={progress.loss:.4f} | oracle 행동 일치율={progress.accuracy:.3f}{selected_utility} "
        f"| ETA {_duration(progress.eta_seconds)}"
    )


def _dataset_seed(seed: int, dataset: str) -> int:
    return seed ^ int(hashlib.sha256(dataset.encode("utf-8")).hexdigest()[:8], 16)


def _retrieval_config() -> dict:
    return {
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "reranker_candidates": settings.reranker_candidates,
        "router_seed_count": settings.router_seed_count,
        "graph_small_budget": settings.graph_small_budget,
        "graph_large_budget": settings.graph_large_budget,
        "graph_expansion_decay": settings.graph_expansion_decay,
        "graph_semantic_neighbors": settings.graph_semantic_neighbors,
        "graph_title_bridges": settings.graph_title_bridges,
        "graph_max_title_bridges_per_chunk": settings.graph_max_title_bridges_per_chunk,
        "graph_rerank_prior_weight": settings.graph_rerank_prior_weight,
        "graph_source_diversity_bonus": settings.graph_source_diversity_bonus,
        "graph_min_expanded_results": settings.graph_min_expanded_results,
    }


def _optional_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}시간 {minutes:02d}분 {seconds:02d}초"
    if minutes:
        return f"{minutes}분 {seconds:02d}초"
    return f"{seconds}초"


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
