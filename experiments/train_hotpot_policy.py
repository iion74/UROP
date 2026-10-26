"""Train a retrieval-action policy from an official HotpotQA JSON split.

Example:
    python -m experiments.train_hotpot_policy \
      --hotpot data/benchmarks/hotpotqa/distractor \
      --split train \
      --index-dir data/benchmarks/hotpot/index \
      --model-out data/models/hotpot_action_policy.npz \
      --max-examples 2000

The script never calls the generation LLM.  It uses HotpotQA's supporting
document annotations to supervise the expected utility of every bounded
retrieval action under an explicit extra-reranking penalty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from src.document_loader import SourceDocument, chunk_documents
from src.policy import (
    LinearUtilityPolicy,
    POLICY_FEATURE_NAMES,
    PolicyTrainingProgress,
    profile_features,
)
from src.retriever import SemanticRetriever
from src.routing import POLICY_ACTION_IDS, EvidenceProfile
from src.settings import settings


@dataclass(frozen=True)
class HotpotExample:
    example_id: str
    question: str
    contexts: tuple[tuple[str, str], ...]
    supporting_sources: frozenset[str]


def main() -> None:
    args = _arguments()
    run_started = perf_counter()
    _log("HotpotQA 검색 행동 정책 학습을 시작합니다. (생성 LLM은 사용하지 않습니다.)")
    _log(
        "설정: "
        f"split={args.split}, max_examples={args.max_examples or '전체'}, top_k={args.top_k}, "
        f"epochs={args.epochs}, graph_positive_weight={args.graph_positive_weight}, seed={args.seed}"
    )

    stage_started = perf_counter()
    examples = _load_hotpot(Path(args.hotpot), args.split, args.max_examples, args.seed)
    if not examples:
        raise SystemExit("유효한 HotpotQA 예시를 찾지 못했습니다.")
    _log(f"[1/5] 데이터 준비 완료: 유효 질문 {len(examples):,}개 · {_duration(perf_counter() - stage_started)}")

    stage_started = perf_counter()
    retriever = _load_or_build_retriever(examples, Path(args.index_dir), args.rebuild_index)
    _log(f"[2/5] 검색 인덱스 준비 완료 · {_duration(perf_counter() - stage_started)}")

    _log(f"[3/5] 행동 oracle 라벨 생성 시작: 질문 {len(examples):,}개 · 행동 {len(POLICY_ACTION_IDS)}개/질문")
    stage_started = perf_counter()
    rows = _label_actions(
        retriever,
        examples,
        args.top_k,
        args.extra_candidate_penalty,
        args.label_log_every,
    )
    rows_path = Path(args.rows_out)
    _write_jsonl(rows_path, rows)
    _log(f"[3/5] 라벨 파일 저장 완료: {rows_path} · {_duration(perf_counter() - stage_started)}")

    train_rows, dev_rows = _split_rows(rows, args.dev_ratio)
    _log(f"[4/5] 정책 학습 시작: train={len(train_rows):,}, dev={len(dev_rows):,}")
    stage_started = perf_counter()
    train_features, train_labels = _arrays(train_rows)
    train_utilities = _utility_matrix(train_rows)
    training_weights = _graph_positive_weights(train_rows, args.graph_positive_weight)
    _log(
        f"      graph 우위 학습 행 {int((training_weights > 1).sum()):,}개 · "
        f"가중치 {args.graph_positive_weight:.1f}배"
    )
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
    policy.save(Path(args.model_out))
    _log(f"[5/5] 모델 저장 완료: {args.model_out}")
    _report(policy, retriever, train_rows, dev_rows)
    print(f"\n학습 행 {len(train_rows)}개 · 개발 행 {len(dev_rows)}개")
    print(f"행동 라벨: {dict(sorted(Counter(train_labels).items()))}")
    print(f"정책 모델: {args.model_out}")
    print(f"라벨 데이터: {rows_path}")
    _log(f"전체 완료 · 총 소요 {_duration(perf_counter() - run_started)}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HotpotQA supporting-evidence action policy training")
    parser.add_argument("--hotpot", required=True, help="공식 HotpotQA JSON 파일 또는 datasets 저장 폴더")
    parser.add_argument("--split", default="train", help="datasets 저장 폴더일 때 사용할 split")
    parser.add_argument("--index-dir", required=True, help="이 split 전용 FAISS 인덱스 경로")
    parser.add_argument("--model-out", required=True, help="저장할 .npz 정책 모델 경로")
    parser.add_argument("--rows-out", default="data/benchmarks/hotpot/policy_rows.jsonl", help="생성된 행동 라벨 경로")
    parser.add_argument("--max-examples", type=int, default=2000, help="0이면 전체 예시 사용")
    parser.add_argument("--top-k", type=int, default=5, help="최종 근거 수")
    parser.add_argument("--extra-candidate-penalty", type=float, default=0.02, help="추가 reranking 후보 한 개당 효용 패널티")
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument(
        "--graph-positive-weight",
        type=float,
        default=8.0,
        help="그래프 행동이 vector보다 높은 효용을 낸 학습 행의 가중치",
    )
    parser.add_argument("--label-log-every", type=int, default=25, help="oracle 라벨 진행 로그 간격(질문 수)")
    parser.add_argument("--epoch-log-every", type=int, default=5, help="정책 학습 진행 로그 간격(epoch)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


def _load_hotpot(path: Path, split: str = "train", maximum: int = 0, seed: int = 42) -> list[HotpotExample]:
    if path.is_dir():
        from datasets import load_from_disk

        saved = load_from_disk(str(path))
        if not hasattr(saved, "keys") or split not in saved:
            raise ValueError(f"저장된 HotpotQA에서 split '{split}'을 찾지 못했습니다.")
        raw = saved[split]
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("HotpotQA JSON은 질문 레코드 배열이어야 합니다.")
    raw = _sample_records(raw, maximum, seed)
    examples: list[HotpotExample] = []
    for record in raw:
        contexts = _normalise_context(record.get("context", []))
        supporting_sources = _normalise_supporting_sources(record.get("supporting_facts", []))
        question = record.get("question")
        if isinstance(question, str) and contexts and supporting_sources:
            examples.append(
                HotpotExample(
                    example_id=str(record.get("_id", record.get("id", len(examples)))),
                    question=question,
                    contexts=tuple(contexts),
                    supporting_sources=supporting_sources,
                )
            )
    return examples


def _normalise_context(raw_context) -> list[tuple[str, str]]:
    """Accept official list-pair JSON and Hugging Face's column-dict form."""
    if isinstance(raw_context, dict):
        raw_context = list(zip(raw_context.get("title", []), raw_context.get("sentences", []), strict=False))
    contexts: list[tuple[str, str]] = []
    for item in raw_context:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        title, sentences = item
        if not isinstance(title, str) or not isinstance(sentences, list):
            continue
        text = " ".join(sentence.strip() for sentence in sentences if isinstance(sentence, str)).strip()
        if text:
            contexts.append((title, text))
    return contexts


def _normalise_supporting_sources(raw_facts) -> frozenset[str]:
    if isinstance(raw_facts, dict):
        raw_facts = list(zip(raw_facts.get("title", []), raw_facts.get("sent_id", []), strict=False))
    return frozenset(
        item[0]
        for item in raw_facts
        if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str)
    )


def _sample_records(records, maximum: int, seed: int):
    if maximum == 0 or maximum >= len(records):
        return records
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(len(records), size=maximum, replace=False).tolist())
    if hasattr(records, "select"):
        return records.select(selected)
    return [records[index] for index in selected]


def _load_or_build_retriever(
    examples: list[HotpotExample],
    index_dir: Path,
    rebuild: bool,
) -> SemanticRetriever:
    retriever = SemanticRetriever(index_dir)
    corpus_fingerprint = _corpus_fingerprint(examples)
    manifest_path = index_dir / "hotpot_policy_manifest.json"
    if not rebuild and retriever.load() and _matches_manifest(manifest_path, corpus_fingerprint):
        _log(f"[2/5] 기존 HotpotQA 인덱스를 재사용합니다: {index_dir}")
        return retriever

    documents: dict[str, str] = {}
    for example in examples:
        for title, text in example.contexts:
            documents.setdefault(title, text)
    source_documents = [SourceDocument(source=title, text=text) for title, text in sorted(documents.items())]
    chunks = chunk_documents(source_documents, settings.chunk_size, settings.chunk_overlap)
    _log(
        f"[2/5] 새 인덱스 생성: 문서 {len(source_documents):,}개 · 청크 {len(chunks):,}개\n"
        "      임베딩 진행률이 아래 표시됩니다. 첫 실행에서는 임베딩/리랭커 모델 다운로드가 필요할 수 있습니다."
    )
    retriever.build(chunks, show_embedding_progress=True)
    manifest_path.write_text(
        json.dumps(
            {
                "corpus_fingerprint": corpus_fingerprint,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _log(
        f"[2/5] HotpotQA 인덱스 저장 완료: 문서 {len(source_documents):,}개 · 청크 {len(chunks):,}개 "
        f"· title bridge 간선 {retriever.graph_title_bridge_edges:,}개"
    )
    return retriever


def _corpus_fingerprint(examples: list[HotpotExample]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.example_id.encode("utf-8"))
        for title, text in example.contexts:
            digest.update(title.encode("utf-8"))
            digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _matches_manifest(path: Path, corpus_fingerprint: str) -> bool:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        return (
            saved.get("corpus_fingerprint") == corpus_fingerprint
            and saved.get("chunk_size") == settings.chunk_size
            and saved.get("chunk_overlap") == settings.chunk_overlap
        )
    except (OSError, ValueError, TypeError):
        return False


def _label_actions(
    retriever: SemanticRetriever,
    examples: list[HotpotExample],
    top_k: int,
    penalty: float,
    log_every: int,
) -> list[dict]:
    if log_every < 1:
        raise ValueError("label_log_every는 1 이상이어야 합니다.")
    rows: list[dict] = []
    started = perf_counter()
    oracle_recall_total = 0.0
    oracle_utility_total = 0.0
    label_counts: Counter[str] = Counter()
    expanded_action_count = 0
    graph_action_count = 0
    added_candidate_total = 0
    for position, example in enumerate(examples, start=1):
        initial = retriever.search(example.question, top_k=max(top_k * 3, 12))
        profile = retriever._evidence_profile(initial)
        action_stats = _evaluate_actions(retriever, example, initial, profile, top_k, penalty)
        label = max(
            POLICY_ACTION_IDS,
            key=lambda action: (
                action_stats[action]["utility"],
                action_stats[action]["source_recall"],
                -action_stats[action]["extra_candidates"],
                -POLICY_ACTION_IDS.index(action),
            ),
        )
        label_counts[label] += 1
        oracle_recall_total += max(stats["source_recall"] for stats in action_stats.values())
        oracle_utility_total += max(stats["utility"] for stats in action_stats.values())
        for action_id in POLICY_ACTION_IDS:
            if action_id == "vector":
                continue
            graph_action_count += 1
            extra_candidates = int(action_stats[action_id]["extra_candidates"])
            added_candidate_total += extra_candidates
            expanded_action_count += int(extra_candidates > 0)
        rows.append(
            {
                "id": example.example_id,
                "question": example.question,
                "features": dict(zip(POLICY_FEATURE_NAMES, profile_features(profile).tolist(), strict=True)),
                "label": label,
                "actions": action_stats,
            }
        )
        if position % log_every == 0 or position == len(examples):
            elapsed = perf_counter() - started
            eta = elapsed / position * (len(examples) - position)
            distribution = ", ".join(
                f"{action}={count / position:.0%}" for action, count in sorted(label_counts.items())
            )
            _log(
                f"[3/5] 라벨 {position:,}/{len(examples):,} ({position / len(examples):.1%}) "
                f"| {elapsed / position:.2f}초/질문 | ETA {_duration(eta)} "
                f"| oracle Recall@{top_k}={oracle_recall_total / position:.3f} "
                f"| oracle 효용={oracle_utility_total / position:.3f} "
                f"| 그래프 새 후보={expanded_action_count / max(1, graph_action_count):.0%} "
                f"(평균 {added_candidate_total / max(1, graph_action_count):.1f}개/행동) "
                f"| 행동 분포: {distribution}"
            )
    return rows


def _evaluate_actions(
    retriever: SemanticRetriever,
    example: HotpotExample,
    initial: list[tuple],
    profile: EvidenceProfile,
    top_k: int,
    penalty: float,
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    for action_id in POLICY_ACTION_IDS:
        decision = retriever.rule_router.decision_for_action(profile, action_id, reason="oracle_action")
        if decision.graph_hops == 0:
            sources = initial[:top_k]
            extra_candidates = 0
        else:
            expanded = retriever._expand_for_plan(initial, decision)
            sources = retriever._rerank_graph_expansion(example.question, initial, expanded, top_k)
            extra_candidates = len(expanded) - len(initial)
        retrieved_sources = {chunk.source for chunk, _ in sources}
        source_recall = len(retrieved_sources & example.supporting_sources) / len(example.supporting_sources)
        results[action_id] = {
            "source_recall": source_recall,
            "extra_candidates": extra_candidates,
            "utility": source_recall - penalty * extra_candidates,
        }
    return results


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_rows(rows: list[dict], dev_ratio: float) -> tuple[list[dict], list[dict]]:
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError("dev_ratio는 0과 1 사이여야 합니다.")
    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    for row in rows:
        bucket = int(hashlib.sha256(row["id"].encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        (dev_rows if bucket < dev_ratio else train_rows).append(row)
    if not train_rows or not dev_rows:
        raise ValueError("학습/개발 분할이 비었습니다. --max-examples를 늘리세요.")
    return train_rows, dev_rows


def _arrays(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    features = np.asarray(
        [[row["features"][name] for name in POLICY_FEATURE_NAMES] for row in rows],
        dtype="float64",
    )
    return features, [str(row["label"]) for row in rows]


def _utility_matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[float(row["actions"][action]["utility"]) for action in POLICY_ACTION_IDS] for row in rows],
        dtype="float64",
    )


def _graph_positive_weights(rows: list[dict], graph_positive_weight: float) -> np.ndarray:
    if graph_positive_weight < 1.0:
        raise ValueError("graph_positive_weight는 1 이상이어야 합니다.")
    weights: list[float] = []
    for row in rows:
        vector_utility = float(row["actions"]["vector"]["utility"])
        graph_utility = max(float(row["actions"][action]["utility"]) for action in POLICY_ACTION_IDS if action != "vector")
        weights.append(graph_positive_weight if graph_utility > vector_utility + 1e-9 else 1.0)
    return np.asarray(weights, dtype="float64")


def _profile_from_row(row: dict) -> EvidenceProfile:
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


def _report(policy: LinearUtilityPolicy, retriever: SemanticRetriever, train_rows: list[dict], dev_rows: list[dict]) -> None:
    del train_rows
    features, labels = _arrays(dev_rows)
    predicted_utilities = policy.predict_utilities(features)
    predicted = [policy.actions[int(index)] for index in np.argmax(predicted_utilities, axis=1)]
    action_accuracy = sum(prediction == label for prediction, label in zip(predicted, labels, strict=True)) / len(labels)
    policy_utility = np.mean([row["actions"][action]["utility"] for row, action in zip(dev_rows, predicted, strict=True)])
    oracle_utility = np.mean([max(stats["utility"] for stats in row["actions"].values()) for row in dev_rows])
    rule_actions = [retriever.rule_router.action_id_for(retriever.rule_router.decide(_profile_from_row(row))) for row in dev_rows]
    rule_utility = np.mean([row["actions"][action]["utility"] for row, action in zip(dev_rows, rule_actions, strict=True)])
    vector_utility = np.mean([row["actions"]["vector"]["utility"] for row in dev_rows])
    utility_mae = np.abs(predicted_utilities - _utility_matrix(dev_rows)).mean()
    print("\n[개발 분할]")
    print(f"정책 oracle 행동 일치율: {action_accuracy:.3f}")
    print(f"제안 정책 평균 효용: {policy_utility:.3f}")
    print(f"Vector 평균 효용: {vector_utility:.3f}")
    print(f"규칙 평균 효용: {rule_utility:.3f}")
    print(f"Oracle 평균 효용: {oracle_utility:.3f}")
    print(f"행동 효용 예측 MAE: {utility_mae:.4f}")


def _log_training_progress(progress: PolicyTrainingProgress) -> None:
    selected_utility = (
        f" | train 선택 효용={progress.selected_utility:.3f}"
        if progress.selected_utility is not None
        else ""
    )
    _log(
        f"[4/5] epoch {progress.epoch:>3}/{progress.epochs} "
        f"({progress.epoch / progress.epochs:.0%}) | train loss={progress.loss:.4f} "
        f"| oracle 행동 일치율={progress.accuracy:.3f}{selected_utility} "
        f"| 경과 {_duration(progress.elapsed_seconds)} | ETA {_duration(progress.eta_seconds)}"
    )


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
