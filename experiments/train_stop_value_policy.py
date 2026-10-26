"""Train a cost-aware long-term value policy for adaptive retrieval stopping.

The frozen retrieval trajectories expose four possible answer contexts: Direct
and three successive CONTINUE prefixes.  For every non-terminal state this
trainer learns two values from information available at inference time:

* Q(STOP): Answer F1 when answering from the current evidence.
* Q(CONTINUE): cost-adjusted future answer utility after another expansion.

The policy continues only when the predicted advantage exceeds a threshold.
Question-level nested out-of-fold evaluation prevents a question from being
scored by a value network fitted on that question.  Gold answers, support
labels and gold hop counts are targets/audits only and never model inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.train_answer_policy import _bucket, _fraction, _macro_value
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _duration, _iter_jsonl
from experiments.train_stop_policy import ACTIONS, MAX_SEARCH_STEPS, _numeric_state
from src.settings import settings


FORMAT_VERSION = 7
EMBEDDING_FORMAT_VERSION = 2
NUMERIC_FEATURE_NAMES = (
    "completed_step_fraction",
    "reader_source_fraction",
    "memory_size_fraction",
    "previous_query_fraction",
    "last_added_score",
    "last_added_missing",
    "no_new_fraction",
    "query_unique_ratio",
    "last_query_repeated",
    "evidence_score_max",
    "evidence_score_mean",
    "evidence_score_min",
    "evidence_score_std",
    "evidence_score_margin",
    "evidence_changed_fraction",
    "new_evidence_fraction",
    "removed_evidence_fraction",
    "graph_frontier_fraction",
    "graph_internal_edge_fraction",
    "new_graph_frontier_fraction",
    "planner_evidence_sufficient",
    "planner_sufficiency_missing",
    "planner_gap_length_fraction",
)


@dataclass(frozen=True)
class StateExample:
    question_id: str
    dataset: str
    hop_count: int
    step: int
    text: str
    question_text: str
    evidence_texts: tuple[str, ...]
    new_evidence_texts: tuple[str, ...]
    removed_evidence_texts: tuple[str, ...]
    history_text: str
    numeric: tuple[float, ...]
    stop_utility: float
    continue_utility: float

    @property
    def advantage(self) -> float:
        return self.continue_utility - self.stop_utility


@dataclass(frozen=True)
class NumericScaler:
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class GraphStateIndex:
    graph: dict[int, tuple[int, ...]]
    positions_by_source: dict[str, tuple[int, ...]]
    source_by_position: tuple[str, ...]


class StopValueNetwork(nn.Module):
    """Small MLP over frozen semantic state embeddings and retrieval signals."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        middle = max(32, hidden_size // 2)
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, middle),
            nn.GELU(),
            nn.Dropout(dropout),
            # Q(STOP), Q(CONTINUE), and a separately calibrated benefit logit.
            nn.Linear(middle, 3),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


def main() -> None:
    args = _arguments()
    started = perf_counter()
    _seed_everything(args.seed)
    rows, trajectories, contents, graph_indices = _load_inputs(args)
    states = _state_examples(
        rows,
        trajectories,
        contents,
        graph_indices,
        args.evidence_characters,
        args.continue_target,
        args.f1_weight,
        args.em_weight,
        args.search_cost,
    )
    labels = Counter("CONTINUE" if state.advantage > 0.0 else "STOP" for state in states)
    device = _device(args.device)
    _log(
        f"Cost-aware Bellman STOP value v7 시작: {len(rows):,}문항 · 상태 {len(states):,}개 "
        f"· target={dict(sorted(labels.items()))} · device={device} · nested folds={args.folds}"
    )
    _report_static(rows, args.datasets)
    embeddings = _load_or_encode_embeddings(states, args, device)
    state_indices = _state_indices(states)
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    records, fold_gains = _nested_oof(
        rows,
        states,
        embeddings,
        state_indices,
        args,
        thresholds,
        device,
    )
    audit = _report_records(records, args.datasets)
    audit["ranking"] = _report_ranking(records, rows, args)
    model, scaler, final_threshold = _fit_final(
        rows,
        states,
        embeddings,
        state_indices,
        args,
        thresholds,
        device,
    )
    _save(model, scaler, final_threshold, rows, states, records, audit, fold_gains, args)
    _log(f"전체 완료 · 총 소요 {_duration(perf_counter() - started)}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU cost-sensitive long-term STOP/CONTINUE value policy")
    parser.add_argument("--answer-root", default="data/benchmarks/multihop/stop_answer_oracle_v1_1000")
    parser.add_argument("--trajectory-root", default="data/benchmarks/multihop/sequential_search_v5_4hop_1000")
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--max-rows-per-dataset", type=int, default=0)
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    parser.add_argument("--embedding-cache", default="data/cache/stop_value_policy_v2_1000_embeddings.npz")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--evidence-characters",
        type=int,
        default=1400,
        help="문서별 최대 글자 수. 문서를 개별 임베딩하므로 전체 합계가 아닙니다.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gain-weight", type=float, default=3.0)
    parser.add_argument("--classification-weight", type=float, default=0.35)
    parser.add_argument(
        "--continue-positive-weight",
        type=float,
        default=2.0,
        help="희소한 유익 CONTINUE 상태의 분류 손실 가중치",
    )
    parser.add_argument("--f1-weight", type=float, default=0.5)
    parser.add_argument("--em-weight", type=float, default=0.5)
    parser.add_argument(
        "--continue-target",
        choices=("bellman", "next_step", "best_future"),
        default="bellman",
        help=(
            "bellman은 검색 비용을 포함한 장기 최적 중단 가치, next_step은 바로 다음 "
            "검색의 실제 ΔEM/F1, best_future는 비용 없는 oracle-style ablation입니다."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--threshold-min", type=float, default=-20.0)
    parser.add_argument("--threshold-max", type=float, default=5.0)
    parser.add_argument("--threshold-steps", type=int, default=251)
    parser.add_argument("--dataset-harm-penalty", type=float, default=2.0)
    parser.add_argument(
        "--balance-groups",
        choices=("dataset", "dataset_hop"),
        default="dataset",
        help=(
            "학습 가중치와 threshold 보정에 사용할 집단. dataset은 gold hop을 전혀 "
            "사용하지 않으며, dataset_hop은 이전 v6 재현용 ablation입니다."
        ),
    )
    parser.add_argument(
        "--search-cost",
        type=float,
        default=0.002,
        help="추가 검색 1회에 부여할 정규화된 EM/F1 utility 비용",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default="data/models/stop_value_policy_v3_step_thresholds_1000.pt")
    parser.add_argument("--predictions-out")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if args.max_rows_per_dataset < 0 or args.folds < 3:
        parser.error("max rows는 0 이상, folds는 3 이상이어야 합니다.")
    if not 0.0 < args.calibration_ratio < 0.5:
        parser.error("calibration ratio는 0과 0.5 사이여야 합니다.")
    if args.epochs < 1 or args.batch_size < 1 or args.embedding_batch_size < 1:
        parser.error("epoch와 batch 크기는 1 이상이어야 합니다.")
    if args.hidden_size < 16 or not 0.0 <= args.dropout < 1.0:
        parser.error("hidden size/dropout 설정을 확인하세요.")
    if args.threshold_steps < 2 or args.threshold_min >= args.threshold_max:
        parser.error("threshold 범위를 확인하세요.")
    if (
        args.search_cost < 0.0
        or args.gain_weight < 0.0
        or args.classification_weight < 0.0
        or args.continue_positive_weight <= 0.0
    ):
        parser.error("cost와 loss 가중치는 0 이상이어야 합니다.")
    if args.f1_weight < 0.0 or args.em_weight < 0.0 or args.f1_weight + args.em_weight <= 0.0:
        parser.error("F1/EM 가중치는 0 이상이며 합이 0보다 커야 합니다.")
    total_metric_weight = args.f1_weight + args.em_weight
    args.f1_weight /= total_metric_weight
    args.em_weight /= total_metric_weight
    return args


def _load_inputs(
    args,
) -> tuple[list[dict], dict[str, dict], dict[str, dict], dict[str, GraphStateIndex]]:
    rows: list[dict] = []
    trajectories: dict[str, dict] = {}
    contents: dict[str, dict] = {}
    graph_indices: dict[str, GraphStateIndex] = {}
    for dataset in args.datasets:
        answer_path = Path(args.answer_root) / f"{dataset}_{args.split}.jsonl"
        trajectory_path = Path(args.trajectory_root) / f"{dataset}_{args.split}.jsonl"
        answer_rows = list(_iter_jsonl(answer_path))
        trajectory_rows = list(_iter_jsonl(trajectory_path))
        if args.max_rows_per_dataset:
            answer_rows = answer_rows[: args.max_rows_per_dataset]
        answer_ids = {str(row["id"]) for row in answer_rows}
        trajectory_by_id = {str(row["id"]): row for row in trajectory_rows if str(row["id"]) in answer_ids}
        if len(trajectory_by_id) != len(answer_rows):
            raise SystemExit(f"{dataset}: Answer와 trajectory ID가 일치하지 않습니다.")
        chunk_path = Path(args.index_root) / args.split / dataset / "chunks.json"
        raw_chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
        by_source: dict[str, list[dict]] = defaultdict(list)
        positions_by_source: dict[str, list[int]] = defaultdict(list)
        for position, chunk in enumerate(raw_chunks):
            by_source[str(chunk["source"])].append(chunk)
            positions_by_source[str(chunk["source"])].append(position)
        for source, chunks in by_source.items():
            contents[source] = {
                "title": str(chunks[0].get("title", "")),
                "text": " ".join(str(chunk.get("text", "")) for chunk in chunks),
            }
        graph_path = Path(args.index_root) / args.split / dataset / "chunk_graph.json"
        raw_graph = json.loads(graph_path.read_text(encoding="utf-8"))
        graph_indices[dataset] = GraphStateIndex(
            graph={int(key): tuple(int(value) for value in values) for key, values in raw_graph.items()},
            positions_by_source={key: tuple(values) for key, values in positions_by_source.items()},
            source_by_position=tuple(str(chunk["source"]) for chunk in raw_chunks),
        )
        for row in answer_rows:
            if tuple(row.get("actions", {})) != ACTIONS or len(row.get("planner_audit", [])) != MAX_SEARCH_STEPS:
                raise SystemExit(f"{dataset}: 지원하지 않는 STOP Answer 행입니다: {row.get('id')}")
        rows.extend(answer_rows)
        trajectories.update(trajectory_by_id)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)) or len(rows) < max(20, len(args.datasets) * 10):
        raise SystemExit("중복 ID가 있거나 정책 학습 행이 너무 적습니다.")
    return rows, trajectories, contents, graph_indices


def _state_examples(
    rows: list[dict],
    trajectories: dict[str, dict],
    contents: dict[str, dict],
    graph_indices: dict[str, GraphStateIndex],
    evidence_characters: int,
    continue_target: str,
    f1_weight: float,
    em_weight: float,
    search_cost: float,
) -> list[StateExample]:
    examples: list[StateExample] = []
    for row in rows:
        trajectory = trajectories[str(row["id"])]
        queries = [str(item.get("query", "")) for item in row["planner_audit"]]
        stop_utilities, continue_utilities = _decision_utilities(
            row, continue_target, f1_weight, em_weight, search_cost
        )
        for step in range(MAX_SEARCH_STEPS):
            current_action = ACTIONS[step]
            evidence = trajectory["actions"][current_action].get("evidence", [])
            prior_evidence = (
                trajectory["actions"][ACTIONS[step - 1]].get("evidence", []) if step else []
            )
            graph_index = graph_indices[str(row["dataset"])]
            numeric = _value_numeric(
                row["planner_audit"][step]["policy_state"],
                queries[:step],
                evidence,
                prior_evidence,
                graph_index,
            )
            current_ids = {str(item.get("content_id", "")) for item in evidence}
            prior_ids = {str(item.get("content_id", "")) for item in prior_evidence}
            evidence_texts = tuple(
                _document_text(item, contents, evidence_characters) for item in evidence
            )
            new_evidence_texts = tuple(
                _document_text(item, contents, evidence_characters)
                for item in evidence
                if str(item.get("content_id", "")) not in prior_ids
            )
            removed_evidence_texts = tuple(
                _document_text(item, contents, evidence_characters)
                for item in prior_evidence
                if str(item.get("content_id", "")) not in current_ids
            )
            state_audit = row["planner_audit"][step]["policy_state"]
            history_text = "query: " + (queries[step - 1] if step else str(row["question"]))
            missing_information = str(state_audit.get("missing_information", "")).strip()
            if missing_information:
                history_text += "\nmissing relation: " + missing_information
            examples.append(
                StateExample(
                    question_id=str(row["id"]),
                    dataset=str(row["dataset"]),
                    hop_count=int(row.get("hop_count", 0)),
                    step=step,
                    text=_state_text(row["question"], queries[:step], evidence, contents, evidence_characters),
                    question_text="query: " + str(row["question"]),
                    evidence_texts=evidence_texts,
                    new_evidence_texts=new_evidence_texts,
                    removed_evidence_texts=removed_evidence_texts,
                    history_text=history_text,
                    numeric=numeric,
                    stop_utility=stop_utilities[step],
                    continue_utility=continue_utilities[step],
                )
            )
    return examples


def _decision_utilities(
    row: dict,
    continue_target: str,
    f1_weight: float,
    em_weight: float,
    search_cost: float,
) -> tuple[list[float], list[float]]:
    """Offline targets only; no gold-derived value is an inference feature."""
    answer_values = [
        f1_weight * float(row["actions"][action]["f1"])
        + em_weight * float(row["actions"][action]["em"])
        for action in ACTIONS
    ]
    stop = answer_values[:MAX_SEARCH_STEPS]
    if continue_target == "bellman":
        value = answer_values.copy()
        continued = [0.0] * MAX_SEARCH_STEPS
        for step in range(MAX_SEARCH_STEPS - 1, -1, -1):
            continued[step] = -search_cost + value[step + 1]
            value[step] = max(answer_values[step], continued[step])
        return stop, continued
    if continue_target == "next_step":
        return stop, [answer_values[step + 1] - search_cost for step in range(MAX_SEARCH_STEPS)]
    return stop, [
        max(
            answer_values[future] - search_cost * (future - step)
            for future in range(step + 1, len(ACTIONS))
        )
        for step in range(MAX_SEARCH_STEPS)
    ]


def _value_numeric(
    state: dict,
    prior_queries: list[str],
    evidence: list[dict],
    prior: list[dict],
    graph_index: GraphStateIndex | None = None,
) -> tuple[float, ...]:
    base = _numeric_state(state, prior_queries)
    scores = np.asarray([float(item.get("score", 0.0)) for item in evidence], dtype="float64")
    if len(scores):
        ordered = np.sort(scores)[::-1]
        maximum, mean, minimum, std = float(scores.max()), float(scores.mean()), float(scores.min()), float(scores.std())
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else maximum
    else:
        maximum = mean = minimum = std = margin = 0.0
    previous_ids = {str(item.get("content_id", "")) for item in prior}
    current_ids = {str(item.get("content_id", "")) for item in evidence}
    changed = len(current_ids.symmetric_difference(previous_ids)) / max(1, len(current_ids | previous_ids))
    new_ids = current_ids - previous_ids
    removed_ids = previous_ids - current_ids
    frontier, internal, new_frontier = _graph_features(current_ids, new_ids, graph_index)
    sufficient = state.get("evidence_sufficient")
    gap = str(state.get("missing_information", ""))
    return (
        *base,
        maximum,
        mean,
        minimum,
        std,
        margin,
        float(changed),
        len(new_ids) / max(1, len(current_ids)),
        len(removed_ids) / max(1, len(previous_ids)),
        frontier,
        internal,
        new_frontier,
        float(sufficient is True),
        float(sufficient is None),
        min(1.0, len(gap) / 320.0),
    )


def _graph_features(
    current_ids: set[str],
    new_ids: set[str],
    graph_index: GraphStateIndex | None,
) -> tuple[float, float, float]:
    if graph_index is None or not current_ids:
        return 0.0, 0.0, 0.0
    selected_positions = {
        position
        for source in current_ids
        for position in graph_index.positions_by_source.get(source, ())
    }
    new_positions = {
        position
        for source in new_ids
        for position in graph_index.positions_by_source.get(source, ())
    }
    frontier_sources: set[str] = set()
    new_frontier_sources: set[str] = set()
    internal_edges = 0
    total_edges = 0
    for position in selected_positions:
        neighbours = graph_index.graph.get(position, ())
        total_edges += len(neighbours)
        for neighbour in neighbours:
            if neighbour in selected_positions:
                internal_edges += 1
            elif 0 <= neighbour < len(graph_index.source_by_position):
                frontier_sources.add(graph_index.source_by_position[neighbour])
    for position in new_positions:
        for neighbour in graph_index.graph.get(position, ()):
            if neighbour not in selected_positions and 0 <= neighbour < len(graph_index.source_by_position):
                new_frontier_sources.add(graph_index.source_by_position[neighbour])
    return (
        min(1.0, len(frontier_sources) / 200.0),
        internal_edges / max(1, total_edges),
        min(1.0, len(new_frontier_sources) / 100.0),
    )


def _document_text(item: dict, contents: dict[str, dict], character_budget: int) -> str:
    content_id = str(item.get("content_id", ""))
    content = contents.get(content_id, {})
    title = str(content.get("title") or item.get("title", ""))
    body = str(content.get("text", ""))[:character_budget]
    return f"passage: {title}\n{body}"


def _state_text(
    question: str,
    prior_queries: list[str],
    evidence: list[dict],
    contents: dict[str, dict],
    character_budget: int,
) -> str:
    sections = [f"Question: {question}"]
    if prior_queries:
        sections.append("Previous searches: " + " | ".join(prior_queries))
    remaining = character_budget
    evidence_sections = []
    for item in evidence:
        content_id = str(item.get("content_id", ""))
        content = contents.get(content_id, {})
        title = str(content.get("title") or item.get("title", ""))
        prefix = f"{title}: "
        text = str(content.get("text", ""))
        available = max(0, remaining - len(prefix))
        if available <= 0:
            break
        excerpt = text[:available]
        evidence_sections.append(prefix + excerpt)
        remaining -= len(prefix) + len(excerpt)
    sections.append("Current evidence:\n" + "\n".join(evidence_sections))
    return "\n".join(sections)


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda를 요청했지만 PyTorch에서 CUDA를 사용할 수 없습니다.")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _embedding_signature(states: list[StateExample], args) -> str:
    digest = hashlib.sha256()
    digest.update(str(EMBEDDING_FORMAT_VERSION).encode())
    digest.update(args.embedding_model.encode())
    for state in states:
        digest.update(state.question_id.encode())
        digest.update(str(state.step).encode())
        digest.update(state.question_text.encode())
        digest.update(state.history_text.encode())
        for group in (
            state.evidence_texts,
            state.new_evidence_texts,
            state.removed_evidence_texts,
        ):
            for value in group:
                digest.update(value.encode())
    return digest.hexdigest()


def _load_or_encode_embeddings(states: list[StateExample], args, device: torch.device) -> np.ndarray:
    path = Path(args.embedding_cache)
    signature = _embedding_signature(states, args)
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if str(cached["signature"].item()) == signature:
            values = np.asarray(cached["embeddings"], dtype="float32")
            if len(values) == len(states):
                _log(f"상태 임베딩 캐시 재사용: {path} · shape={values.shape}")
                return values
    unique_texts = sorted(
        {
            text
            for state in states
            for text in (
                state.question_text,
                state.history_text,
                *state.evidence_texts,
                *state.new_evidence_texts,
                *state.removed_evidence_texts,
            )
            if text
        }
    )
    _log(
        f"구조화 임베딩 생성: model={args.embedding_model} · unique text {len(unique_texts):,}개 "
        f"· state {len(states):,}개 · device={device}"
    )
    encoder = SentenceTransformer(args.embedding_model, device=str(device))
    encoded = encoder.encode(
        unique_texts,
        batch_size=args.embedding_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    lookup = {text: encoded[index] for index, text in enumerate(unique_texts)}
    zero = np.zeros(encoded.shape[1], dtype="float32")

    def aggregate(texts: tuple[str, ...], *, maximum: bool = False) -> np.ndarray:
        if not texts:
            return zero
        matrix = np.stack([lookup[text] for text in texts])
        return matrix.max(axis=0) if maximum else matrix.mean(axis=0)

    state_vectors = []
    for state in states:
        question = lookup[state.question_text]
        evidence_mean = aggregate(state.evidence_texts)
        evidence_max = aggregate(state.evidence_texts, maximum=True)
        new_evidence = aggregate(state.new_evidence_texts)
        removed_evidence = aggregate(state.removed_evidence_texts)
        history = lookup[state.history_text]
        state_vectors.append(
            np.concatenate(
                [
                    question,
                    evidence_mean,
                    evidence_max,
                    new_evidence,
                    removed_evidence,
                    history,
                    question * evidence_mean,
                    np.abs(question - evidence_mean),
                    question * new_evidence,
                ]
            )
        )
    values = np.asarray(state_vectors, dtype="float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, signature=np.asarray(signature), embeddings=values)
    temporary.replace(path)
    _log(f"구조화 상태 임베딩 저장: {path} · shape={values.shape}")
    return values


def _state_indices(states: list[StateExample]) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        indices[state.question_id].append(index)
    if any(len(values) != MAX_SEARCH_STEPS for values in indices.values()):
        raise AssertionError("질문별 policy state 수가 일치하지 않습니다.")
    return dict(indices)


def _fit_model(
    states: list[StateExample],
    embeddings: np.ndarray,
    indices: list[int],
    args,
    device: torch.device,
    seed_offset: int,
    label: str,
) -> tuple[StopValueNetwork, NumericScaler]:
    _seed_everything(args.seed + seed_offset)
    numeric = np.asarray([states[index].numeric for index in indices], dtype="float32")
    mean = numeric.mean(axis=0)
    scale = numeric.std(axis=0)
    scale[scale < 1e-6] = 1.0
    inputs = np.concatenate([embeddings[indices], (numeric - mean) / scale], axis=1).astype("float32")
    targets = np.asarray(
        [[states[index].stop_utility, states[index].continue_utility] for index in indices], dtype="float32"
    )
    advantage = targets[:, 1] - targets[:, 0]
    group_counts = Counter(_balance_group(states[index], args) for index in indices)
    group_weights = np.asarray(
        [
            len(indices) / (len(group_counts) * group_counts[_balance_group(states[index], args)])
            for index in indices
        ],
        dtype="float32",
    )
    weights = group_weights * (1.0 + args.gain_weight * np.abs(advantage))
    weights /= weights.mean()
    dataset = TensorDataset(
        torch.from_numpy(inputs),
        torch.from_numpy(targets),
        torch.from_numpy(weights.astype("float32")),
    )
    generator = torch.Generator().manual_seed(args.seed + seed_offset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    model = StopValueNetwork(inputs.shape[1], args.hidden_size, args.dropout).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    log_every = max(1, args.epochs // 6)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch_inputs, batch_targets, batch_weights in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_weights = batch_weights.to(device)
            logits = model(batch_inputs)
            values = torch.sigmoid(logits[:, :2])
            value_loss = (((values - batch_targets) ** 2).mean(dim=1) * batch_weights).mean()
            benefit = (batch_targets[:, 1] > batch_targets[:, 0]).float()
            sign_loss = F.binary_cross_entropy_with_logits(
                logits[:, 2],
                benefit,
                weight=batch_weights,
                pos_weight=torch.as_tensor(args.continue_positive_weight, device=device),
            )
            loss = value_loss + args.classification_weight * sign_loss
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(batch_inputs)
            seen += len(batch_inputs)
        if epoch == 1 or epoch % log_every == 0 or epoch == args.epochs:
            _log(f"  {label} epoch {epoch:>3}/{args.epochs} | loss={total / max(1, seen):.5f}")
    return model, NumericScaler(mean=mean, scale=scale)


def _predict_advantages(
    model: StopValueNetwork,
    scaler: NumericScaler,
    states: list[StateExample],
    embeddings: np.ndarray,
    indices: list[int],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    numeric = np.asarray([states[index].numeric for index in indices], dtype="float32")
    numeric = numeric[:, : len(scaler.mean)]
    inputs = np.concatenate(
        [embeddings[indices], (numeric - scaler.mean) / scaler.scale], axis=1
    ).astype("float32")
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            logits = model(torch.from_numpy(inputs[start : start + batch_size]).to(device))
            # The independently trained benefit logit is the ranking score.
            # Q values regularise the shared representation but their
            # difference was poorly calibrated in v1, especially after step 1.
            output.append(logits[:, 2].cpu().numpy())
    return np.concatenate(output).astype("float64")


def _by_question(
    question_ids: set[str],
    states: list[StateExample],
    state_indices: dict[str, list[int]],
    advantages: np.ndarray,
    indices: list[int],
) -> dict[str, list[float]]:
    predicted = {index: float(value) for index, value in zip(indices, advantages, strict=True)}
    return {
        question_id: [predicted[index] for index in state_indices[question_id]]
        for question_id in question_ids
    }


def _simulate(
    rows: list[dict],
    advantages: dict[str, list[float]],
    threshold: float | tuple[float, ...] | list[float],
) -> list[dict]:
    thresholds = (
        (float(threshold),) * MAX_SEARCH_STEPS
        if isinstance(threshold, (float, int, np.floating))
        else tuple(float(value) for value in threshold)
    )
    if len(thresholds) != MAX_SEARCH_STEPS:
        raise ValueError("단계별 threshold 수가 search step 수와 일치하지 않습니다.")
    records = []
    for row in rows:
        values = advantages[str(row["id"])]
        stop_step = MAX_SEARCH_STEPS
        for step, advantage in enumerate(values):
            if advantage <= thresholds[step]:
                stop_step = step
                break
        action = ACTIONS[stop_step]
        result = row["actions"][action]
        records.append(
            {
                "id": str(row["id"]),
                "dataset": str(row["dataset"]),
                "hop_count": int(row.get("hop_count", 0)),
                "selected_action": action,
                "stop_step": stop_step,
                "predicted_advantages": values,
                "direct_em": float(row["direct_em"]),
                "direct_f1": float(row["direct_f1"]),
                "policy_em": float(result["em"]),
                "policy_f1": float(result["f1"]),
                "oracle_f1": float(row["oracle_f1"]),
            }
        )
    return records


def _calibrate(
    rows: list[dict],
    advantages: dict[str, list[float]],
    args,
    thresholds: np.ndarray,
) -> tuple[float, ...]:
    def score(candidate: tuple[float, ...]) -> tuple[float, float, float, tuple[float, ...]]:
        records = _simulate(rows, advantages, candidate)
        utility_objective = _balanced_objective(rows, records, args)
        average_steps = float(np.mean([record["stop_step"] for record in records]))
        return (
            utility_objective - args.search_cost * average_steps,
            utility_objective,
            -average_steps,
            candidate,
        )

    # Coordinate search is fast enough for 251 candidates per step and avoids
    # a 251^3 exhaustive grid.  Multiple starts reduce local-optimum risk.
    midpoint = float(thresholds[len(thresholds) // 2])
    starts = (
        (float(thresholds[0]),) * MAX_SEARCH_STEPS,
        (midpoint,) * MAX_SEARCH_STEPS,
        (float(thresholds[-1]),) * MAX_SEARCH_STEPS,
    )
    solutions = []
    for start in starts:
        current = start
        for _ in range(5):
            previous = current
            for step in range(MAX_SEARCH_STEPS):
                candidates = []
                for value in thresholds:
                    candidate = list(current)
                    candidate[step] = float(value)
                    candidates.append(score(tuple(candidate)))
                current = max(candidates)[-1]
            if current == previous:
                break
        solutions.append(score(current))
    return max(solutions)[-1]


def _balance_group(row, args):
    dataset = str(row.dataset if isinstance(row, StateExample) else row["dataset"])
    if getattr(args, "balance_groups", "dataset_hop") == "dataset_hop":
        hop = int(row.hop_count if isinstance(row, StateExample) else row.get("hop_count", 0))
        return dataset, hop
    return (dataset,)


def _balanced_objective(rows: list[dict], records: list[dict], args) -> float:
    """Macro utility across deployable calibration groups.

    The v7 default balances datasets only.  This matches the reported Macro
    metric and keeps gold hop entirely out of training and calibration.  The
    old dataset-hop grouping remains available as an explicit ablation.
    """

    groups = sorted({_balance_group(row, args) for row in rows})
    records_by_id = {str(record["id"]): record for record in records}
    policy = np.asarray(
        [
            np.mean(
                [
                    args.f1_weight * records_by_id[str(row["id"])]["policy_f1"]
                    + args.em_weight * records_by_id[str(row["id"])]["policy_em"]
                    for row in rows
                    if _balance_group(row, args) == group
                ]
            )
            for group in groups
        ],
        dtype="float64",
    )
    direct = np.asarray(
        [
            np.mean(
                [
                    args.f1_weight * float(row["direct_f1"])
                    + args.em_weight * float(row["direct_em"])
                    for row in rows
                    if _balance_group(row, args) == group
                ]
            )
            for group in groups
        ],
        dtype="float64",
    )
    return float(np.mean(policy) - args.dataset_harm_penalty * np.maximum(0.0, direct - policy).sum())


def _hop_balanced_objective(rows: list[dict], records: list[dict], args) -> float:
    """Backward-compatible v6 objective used by the ablation and its tests."""

    if not hasattr(args, "balance_groups"):
        args = argparse.Namespace(**vars(args), balance_groups="dataset_hop")
    return _balanced_objective(rows, records, args)


def _indices_for(rows: list[dict], state_indices: dict[str, list[int]]) -> list[int]:
    return [index for row in rows for index in state_indices[str(row["id"])]]


def _nested_oof(rows, states, embeddings, state_indices, args, thresholds, device):
    records = []
    fold_gains = []
    for fold in range(args.folds):
        test = [row for row in rows if _bucket(str(row["id"]), "stop-value-outer", args.folds) == fold]
        pool = [row for row in rows if _bucket(str(row["id"]), "stop-value-outer", args.folds) != fold]
        calibration = [
            row for row in pool
            if _fraction(str(row["id"]), f"stop-value-calibration:{fold}") < args.calibration_ratio
        ]
        calibration_ids = {str(row["id"]) for row in calibration}
        fit = [row for row in pool if str(row["id"]) not in calibration_ids]
        model, scaler = _fit_model(
            states,
            embeddings,
            _indices_for(fit, state_indices),
            args,
            device,
            fold * 100,
            f"fold {fold + 1}/{args.folds}",
        )
        cal_indices = _indices_for(calibration, state_indices)
        cal_adv = _predict_advantages(model, scaler, states, embeddings, cal_indices, device, args.batch_size)
        threshold = _calibrate(
            calibration,
            _by_question({str(row["id"]) for row in calibration}, states, state_indices, cal_adv, cal_indices),
            args,
            thresholds,
        )
        test_indices = _indices_for(test, state_indices)
        test_adv = _predict_advantages(model, scaler, states, embeddings, test_indices, device, args.batch_size)
        fold_records = _simulate(
            test,
            _by_question({str(row["id"]) for row in test}, states, state_indices, test_adv, test_indices),
            threshold,
        )
        for record in fold_records:
            record["fold"] = fold
            record["threshold"] = threshold
        records.extend(fold_records)
        direct = _macro_value(fold_records, "direct_f1", args.datasets)
        policy = _macro_value(fold_records, "policy_f1", args.datasets)
        steps = float(np.mean([record["stop_step"] for record in fold_records]))
        fold_gains.append(policy - direct)
        _log(
            f"fold {fold + 1}/{args.folds} 평가: fit/cal/test={len(fit)}/{len(calibration)}/{len(test)} "
            f"| Macro ΔF1={policy - direct:+.4f} | avg extra searches={steps:.2f} "
            f"| thresholds={_format_thresholds(threshold)}"
        )
    if len(records) != len(rows) or len({record["id"] for record in records}) != len(rows):
        raise AssertionError("Nested OOF가 모든 질문을 정확히 한 번 포함해야 합니다.")
    return records, fold_gains


def _fit_final(rows, states, embeddings, state_indices, args, thresholds, device):
    probabilities: dict[str, list[float]] = {}
    for fold in range(args.folds):
        test = [row for row in rows if _bucket(str(row["id"]), "stop-value-final", args.folds) == fold]
        fit = [row for row in rows if _bucket(str(row["id"]), "stop-value-final", args.folds) != fold]
        model, scaler = _fit_model(
            states,
            embeddings,
            _indices_for(fit, state_indices),
            args,
            device,
            10_000 + fold * 100,
            f"final threshold {fold + 1}/{args.folds}",
        )
        test_indices = _indices_for(test, state_indices)
        values = _predict_advantages(model, scaler, states, embeddings, test_indices, device, args.batch_size)
        probabilities.update(
            _by_question({str(row["id"]) for row in test}, states, state_indices, values, test_indices)
        )
    threshold = _calibrate(rows, probabilities, args, thresholds)
    model, scaler = _fit_model(
        states,
        embeddings,
        _indices_for(rows, state_indices),
        args,
        device,
        20_000,
        "final model",
    )
    return model, scaler, threshold


def _report_static(rows: list[dict], datasets: tuple[str, ...]) -> None:
    _log("[고정 STOP 시점 Macro F1]")
    for action in ACTIONS:
        value = float(
            np.mean(
                [
                    np.mean([float(row["actions"][action]["f1"]) for row in rows if row["dataset"] == dataset])
                    for dataset in datasets
                ]
            )
        )
        _log(f"  {action}: {value:.4f}")


def _report_records(records: list[dict], datasets: tuple[str, ...]) -> dict:
    _log("[Nested OOF cost-aware Bellman value 정책 v7]")
    summary = {}
    for name in (*datasets, "Macro"):
        selected = records if name == "Macro" else [row for row in records if row["dataset"] == name]
        if name == "Macro":
            direct_em = _macro_value(records, "direct_em", datasets)
            direct_f1 = _macro_value(records, "direct_f1", datasets)
            policy_em = _macro_value(records, "policy_em", datasets)
            policy_f1 = _macro_value(records, "policy_f1", datasets)
            oracle_f1 = _macro_value(records, "oracle_f1", datasets)
        else:
            direct_em = float(np.mean([row["direct_em"] for row in selected]))
            direct_f1 = float(np.mean([row["direct_f1"] for row in selected]))
            policy_em = float(np.mean([row["policy_em"] for row in selected]))
            policy_f1 = float(np.mean([row["policy_f1"] for row in selected]))
            oracle_f1 = float(np.mean([row["oracle_f1"] for row in selected]))
        average_steps = float(np.mean([row["stop_step"] for row in selected]))
        continue_rate = float(np.mean([row["stop_step"] > 0 for row in selected]))
        _log(
            f"{name}: n={len(selected):,} | Direct EM/F1={direct_em:.4f}/{direct_f1:.4f} "
            f"| Policy={policy_em:.4f}/{policy_f1:.4f} | gain={policy_em - direct_em:+.4f}/"
            f"{policy_f1 - direct_f1:+.4f} | Oracle F1={oracle_f1:.4f} "
            f"| CONTINUE={continue_rate:.1%} | avg extra searches={average_steps:.2f}"
        )
        summary[name] = {
            "n": len(selected),
            "direct_em": direct_em,
            "direct_f1": direct_f1,
            "policy_em": policy_em,
            "policy_f1": policy_f1,
            "em_gain": policy_em - direct_em,
            "f1_gain": policy_f1 - direct_f1,
            "oracle_f1": oracle_f1,
            "continue_rate": continue_rate,
            "average_extra_searches": average_steps,
        }
    _log(f"OOF STOP step={dict(sorted(Counter(row['stop_step'] for row in records).items()))}")
    _log("[OOF hop 진단]")
    for hop in sorted({int(row["hop_count"]) for row in records if int(row["hop_count"]) > 0}):
        selected = [row for row in records if int(row["hop_count"]) == hop]
        direct = float(np.mean([row["direct_f1"] for row in selected]))
        policy = float(np.mean([row["policy_f1"] for row in selected]))
        oracle = float(np.mean([row["oracle_f1"] for row in selected]))
        steps = float(np.mean([row["stop_step"] for row in selected]))
        _log(
            f"  hop {hop}: n={len(selected):,} | Direct/Policy/Oracle F1="
            f"{direct:.4f}/{policy:.4f}/{oracle:.4f} | gain={policy - direct:+.4f} "
            f"| avg extra searches={steps:.2f}"
        )
    return summary


def _report_ranking(records: list[dict], rows: list[dict], args) -> dict:
    by_id = {str(row["id"]): row for row in rows}
    audit = {}
    _log("[OOF CONTINUE ranking 진단]")
    for step in range(MAX_SEARCH_STEPS):
        labels = []
        for record in records:
            row = by_id[str(record["id"])]
            stop_values, continue_values = _decision_utilities(
                row,
                args.continue_target,
                args.f1_weight,
                args.em_weight,
                args.search_cost,
            )
            labels.append(int(continue_values[step] > stop_values[step]))
        labels = np.asarray(labels, dtype="int64")
        scores = np.asarray(
            [float(record["predicted_advantages"][step]) for record in records],
            dtype="float64",
        )
        auc = float(roc_auc_score(labels, scores))
        average_precision = float(average_precision_score(labels, scores))
        _log(
            f"  step {step}: positive={labels.mean():.1%} | ROC-AUC={auc:.3f} "
            f"| Average Precision={average_precision:.3f}"
        )
        audit[str(step)] = {
            "positive_rate": float(labels.mean()),
            "roc_auc": auc,
            "average_precision": average_precision,
        }
    return audit


def _save(model, scaler, threshold, rows, states, records, audit, fold_gains, args) -> None:
    model_path = Path(args.model_out)
    predictions_path = (
        Path(args.predictions_out)
        if args.predictions_out
        else model_path.with_suffix(".nested_predictions.jsonl")
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "policy_type": "cost_aware_bellman_stop_value_v7",
            "embedding_format_version": EMBEDDING_FORMAT_VERSION,
            "embedding_model": args.embedding_model,
            "structured_embedding_size": int(model.layers[0].in_features - len(NUMERIC_FEATURE_NAMES)),
            "numeric_feature_names": NUMERIC_FEATURE_NAMES,
            "numeric_mean": scaler.mean,
            "numeric_scale": scaler.scale,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "threshold": threshold,
            "continue_target": args.continue_target,
            "f1_weight": args.f1_weight,
            "em_weight": args.em_weight,
            "search_cost": args.search_cost,
            "continue_positive_weight": args.continue_positive_weight,
            "max_search_steps": MAX_SEARCH_STEPS,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        temporary,
    )
    temporary.replace(model_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_tmp = predictions_path.with_suffix(predictions_path.suffix + ".tmp")
    with predictions_tmp.open("w", encoding="utf-8") as destination:
        for record in sorted(records, key=lambda row: (row["dataset"], row["id"])):
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    predictions_tmp.replace(predictions_path)
    manifest = {
        "format_version": FORMAT_VERSION,
        "policy_type": "cost_aware_bellman_stop_value_v7",
        "embedding_format_version": EMBEDDING_FORMAT_VERSION,
        "model": str(model_path),
        "nested_predictions": str(predictions_path),
        "datasets": list(args.datasets),
        "split": args.split,
        "rows": len(rows),
        "states": len(states),
        "target": "Q_stop_Q_continue_and_continue_benefit",
        "continue_target": args.continue_target,
        "utility": {
            "f1_weight": args.f1_weight,
            "em_weight": args.em_weight,
            "search_cost": args.search_cost,
        },
        "input_uses_gold": False,
        "gold_hop_usage": (
            "audit only; never an input, training weight, or calibration group"
            if args.balance_groups == "dataset"
            else "v6 ablation: training weight and calibration only; never an input feature"
        ),
        "threshold": threshold,
        "search_cost": args.search_cost,
        "nested_audit": audit,
        "fold_macro_gains": fold_gains,
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"datasets"}
        },
        "row_fingerprint": hashlib.sha256(
            "\n".join(sorted(str(row["id"]) for row in rows)).encode()
        ).hexdigest(),
    }
    manifest["config"]["datasets"] = list(args.datasets)
    model_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(f"정책 저장: {model_path} · thresholds={_format_thresholds(threshold)}")
    _log(f"Nested OOF 예측 저장: {predictions_path}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def _format_thresholds(values: tuple[float, ...] | list[float]) -> str:
    return "/".join(f"{float(value):+.3f}" for value in values)


def _normalise_query(text: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", " ", text.casefold()).strip()


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
