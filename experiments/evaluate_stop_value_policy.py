"""Evaluate a frozen STOP/CONTINUE value policy on held-out trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.train_multihop_policy import SUPPORTED_DATASETS, _iter_jsonl
from experiments.train_stop_policy import ACTIONS
from experiments.train_stop_value_policy import (
    GraphStateIndex,
    NumericScaler,
    StopValueNetwork,
    _by_question,
    _device,
    _indices_for,
    _load_or_encode_embeddings,
    _predict_advantages,
    _simulate,
    _state_examples,
    _state_indices,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Held-out evaluation of a frozen STOP value policy")
    parser.add_argument("--answer-root", required=True)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--index-root", default="data/benchmarks/multihop/fixed_corpus/index")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--max-rows-per-dataset", type=int, default=0)
    parser.add_argument("--model", default="data/models/stop_value_policy_v3_step_thresholds_1000.pt")
    parser.add_argument("--embedding-cache", default="data/cache/stop_value_policy_validation_embeddings.npz")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--evidence-characters", type=int, default=1400)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", default="data/benchmarks/multihop/policy_eval/stop_value_validation_summary.json")
    parser.add_argument(
        "--predictions-out",
        help="표본 seed 반복 분석에 사용할 질문별 정책 선택·점수를 JSONL로 저장",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    return args


def _load_rows(args):
    rows, trajectories, contents, graph_indices = [], {}, {}, {}
    for dataset in args.datasets:
        answer = list(_iter_jsonl(Path(args.answer_root) / f"{dataset}_{args.split}.jsonl"))
        trajectory = list(_iter_jsonl(Path(args.trajectory_root) / f"{dataset}_{args.split}.jsonl"))
        if args.max_rows_per_dataset:
            answer = answer[: args.max_rows_per_dataset]
        selected = {str(row["id"]) for row in answer}
        trajectory_by_id = {str(row["id"]): row for row in trajectory if str(row["id"]) in selected}
        if len(trajectory_by_id) != len(answer):
            raise SystemExit(f"{dataset}: answer/trajectory ID mismatch")
        chunks = json.loads((Path(args.index_root) / args.split / dataset / "chunks.json").read_text())
        graph = json.loads((Path(args.index_root) / args.split / dataset / "chunk_graph.json").read_text())
        positions: dict[str, list[int]] = {}
        contents_by_source: dict[str, dict] = {}
        for position, chunk in enumerate(chunks):
            source = str(chunk["source"])
            positions.setdefault(source, []).append(position)
            entry = contents_by_source.setdefault(source, {"title": str(chunk.get("title", "")), "texts": []})
            entry["texts"].append(str(chunk.get("text", "")))
        contents.update({key: {"title": value["title"], "text": " ".join(value["texts"])} for key, value in contents_by_source.items()})
        graph_indices[dataset] = GraphStateIndex(
            graph={int(key): tuple(int(item) for item in value) for key, value in graph.items()},
            positions_by_source={key: tuple(value) for key, value in positions.items()},
            source_by_position=tuple(str(chunk["source"]) for chunk in chunks),
        )
        rows.extend(answer)
        trajectories.update(trajectory_by_id)
    return rows, trajectories, contents, graph_indices


def _mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def _paired_bootstrap_all(records, answer_by_id, datasets, samples, seed):
    """Vectorized dataset-stratified paired CIs against every fixed action."""
    metrics = ("em", "f1")
    columns = tuple((action, metric) for action in ACTIONS for metric in metrics)
    rng = np.random.default_rng(seed)
    draws = np.zeros((samples, len(columns)), dtype="float64")
    observed = np.zeros(len(columns), dtype="float64")
    for dataset in datasets:
        selected = [record for record in records if record["dataset"] == dataset]
        differences = np.asarray(
            [
                [
                    float(record[f"policy_{metric}"])
                    - float(answer_by_id[str(record["id"])]["actions"][action][metric])
                    for action, metric in columns
                ]
                for record in selected
            ],
            dtype="float64",
        )
        observed += differences.mean(axis=0) / len(datasets)
        for start in range(0, samples, 128):
            end = min(samples, start + 128)
            indices = rng.integers(0, len(differences), size=(end - start, len(differences)))
            draws[start:end] += differences[indices].mean(axis=1) / len(datasets)
    result = {}
    for column, (action, metric) in enumerate(columns):
        result.setdefault(f"policy_vs_{action}", {})[metric] = {
            "observed_gain": float(observed[column]),
            "ci95_low": float(np.quantile(draws[:, column], 0.025)),
            "ci95_high": float(np.quantile(draws[:, column], 0.975)),
            "probability_positive": float(np.mean(draws[:, column] > 0.0)),
        }
    return result


def main() -> None:
    args = _arguments()
    payload = torch.load(args.model, map_location="cpu", weights_only=False)
    args.embedding_model = str(payload["embedding_model"])
    args.hidden_size = int(payload["hidden_size"])
    args.dropout = float(payload["dropout"])
    rows, trajectories, contents, graph_indices = _load_rows(args)
    # v3 checkpoints predating this field used the best-future oracle target.
    # New checkpoints are trained on the deployable one-step decision target.
    continue_target = str(payload.get("continue_target", "best_future"))
    f1_weight = float(payload.get("f1_weight", 1.0))
    em_weight = float(payload.get("em_weight", 0.0))
    search_cost = float(payload.get("search_cost", 0.0))
    states = _state_examples(
        rows,
        trajectories,
        contents,
        graph_indices,
        args.evidence_characters,
        continue_target,
        f1_weight,
        em_weight,
        search_cost,
    )
    device = _device(args.device)
    embeddings = _load_or_encode_embeddings(states, args, device)
    indices = _state_indices(states)
    scaler = NumericScaler(mean=np.asarray(payload["numeric_mean"]), scale=np.asarray(payload["numeric_scale"]))
    model = StopValueNetwork(embeddings.shape[1] + len(scaler.mean), args.hidden_size, args.dropout).to(device)
    model.load_state_dict(payload["state_dict"])
    all_indices = _indices_for(rows, indices)
    predicted = _predict_advantages(model, scaler, states, embeddings, all_indices, device, args.embedding_batch_size)
    advantages = _by_question({str(row["id"]) for row in rows}, states, indices, predicted, all_indices)
    records = _simulate(rows, advantages, payload["threshold"])
    answer_by_id = {str(row["id"]): row for row in rows}
    result = {
        "model": args.model,
        "continue_target": continue_target,
        "utility": {
            "f1_weight": f1_weight,
            "em_weight": em_weight,
            "search_cost": search_cost,
        },
        "threshold": list(payload["threshold"]),
        "rows": len(rows),
        "metrics": {},
        "stop_steps": dict(sorted(Counter(record["stop_step"] for record in records).items())),
    }
    for dataset in (*args.datasets, "Macro"):
        selected = records if dataset == "Macro" else [row for row in records if row["dataset"] == dataset]
        direct_em, direct_f1 = _mean(selected, "direct_em"), _mean(selected, "direct_f1")
        policy_em, policy_f1 = _mean(selected, "policy_em"), _mean(selected, "policy_f1")
        static = {}
        for action in ACTIONS:
            action_rows = [row["actions"][action] for row in rows if dataset == "Macro" or row["dataset"] == dataset]
            static[action] = {"em": _mean(action_rows, "em"), "f1": _mean(action_rows, "f1")}
        result["metrics"][dataset] = {
            "direct": {"em": direct_em, "f1": direct_f1},
            "policy": {"em": policy_em, "f1": policy_f1},
            "gain": {"em": policy_em - direct_em, "f1": policy_f1 - direct_f1},
            "oracle_f1": _mean(selected, "oracle_f1"),
            "average_extra_searches": float(np.mean([row["stop_step"] for row in selected])),
            "static": static,
        }
    result["paired_bootstrap"] = _paired_bootstrap_all(
        records,
        answer_by_id,
        args.datasets,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    result["bootstrap_samples"] = args.bootstrap_samples
    result["bootstrap_seed"] = args.bootstrap_seed
    if args.predictions_out:
        predictions = Path(args.predictions_out)
        predictions.parent.mkdir(parents=True, exist_ok=True)
        temporary_predictions = predictions.with_suffix(predictions.suffix + ".tmp")
        with temporary_predictions.open("w", encoding="utf-8") as destination:
            for record in sorted(records, key=lambda row: (row["dataset"], row["id"])):
                destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary_predictions.replace(predictions)
        result["predictions"] = str(predictions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
