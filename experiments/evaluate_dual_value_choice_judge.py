"""Arbitrate baseline and adaptive-thinking answers using evidence only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import requests

from experiments.evaluate_context_answer_oracle import _iter_processed_examples
from experiments.evaluate_dual_value_reasoning_pilot import _context
from experiments.train_multihop_policy import SUPPORTED_DATASETS, _duration, _iter_jsonl
from src.qa_evaluation import ollama_model_identity
from src.settings import settings


FORMAT_VERSION = 1
CHOICE_SCHEMA = {
    "type": "object",
    "properties": {"choice": {"type": "string", "enum": ["A", "B"]}},
    "required": ["choice"],
    "additionalProperties": False,
}
_CHOICE = re.compile(r"\b([AB])\b", flags=re.IGNORECASE)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence-grounded answer arbitration pilot")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument(
        "--pilot-root",
        default="data/benchmarks/multihop/context_reader_aware/dual_value_reasoning_pilot",
    )
    parser.add_argument("--processed-root", default="data/benchmarks/processed")
    parser.add_argument(
        "--prefix-root",
        default="data/benchmarks/multihop/context_reader_aware/full_validation",
    )
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=list(SUPPORTED_DATASETS))
    parser.add_argument(
        "--output-root",
        default="data/benchmarks/multihop/context_reader_aware/dual_value_choice_judge",
    )
    parser.add_argument(
        "--cache-path",
        default="data/benchmarks/multihop/dual_value_choice_judge_cache.sqlite3",
    )
    parser.add_argument("--context-character-budget", type=int, default=6000)
    parser.add_argument("--ollama-host", default=settings.ollama_host)
    parser.add_argument("--model", default=settings.ollama_model)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--ollama-keep-alive", default="0")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--skip-identical", action="store_true",
        help="두 후보 답변 문자열이 같으면 결과가 같으므로 judge 호출을 생략합니다.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.datasets = tuple(dict.fromkeys(args.datasets))
    if min(
        args.context_character_budget,
        args.max_tokens,
        args.timeout,
        args.log_every,
    ) < 1:
        parser.error("context/token/timeout/log는 1 이상이어야 합니다.")
    return args


def _prompt(question: str, context: str, answer_a: str, answer_b: str) -> str:
    return f"""Choose which candidate answer is better supported by the evidence.
Check every multi-hop link and ensure the answer directly matches what the question asks for.
For an either/or comparison, the answer must be the correct named option, not its director, date,
or another intermediate entity. Preserve an explicit full date or name when shortening it would
change the answer. Do not prefer a candidate merely because it is longer or was produced later.
Use only the evidence. You must choose A or B and return exactly {{"choice":"A"}} or
{{"choice":"B"}}.

Question: {question}

Evidence:
{context}

Candidate A: {answer_a}
Candidate B: {answer_b}

Choice:"""


class ChoiceCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS choice_cache (
                cache_key TEXT PRIMARY KEY,
                raw_response TEXT NOT NULL,
                choice TEXT NOT NULL,
                latency_seconds REAL NOT NULL,
                prompt_tokens INTEGER,
                generated_tokens INTEGER
            )
            """
        )
        self.connection.commit()

    def get(self, key: str):
        row = self.connection.execute(
            "SELECT raw_response, choice, latency_seconds, prompt_tokens, generated_tokens "
            "FROM choice_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        return row

    def put(self, key: str, values: tuple) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO choice_cache VALUES (?, ?, ?, ?, ?, ?)",
            (key, *values),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def _parse_choice(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict) and str(payload.get("choice", "")).upper() in {"A", "B"}:
        return str(payload["choice"]).upper()
    match = _CHOICE.search(raw)
    if match:
        return match.group(1).upper()
    raise ValueError(f"choice 응답을 해석할 수 없습니다: {raw[:200]}")


def _judge(session, cache, config: dict, prompt: str):
    key = hashlib.sha256(
        json.dumps(
            {"format_version": FORMAT_VERSION, "config": config, "prompt": prompt},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cached = cache.get(key)
    if cached is not None:
        return str(cached[1]), True, float(cached[2]), cached[3], cached[4], str(cached[0])
    started = perf_counter()
    response = session.post(
        f"{config['host'].rstrip('/')}/api/generate",
        json={
            "model": config["model"],
            "prompt": prompt,
            "format": CHOICE_SCHEMA,
            "stream": False,
            "think": False,
            "keep_alive": config["keep_alive"],
            "options": {
                "temperature": 0.0,
                "num_predict": config["max_tokens"],
                "seed": config["seed"],
            },
        },
        timeout=config["timeout_seconds"],
    )
    response.raise_for_status()
    payload = response.json()
    raw = str(payload.get("response", "")).strip()
    choice = _parse_choice(raw)
    latency = perf_counter() - started
    values = (
        raw,
        choice,
        latency,
        payload.get("prompt_eval_count"),
        payload.get("eval_count"),
    )
    cache.put(key, values)
    return choice, False, latency, values[3], values[4], raw


def _summary(rows: list[dict]) -> dict:
    result = {}
    for name in ("baseline", "repair", "judged", "oracle"):
        result[f"{name}_em"] = float(np.mean([row[f"{name}_em"] for row in rows]))
        result[f"{name}_f1"] = float(np.mean([row[f"{name}_f1"] for row in rows]))
    result.update(
        {
            "count": len(rows),
            "em_change_points_vs_baseline": (result["judged_em"] - result["baseline_em"]) * 100.0,
            "f1_change_points_vs_baseline": (result["judged_f1"] - result["baseline_f1"]) * 100.0,
            "selected_a": sum(row["choice"] == "A" for row in rows),
            "selected_b": sum(row["choice"] == "B" for row in rows),
            "average_judge_latency_seconds": float(np.mean([row["judge_latency_seconds"] for row in rows])),
            "wrong_to_exact": sum(row["baseline_em"] < 1.0 and row["judged_em"] == 1.0 for row in rows),
            "exact_to_wrong": sum(row["baseline_em"] == 1.0 and row["judged_em"] < 1.0 for row in rows),
        }
    )
    return result


def _run_dataset(args, dataset: str, session, cache, config: dict) -> dict:
    pilot_rows = list(_iter_jsonl(Path(args.pilot_root) / f"{dataset}_{args.split}.jsonl"))
    prefix_by_id = {
        str(row["id"]): row
        for row in _iter_jsonl(Path(args.prefix_root) / f"{dataset}_{args.split}.jsonl")
    }
    pilot_ids = {str(row["id"]) for row in pilot_rows}
    examples_by_id = {
        example.policy_id: example
        for example in _iter_processed_examples(
            Path(args.processed_root) / dataset / f"{args.split}.jsonl"
        )
        if example.policy_id in pilot_ids
    }
    output = Path(args.output_root) / f"{dataset}_{args.split}.jsonl"
    rows_by_id = {}
    if output.exists() and not args.overwrite:
        rows_by_id = {str(row["id"]): row for row in _iter_jsonl(output)}
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite or not output.exists() else "a"
    if mode == "w":
        rows_by_id = {}
    started = perf_counter()
    new = 0
    with output.open(mode, encoding="utf-8") as destination:
        for pilot in pilot_rows:
            identifier = str(pilot["id"])
            if identifier in rows_by_id:
                continue
            example = examples_by_id[identifier]
            context, _ = _context(
                example,
                prefix_by_id[identifier],
                args.context_character_budget,
            )
            prompt = _prompt(
                str(pilot["question"]),
                context,
                str(pilot["baseline_prediction"]),
                str(pilot["repair_prediction"]),
            )
            identical = str(pilot["baseline_prediction"]) == str(pilot["repair_prediction"])
            if args.skip_identical and identical:
                if (float(pilot["baseline_em"]) != float(pilot["repair_em"]) or
                        float(pilot["baseline_f1"]) != float(pilot["repair_f1"])):
                    raise ValueError(f"동일 답변의 평가 점수가 다릅니다: {identifier}")
                choice, cache_hit, latency, prompt_tokens, generated_tokens, raw = (
                    "A", False, 0.0, 0, 0, '{"choice":"A"}',
                )
            else:
                choice, cache_hit, latency, prompt_tokens, generated_tokens, raw = _judge(
                    session, cache, config, prompt
                )
            prefix = "baseline" if choice == "A" else "repair"
            record = {
                **pilot,
                "choice": choice,
                "judged_prediction": pilot[f"{prefix}_prediction"],
                "judged_em": pilot[f"{prefix}_em"],
                "judged_f1": pilot[f"{prefix}_f1"],
                "oracle_em": max(float(pilot["baseline_em"]), float(pilot["repair_em"])),
                "oracle_f1": max(float(pilot["baseline_f1"]), float(pilot["repair_f1"])),
                "judge_cache_hit": cache_hit,
                "judge_skipped_identical": bool(args.skip_identical and identical),
                "judge_latency_seconds": latency,
                "judge_prompt_tokens": prompt_tokens,
                "judge_generated_tokens": generated_tokens,
                "judge_raw_response": raw,
            }
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            destination.flush()
            rows_by_id[identifier] = record
            new += 1
            completed = len(rows_by_id)
            if completed % args.log_every == 0 or completed == len(pilot_rows):
                current = list(rows_by_id.values())
                elapsed = perf_counter() - started
                eta = elapsed / max(1, new) * max(0, len(pilot_rows) - completed)
                summary = _summary(current)
                print(
                    f"{dataset} judge {completed}/{len(pilot_rows)} | ETA {_duration(eta)} | "
                    f"EM={summary['baseline_em']*100:.2f}->{summary['judged_em']*100:.2f} "
                    f"F1={summary['baseline_f1']*100:.2f}->{summary['judged_f1']*100:.2f}",
                    flush=True,
                )
    return _summary([rows_by_id[str(row["id"])] for row in pilot_rows])


def main() -> None:
    args = _arguments()
    session = requests.Session()
    model, digest = ollama_model_identity(args.ollama_host, args.model, session=session)
    config = {
        "host": args.ollama_host,
        "model": model,
        "model_digest": digest,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout,
        "seed": args.model_seed,
        "keep_alive": args.ollama_keep_alive,
        "thinking": False,
        "prompt_version": "dual_value_candidate_choice_v1",
    }
    cache = ChoiceCache(Path(args.cache_path))
    try:
        summaries = {
            dataset: _run_dataset(args, dataset, session, cache, config)
            for dataset in args.datasets
        }
    finally:
        cache.close()
        session.close()
    for dataset, result in summaries.items():
        print(
            f"{dataset}: baseline/judged/oracle F1={result['baseline_f1']*100:.2f}/"
            f"{result['judged_f1']*100:.2f}/{result['oracle_f1']*100:.2f} "
            f"delta={result['f1_change_points_vs_baseline']:+.2f}",
            flush=True,
        )
    summary_path = Path(args.output_root) / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "experiment": "dual_value_candidate_choice_judge",
                "judge": config,
                "datasets": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
