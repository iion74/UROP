"""Export previously computed seed-wise confidence intervals for the paper.

Reads completed JSON reports only. Does not train policies, call a reader, or
recompute bootstrap intervals. CSV retains the source floating-point values.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data/benchmarks/multihop/context_reader_aware/analysis/sequential_multiseed"
OUTPUT = ROOT / "paper/evidence"
SEEDS = (42, 43, 44, 45, 46)
DATASETS = {"hotpotqa": ("HotpotQA", 7405), "musique": ("MuSiQue", 2417),
            "2wiki": ("2WikiMultiHopQA", 12576)}


def interval_text(row: dict, baseline: str) -> str:
    return (f"{row[f'delta_{baseline}_f1_points']:+.3f} "
            f"[{row[f'{baseline}_ci95_low']:+.3f}, "
            f"{row[f'{baseline}_ci95_high']:+.3f}]")


def main() -> None:
    rows = []
    sources = []
    cohorts = {dataset: set() for dataset in DATASETS}
    for seed in SEEDS:
        path = REPORTS / f"policy_seed{seed}.json"
        raw = path.read_bytes()
        report = json.loads(raw)
        assert report["training_sample_seed"] == seed
        assert report["training_questions"] == 6000
        assert report["training_states"] == 18000
        assert report["feature_count"] == 105
        sources.append({"path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(raw).hexdigest()})
        for dataset, (label, count) in DATASETS.items():
            cohort = report["training_cohort_by_dataset"][dataset]
            assert cohort["questions"] == 2000
            cohorts[dataset].add(cohort["selected_ids_sha256"])
            result = report["evaluations"]["full_validation"][dataset]
            assert result["questions"] == count
            row = {"training_sample_seed": seed, "dataset": label,
                   "validation_questions": count,
                   "policy_f1": result["policy"]["f1"],
                   "fixed4_f1": result["fixed"]["4"]["f1"],
                   "matched_random_expected_f1": result["matched_depth_random_expected_f1"],
                   "source_report": str(path.relative_to(ROOT))}
            for baseline, key in (("fixed4", "fixed4"),
                                  ("random", "matched_depth_random")):
                delta = result[f"f1_change_vs_{key}_points"]
                low, high = result[f"f1_change_vs_{key}_ci95"]
                assert low <= high
                score = row["fixed4_f1"] if baseline == "fixed4" else row["matched_random_expected_f1"]
                assert abs(row["policy_f1"] - score - delta) < 1e-9
                row.update({f"delta_{baseline}_f1_points": delta,
                            f"{baseline}_ci95_low": low,
                            f"{baseline}_ci95_high": high,
                            f"{baseline}_ci_lower_above_zero": low > 0})
            rows.append(row)
    assert all(len(digests) == len(SEEDS) for digests in cohorts.values())
    assert len(rows) == 15
    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "seed_confidence_intervals.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    text = ["# 시드별 F1 차이 및 95% 신뢰구간 증빙", "",
            "완료된 기존 평가 JSON에서 추출한 수치다. 새로운 학습·생성·부트스트랩 계산은 수행하지 않았다.", "",
            "## 표 A. 고정 4회 대비", "",
            "각 셀은 ΔF1 [95% 신뢰구간]이다. 단위는 100점 기준 F1 점수 차이며, 양수는 제안 정책이 높음을 뜻한다.", "",
            "| 학습 표본 시드 | HotpotQA | MuSiQue | 2WikiMultiHopQA |",
            "|---|---|---|---|"]
    for baseline in ("fixed4", "random"):
        if baseline == "random":
            text.extend(["", "## 표 B. 검색 깊이 분포 일치 무작위 종료 대비", "",
                         "각 셀은 ΔF1 [95% 신뢰구간]이다. 비교 대상은 무작위 배정의 기대 F1이다.", "",
                         "| 학습 표본 시드 | HotpotQA | MuSiQue | 2WikiMultiHopQA |",
                         "|---|---|---|---|"])
        for seed in SEEDS:
            selected = [row for row in rows if row["training_sample_seed"] == seed]
            text.append(f"| {seed} | " + " | ".join(interval_text(row, baseline) for row in selected) + " |")

    text.extend(["", "## 판정 및 단위", "",
                 "- 고정 4회 대비 하한이 0보다 높은 시드: HotpotQA 45·46, MuSiQue 42, 2WikiMultiHopQA 42·43·45·46.",
                 "- 고정 4회 대비 총 15개 중 7개 구간은 하한이 양수이며, 나머지 8개는 0을 포함한다.",
                 "- 무작위 종료 대비 15개 구간 모두 하한이 양수다.",
                 "- 표시값은 소수 셋째 자리 반올림이며, CSV에는 원본 정밀도를 유지했다.",
                 "- 이 구간은 시드별 평균 F1 차이의 문항 단위 신뢰구간이다. 5개 시드를 합친 평균의 신뢰구간이 아니다.",
                 "- 논문 표의 평균±표준편차는 학습 표본 시드 간 변동이며 이 신뢰구간과 다르다.",
                 "- 각 구간은 개별 비교의 95% 구간이다. 15개 비교 전체를 동시에 보장하는 보정 구간은 아니다.", "",
                 "## 계산 근거", "",
                 "1. 동일 평가 문항에서 제안 정책과 비교군의 F1 차이를 계산한다.",
                 "2. 문항 인덱스를 복원 추출하고, 추출된 문항들의 평균 F1 차이를 계산한다.",
                 "3. 반복 평균의 2.5·97.5 백분위수를 95% percentile bootstrap 구간으로 사용한다.",
                 "4. 실행 스크립트의 부트스트랩 반복 설정은 2,000회이며 평가 함수의 기본값도 2,000회다. 원본 결과 JSON에는 반복 횟수 자체가 별도 필드로 기록되어 있지 않다.",
                 "5. 고정 4회 비교의 부트스트랩 난수 시드는 42, 무작위 종료 비교는 43이다. 이는 학습 표본 시드 42~46과 역할이 다르다.",
                 "6. 무작위 종료의 기대 F1은 정책의 검색 깊이 비율로 단계별 F1을 가중합한다. 현재 신뢰구간은 학습된 정책과 이 비율을 고정한 문항 재표집으로 계산한다.", "",
                 "평가 문항 수: HotpotQA 7,405 / MuSiQue 2,417 / 2WikiMultiHopQA 12,576.", "",
                 "## 논문 평가 방법에 넣을 문장", "",
                 "> 비교군 대비 F1 차이의 불확실성을 평가하기 위해 각 학습 표본 시드에서 동일 문항의 F1 차이에 대해 대응 부트스트랩을 적용한다. 문항을 복원 추출하여 계산한 평균 차이 분포의 2.5 및 97.5 백분위수로 95% 신뢰구간을 산출한다. 이 구간은 각 시드의 문항 단위 성능 차이에 대한 것으로, 학습 표본 시드 간 표준편차와 구분한다.", "",
                 "반복 횟수 2,000회는 실행 설정으로 확인된다. 최종 제출 전 실제 실행 로그 또는 재계산 결과로 동일 설정을 확인하면 재현 근거가 더 완전해진다.", "",
                 "## 논문 결과에 넣을 문장", "",
                 "> 고정 4회 대비 F1 차이의 95% 신뢰구간 하한은 HotpotQA에서 시드 45·46, MuSiQue에서 시드 42, 2WikiMultiHopQA에서 시드 42·43·45·46에서 0보다 높았다. 나머지 8개 비교의 구간은 0을 포함하였다. 따라서 고정 4회 대비 결과는 모든 학습 표본에서의 통계적으로 유의한 우위보다, 5개 시드 평균에서 검색 횟수가 감소하면서 F1이 높아진 경향으로 해석한다. 한편 검색 깊이 분포 일치 무작위 종료 대비로는 15개 비교 모두 신뢰구간의 하한이 0보다 높아, 질문 상태에 따른 검색 배분 효과를 뒷받침하였다.", "",
                 "## 원본 자료", ""])
    for source in sources:
        path = ROOT / source["path"]
        text.append(f"- [{path.name}]({path}): SHA-256 `{source['sha256']}`")
    text.extend(["", "계산 코드:", "",
                 f"- [대응 부트스트랩 함수]({ROOT}/experiments/evaluate_answer_gain_search_policy.py:127)",
                 f"- [비교군별 F1 차이와 신뢰구간 저장]({ROOT}/experiments/evaluate_sequential_answer_gain_policy.py:286)",
                 f"- [5개 시드 순차 실행 및 반복 횟수 설정]({ROOT}/experiments/run_sequential_answer_gain_multiseed.py:95)", ""])
    (OUTPUT / "seed_confidence_intervals.md").write_text("\n".join(text), encoding="utf-8")
    (OUTPUT / "seed_confidence_intervals.sources.json").write_text(
        json.dumps({"export_only": True, "source_reports": sources,
                    "rows": len(rows), "score_unit": "F1 points on a 0-100 scale",
                    "bootstrap_replicates_configured": 2000,
                    "bootstrap_replicates_recorded_in_original_reports": False},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(rows)} dataset-seed rows to {OUTPUT.relative_to(ROOT)}")
    for dataset, (label, _) in DATASETS.items():
        selected = [row for row in rows if row["dataset"] == label]
        fixed = sum(row["fixed4_ci_lower_above_zero"] for row in selected)
        random = sum(row["random_ci_lower_above_zero"] for row in selected)
        print(f"{label}: lower bound > 0, fixed4 {fixed}/5; random {random}/5")


if __name__ == "__main__":
    main()
