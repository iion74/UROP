# UROP 적응형 RAG 실험 보고서

> **공식 전체 validation·5개 학습 시드의 최종 결과는
> `FULL_VALIDATION_FINAL_REPORT.md`, 후속 저비용 정책 결과는
> `STUDENT_POLICY_IMPROVEMENT_REPORT.md`를 기준으로 한다.** 이 문서의 500문항 결과는 설계
> 과정의 파일럿이며 최종 성능으로 인용하지 않는다.

## 1. 현재 결론

이번 실험에서 Qwen3:8B 자체는 학습하지 않고 고정했다. 기존 STOP/CONTINUE 정책을 그대로 개선하는 대신, 실패 원인을 다음 두 문제로 분해했다.

1. 검색 후보에 필수 근거가 없거나 선택 과정에서 버려지는 문제
2. 필수 근거가 있어도 reader가 관계·분기 구조를 잘못 따라가는 문제

이를 바탕으로 `반사실 근거 복원 점수기 → 질문별 근거 폭 → 분기 위상 보존 → syndrome detector와 graph repair ranker` 구조를 구현했다. 이 구조는 추가 검색 3회 후의 후보에 적용했을 때 세 데이터셋 모두 Direct Hybrid RAG보다 높은 성능을 냈다. 그러나 이 결과는 고정 3회 검색을 사용한 성능 상한 실험이며, 자율 종료 성능으로 보고하면 안 된다.

별도로 proof-completeness 기반 자율 종료 정책을 학습했으나, 검색 횟수는 줄여도 2Wiki reader 성능이 고정 3회 proof 방식보다 유의하게 낮아 기각했다. 원인은 `gold support 포함`을 `reader가 답할 준비 완료`로 잘못 간주한 것이었다.

이 실패를 바탕으로 reader-aware counterfactual sufficiency 정책을 새로 구현했다. Qwen이 각 검색 prefix에서 `명시적 근거 사슬 + 짧은 답`을 생성하고, 소형 정책은 답 안정성·근거 지지·질문-답 유형·비교 분기 완성도를 보고 STOP을 결정한다. 이후 train-only 표본과 공식 validation 전체를 사용한 5개 학습 시드 반복 검증을 완료했다.

따라서 현재 단계의 정확한 결론은 다음과 같다.

- 근거 선택·구조 보존·희소 repair 모듈의 효과는 확인했다.
- “현재 근거에 gold 문서가 포함되어 있는가”만으로는 reader가 답할 준비가 됐는지 판단할 수 없다.
- 기존 support-completeness 종료 정책은 기각 상태로 유지한다.
- 공식 validation 전체·5개 시드 평균에서 FIXED3 대비 F1 변화는 HotpotQA +0.14, MuSiQue +0.03, 2Wiki +0.20점이었다.
- 평균 검색 절감률은 HotpotQA 5.45%, MuSiQue 2.79%, 2Wiki 5.46%였다.
- HotpotQA와 2Wiki의 F1 변화 95% 신뢰구간은 0보다 높았고, MuSiQue는 0을 포함해 동률로 해석한다.
- 매 prefix의 Qwen reader 호출 때문에 수정된 적응형 지연은 FIXED3보다 1.91~2.08초 길었다.
- 따라서 정확도 연구 후보로는 채택하지만 현재 구현을 MVP에 그대로 배포하지 않는다.

## 2. 무엇을 학습했는가

### 동결한 모델

- 답변·검색어 재작성: Qwen3:8B
- 임베딩: multilingual-e5-base
- 1차 재정렬: cross-encoder reranker

위 모델의 파라미터는 변경하지 않았다.

### 학습한 소형 정책

#### 2.1 Counterfactual proof-completion scorer

훈련 질문에서 필수 근거 문서 하나를 의도적으로 지워 불완전한 proof 상태를 만들었다. 현재 질문·검색 점수·문서 내용·제목 연결성만 보고 어떤 문서가 빠진 경로를 복원하는지 분류하도록 `HistGradientBoostingClassifier`를 학습했다.

- 입력 특징: 검색 점수, 질문-제목/본문 겹침, 문서 간 제목 언급, 현재 선택 문서와의 연결, 새 질문 정보량 등 25개
- 학습 데이터: HotpotQA·MuSiQue·2Wiki train 각 1,000문항
- 생성된 행동 행: 181,244개
- 추론에서 사용하지 않는 것: 데이터셋 ID, gold hop, gold answer, EM, F1

#### 2.2 Evidence-width policy

질문과 support score 분포를 보고 reader에 넣을 근거 수를 2~8개 사이에서 예측하는 `HistGradientBoostingRegressor`를 학습했다. 정답 label은 훈련 데이터에서 모든 필수 근거를 처음 포함하는 최소 문서 수다.

#### 2.3 Topology-aware evidence selection

질문에 두 개 이상의 명시적 anchor가 있으면 한쪽 분기에 문서 예산이 몰리지 않도록 `anchor → bridge → terminal` 슬롯을 보존했다. 데이터셋 이름이나 reasoning type으로 분기하지 않고 질문과 문서 제목 연결만 사용한다.

#### 2.4 Factorized graph repair

단일 repair 분류기는 “repair가 필요한가”와 “어느 문서를 넣을까”를 동시에 풀어 거짓 양성이 많았다. 이를 두 모델로 분리했다.

- syndrome detector: 현재 evidence state에 그래프로 복구 가능한 누락 간선이 있는지 예측
- repair ranker: 누락이 있다고 판단한 경우에만 그래프 이웃 중 복구 문서 1개를 선택

독립 train tuning 구간에서 평균 추가 문서를 0.30개에서 0.05개로 줄이면서 repair precision을 11.54%에서 48.28%로 높였다.

#### 2.5 Adaptive proof-completeness policy — 실험 후 기각

최초 검색과 추가 검색 1·2·3회 prefix마다 현재 proof가 완성됐는지 예측하는 `HistGradientBoostingClassifier`를 학습했다. 3회는 목표가 아니라 안전 상한으로만 사용했고, complete probability가 임계값을 넘으면 조기 종료했다.

하지만 gold support 포함 여부를 잘 예측해도 reader가 관계를 올바르게 따라간다는 보장이 없었다. 특히 2Wiki에서 조기 상태의 support coverage는 높았지만 답변 F1이 크게 떨어져 최종 채택하지 않았다.

#### 2.6 Reader-aware counterfactual sufficiency policy — 최종 연구 후보

Qwen3:8B는 계속 동결했다. 각 prefix에서 Qwen이 1~4개의 명시적 근거 절과 짧은 답을 생성하게 하고, 다음 observable state로 STOP 가능성을 예측했다.

- 소형 모델: `HistGradientBoostingClassifier` + `HistGradientBoostingRegressor`
- 분류 목표: 현재 답이 비용을 고려한 이후 최적 답보다 불리하지 않은가
- 회귀 목표: 지금 멈출 때 예상되는 이후 답변 utility 손실
- 입력: 기존 proof 특징, completeness 확률, unknown 여부, 이전 답과의 F1 일치도, 답의 문서/근거 사슬 포함 여부, 근거 절의 명시적 관계 포함 여부, 질문-답 유형 일치, 비교 질문의 양쪽 분기 관계 완성도
- 금지 입력: 데이터셋 ID, gold hop, gold answer, 실제 EM/F1
- 안전 조건: 한 번의 우연한 일치로 멈추지 않고, 두 번의 근거 변화 뒤에도 답과 근거 사슬이 안정된 시점부터 STOP 허용
- 개발 질문: 데이터셋별 600개(train 100 + validation 1~500 개발 구간)
- 최종 미사용 평가: 데이터셋별 validation 501~1,000번 500개

## 3. 데이터와 평가 절차

- 데이터셋: HotpotQA, MuSiQue, 2Wiki
- 기존 proof 모듈 훈련: 데이터셋별 train 1,000문항
- reader-aware 정책 개발: 데이터셋별 600문항
- reader-aware 설계·임계값 고정 구간: validation 1~500번
- reader-aware 최종 미사용 평가: validation 501~1,000번 500문항
- 생성 설정: Qwen3:8B, temperature 0, seed 42, thinking 비활성화
- 지표: EM, token F1, 평균 추가 검색 수, support recall/precision, all-support recall
- 통계: paired bootstrap 20,000회, 데이터셋별 95% 신뢰구간
- 세 데이터셋을 합친 점수는 보고하지 않는다.

## 4. 정확한 비교군

아래의 `추가 검색 3회`는 최초 Hybrid 검색 뒤 query rewrite 기반 검색을 세 번 실행한 방식이다. 새 proof 방식도 이 후보를 입력으로 받으므로 검색 비용은 동일하게 3회다.

| 데이터셋 | 방식 | EM | F1 | 평균 추가 검색 |
|---|---|---:|---:|---:|
| HotpotQA | Direct Hybrid RAG | 47.75 | 60.96 | 0.00 |
| HotpotQA | 추가 검색 1회 | 51.05 | 64.44 | 1.00 |
| HotpotQA | 추가 검색 2회 | 52.25 | 65.70 | 2.00 |
| HotpotQA | 추가 검색 3회 | 52.40 | 65.92 | 3.00 |
| HotpotQA | 추가 검색 3회 + proof selection/repair | 52.45 | 66.65 | 3.00 |
| MuSiQue | Direct Hybrid RAG | 17.20 | 23.81 | 0.00 |
| MuSiQue | 추가 검색 1회 | 22.00 | 29.21 | 1.00 |
| MuSiQue | 추가 검색 2회 | 22.75 | 30.40 | 2.00 |
| MuSiQue | 추가 검색 3회 | 24.55 | 32.57 | 3.00 |
| MuSiQue | 추가 검색 3회 + proof selection/repair | 25.55 | 33.20 | 3.00 |
| 2Wiki | Direct Hybrid RAG | 30.80 | 36.61 | 0.00 |
| 2Wiki | 추가 검색 1회 | 38.25 | 44.17 | 1.00 |
| 2Wiki | 추가 검색 2회 | 43.80 | 50.22 | 2.00 |
| 2Wiki | 추가 검색 3회 | 45.40 | 51.96 | 3.00 |
| 2Wiki | 추가 검색 3회 + proof selection/repair | 49.60 | 56.04 | 3.00 |

## 5. 고정 3회 proof 방식의 통계

### Direct Hybrid RAG 대비

| 데이터셋 | EM 변화 | F1 변화 | F1 변화 95% 신뢰구간 |
|---|---:|---:|---:|
| HotpotQA | +4.70 | +5.69 | +4.23 ~ +7.15 |
| MuSiQue | +8.35 | +9.39 | +7.73 ~ +11.01 |
| 2Wiki | +18.80 | +19.43 | +17.39 ~ +21.48 |

이 차이에는 추가 검색 3회의 효과가 포함된다. proof 모듈만의 순수 효과로 해석하면 안 된다.

### 동일한 추가 검색 3회 reader 대비 proof 모듈의 추가 효과

| 데이터셋 | EM 변화 | F1 변화 | F1 변화 95% 신뢰구간 |
|---|---:|---:|---:|
| HotpotQA | +0.05 | +0.73 | -0.48 ~ +1.98 |
| MuSiQue | +1.00 | +0.63 | -0.57 ~ +1.84 |
| 2Wiki | +4.20 | +4.08 | +2.43 ~ +5.70 |

처음 500문항을 제외한 1,500문항에서도 proof 모듈의 F1 변화는 HotpotQA +0.25(-1.16~+1.69), MuSiQue +0.38(-0.98~+1.75), 2Wiki +3.05(+1.22~+4.88)였다. 따라서 proof 모듈 자체의 확실한 일반화 개선은 현재 2Wiki에서만 확인됐다.

## 6. 자율 종료 실험 결과

### 6.1 Support-completeness 정책과 기각 이유

훈련 튜닝에서 completeness 임계값 0.970을 선택했다. 첫 500 validation 문항의 실제 reader 결과는 다음과 같다.

| 데이터셋 | Direct EM/F1 | 고정 3회 proof EM/F1 | 자율 proof EM/F1 | 자율 평균 추가 검색 |
|---|---:|---:|---:|---:|
| HotpotQA | 46.80 / 61.17 | 52.20 / 67.83 | 52.00 / 67.35 | 1.96 |
| MuSiQue | 18.80 / 24.66 | 29.60 / 36.20 | 29.40 / 36.19 | 2.88 |
| 2Wiki | 29.40 / 35.08 | 51.20 / 58.80 | 48.60 / 55.75 | 2.29 |

고정 3회 proof 대비 F1 변화는 HotpotQA -0.48, MuSiQue -0.01, 2Wiki -3.05였다. 2Wiki의 95% 신뢰구간은 -4.82~-1.44로 명확한 하락이다. 검색 비용은 줄었지만 성능 유지 조건을 충족하지 못했으므로 자율 proof 정책은 기각했다.

### 6.2 Reader-aware counterfactual sufficiency 500문항 파일럿 결과

아래 결과는 validation 501~1,000번 500문항에서 계산한 파일럿이다. 공식 전체 validation·5개 시드 최종 결과는 별도 최종 보고서를 사용한다. `근거 사슬 고정 3회`와 `reader-aware 자율 proof`는 같은 검색 후보, proof selection/repair, evidence-chain reader를 사용하며 종료 시점만 다르다.

| 데이터셋 | Direct EM/F1 | 짧은 답 고정 3회 proof EM/F1 | 근거 사슬 고정 3회 EM/F1 | reader-aware 자율 EM/F1 | 평균 추가 검색 | 검색 절감률 |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 52.20 / 62.83 | 58.40 / 69.66 | 61.00 / 72.24 | 61.40 / 72.74 | 2.70 | 9.93% |
| MuSiQue | 16.60 / 22.18 | 16.40 / 23.70 | 21.20 / 28.42 | 21.00 / 28.44 | 2.93 | 2.33% |
| 2Wiki | 32.80 / 37.98 | 53.40 / 58.21 | 58.60 / 64.30 | 59.60 / 65.16 | 2.75 | 8.40% |

근거 사슬 고정 3회 대비 reader-aware 자율 F1 변화와 paired bootstrap 95% 신뢰구간은 다음과 같다.

| 데이터셋 | F1 변화 | 95% 신뢰구간 | 1점 비열등 확률 |
|---|---:|---:|---:|
| HotpotQA | +0.50 | -0.10 ~ +1.30 | 100.00% |
| MuSiQue | +0.01 | -0.65 ~ +0.68 | 99.77% |
| 2Wiki | +0.87 | -0.00 ~ +1.85 | 100.00% |

따라서 새 정책은 고정 3회를 비교군에서 삭제해서 좋아 보이게 만든 결과가 아니다. 같은 강한 비교군을 유지한 상태에서 F1 점추정치를 낮추지 않고 검색 수를 줄였다. 다만 HotpotQA와 2Wiki의 우월성 신뢰구간 하한이 0 부근이고 MuSiQue는 사실상 동률이므로, 논문 주장은 `세 데이터셋에서 일관된 정확도 우월`보다 `데이터셋별 F1 비열등성과 검색 비용 절감`으로 제한하는 것이 정확하다.

## 7. 이번 실험에서 실제로 얻은 지식

1. STOP/CONTINUE를 최종 EM/F1 이득으로 직접 학습하면 거의 항상 최대 검색을 선택하는 정책으로 붕괴하기 쉽다.
2. 문서가 필수 근거 역할을 하는지를 반사실 삭제 상태에서 학습하면 고정 top-k보다 근거 재정렬이 좋아진다.
3. 긴 경로는 근거 수를 고정하면 잘리므로 질문별 evidence width가 필요하다.
4. 비교·병렬 질문은 pointwise score만으로 선택하면 한 분기가 사라지므로 proof topology를 보존해야 한다.
5. 그래프 후보를 전부 합치면 후보 수가 8개에서 약 40개로 늘어 기존 점수기의 분포가 붕괴한다. 전체 재랭킹보다 희소 local repair가 낫다.
6. repair 필요 여부와 repair 위치를 한 모델에 맡기면 거짓 양성이 많다. syndrome detector와 repair ranker 분리가 더 안정적이다.
7. 가장 중요한 새 결론은 `gold support가 모두 있음`과 `reader가 답할 준비가 됨`이 같지 않다는 점이다. 2Wiki 조기 종료 실패가 이를 보여준다.

## 8. Reader-aware 방향의 공식 전체 검증 결론

정책 target을 support completeness에서 `reader-aware counterfactual sufficiency`로 변경해 구현·검증했다.

훈련 시에만 각 prefix에서 다음을 계산한다.

1. 현재 근거로 낸 답변
2. 근거 하나를 제거하거나 순서를 바꾼 뒤 낸 답변
3. 다음 검색 후 낸 답변
4. 답변의 변화·근거 grounding·관계 경로 일관성

답이 evidence 변화에도 안정적이고 다음 검색의 예상 utility가 낮을 때 STOP label을 만든다. 추론에서는 gold answer나 실제 EM/F1을 측정하지 않고, 학습된 sufficiency model이 질문·근거·reader 상태 특징만으로 판단한다.

train-only pool과 다섯 무작위 학습 시드, 공식 validation 전체 검증을 완료했다. FIXED3 대비 F1은 HotpotQA +0.14, MuSiQue +0.03, 2Wiki +0.20점이었고 평균 검색 수는 모두 감소했다. 그러나 prefix마다 Qwen reader를 호출해 수정된 온라인 지연은 FIXED3보다 1.91~2.08초 늘었다. 따라서 정확도 연구 후보로는 검증됐지만 현재 구현을 MVP에 그대로 배포하지 않는다. 후속 reader-free Student는 reader 호출을 최종 1회로 줄여 FIXED3 대비 검색 7.44~10.09%, 지연 4.77~6.59%를 절감했지만 F1 우월성은 확인되지 않았다.

## 9. NCS 학습량 분석 도구에 적용할 때

- NCS 문서의 개념·선수지식·평가기준을 evidence node로 본다.
- 학습자 질문에 필요한 개념 경로를 proof selection이 구성한다.
- syndrome detector는 현재 근거에서 빠진 선수지식 연결을 감지한다.
- repair ranker는 보충할 NCS 단원·문단 후보를 제안한다.
- 학습량 분석은 선택된 경로 길이, 누락 prerequisite 수, 반복 repair 횟수로 설명할 수 있다.

다만 영어 멀티홉 QA에서 학습한 정책을 NCS에 그대로 배포해서는 안 된다. NCS 질문-근거-정답 또는 전문가 annotation으로 작은 개발·검증 집합을 먼저 만들어 transfer 성능을 확인해야 한다.

## 10. 기각한 실험

- 기존 Bellman STOP/CONTINUE 정책: validation에서 거의 항상 최대 검색
- one-stage executable proof program: Qwen3:8B가 형식적 proof를 안정적으로 생성하지 못해 HotpotQA·2Wiki 하락
- 질문 중심 문장 압축: 필요한 bridge 문장까지 제거
- 전체 graph/query memory 재랭킹: 후보 수 분포 이동으로 support precision 붕괴
- 단일 graph repair 분류기: repair precision HotpotQA 4.39%, MuSiQue 5.07%, 2Wiki 28.38%
- 질문 텍스트 기반 Direct/proof router: 미사용 구간 AUC 0.452/0.524/0.538로 무작위 수준
- support-completeness 자율 종료: 2Wiki fixed proof 대비 F1 유의 하락

## 11. 재현 파일

- 반사실 support scorer: `data/models/counterfactual_proof_completion_policy.joblib`
- evidence width model: `data/models/evidence_width_policy.joblib`
- 분리형 graph repair: `data/models/factorized_graph_proof_repair_policy.joblib`
- 자율 completeness 정책(기각): `data/models/adaptive_proof_completion_policy_high_threshold.joblib`
- 고정 3회 proof 결과: `data/benchmarks/multihop/causal_evidence_reader/factorized_graph_repair_validation_2000/`
- 고정 3회 proof 최종 통계: `data/benchmarks/multihop/causal_evidence_reader/factorized_graph_repair_validation_2000/final_analysis.json`
- 자율 종료 500문항 결과: `data/benchmarks/multihop/causal_evidence_reader/adaptive_proof_t097_validation_500/`
- reader-aware 최종 정책: `data/models/reader_aware_sufficiency_policy_final_candidate.joblib`
- reader-aware 학습 보고서: `data/models/reader_aware_sufficiency_policy_final_candidate.report.json`
- reader-aware 최종 예측·통계: `data/benchmarks/multihop/reader_aware_proof_prefixes/final_heldout_500_analysis.json`
- reader-aware prefix 결과: `data/benchmarks/multihop/reader_aware_proof_prefixes/chain_validation_1000/`
- 공식 전체 reader-aware prefix: `data/benchmarks/multihop/context_reader_aware/full_validation/`
- 공식 전체 5-seed 정책: `data/models/context_reader_aware_full_validation/`
- 공식 전체 최종 통계: `data/benchmarks/multihop/context_reader_aware/analysis/full_validation_multiseed.json`
- 공식 전체 최종 보고서: `FULL_VALIDATION_FINAL_REPORT.md`
- 저비용 Student 개선 보고서: `STUDENT_POLICY_IMPROVEMENT_REPORT.md`
- 주요 구현: `src/causal_evidence.py`, `src/proof_repair.py`, `src/reader_sufficiency.py`
- 주요 학습기: `experiments/train_counterfactual_proof_policy.py`, `experiments/train_evidence_width_policy.py`, `experiments/train_factorized_graph_repair_policy.py`, `experiments/train_adaptive_proof_completion_policy.py`, `experiments/train_reader_aware_sufficiency_policy.py`
- 최종 설정: `data/models/final_adaptive_rag_config.json`

## 12. 검증 상태

- 기존 proof 전체 validation: 데이터셋별 2,000문항
- 기존 proof 별도 분석 구간: 처음 500문항 제외 후 데이터셋별 1,500문항
- reader-aware 개발 구간: 데이터셋별 train 100문항 + validation 1~500번
- reader-aware 최종 미사용 구간: validation 501~1,000번, 데이터셋별 500문항
- reader-aware 공식 전체 검증: HotpotQA 7,405, MuSiQue 2,417, 2Wiki 12,576문항
- 정책 반복 학습: train-only 표본 시드 42~46, 데이터셋별 시드당 fit 1,000 + tune 200문항
- paired bootstrap: 20,000회
- 자동 테스트: 138개 통과
- Qwen/Ollama 평가 종료 후 모델 unload 확인

## 13. 이중 가치 적응형 RAG 최종 전체 평가

후속 실험에서는 하나의 STOP/CONTINUE 정책 대신 다음 검색의 근거 이득과 현재 답변의 오류
위험을 별도로 예측했다. 다섯 시드 Extra-Trees 앙상블이 `SEARCH/REASON/STOP`을 고르고,
REASON 문항만 Qwen3:8B thinking과 기존 답·새 답 중재를 수행한다. REASON 큐를 포함한
공식 validation 22,398문항 전체 평가 결과는 다음과 같다.

| 데이터셋 | 최종 EM | 최종 F1 | FIXED3 대비 F1 | F1 95% CI | 평균 검색 | 검색 절감 |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 58.03 | 72.18 | +0.41 | [+0.11, +0.71] | 2.65 | 11.75% |
| MuSiQue | 34.96 | 44.57 | +0.08 | [-0.14, +0.31] | 2.95 | 1.74% |
| 2Wiki | 57.55 | 64.61 | +0.59 | [+0.23, +0.95] | 2.61 | 13.15% |

따라서 세 데이터셋 모두 고정 3회보다 평균 F1이 높고 검색 횟수는 낮다. HotpotQA와
2Wiki의 F1 향상은 paired bootstrap 95% 구간도 0보다 높다. MuSiQue는 유의한 우월성은
아니지만 비열등성을 유지한다. 다만 지연과 생성 토큰은 증가하므로 NCS MVP 기본값은
corpus-wide 및 NCS-domain 검증 전까지 교체하지 않는다.

- 전체 시스템 분석기: `experiments/analyze_dual_value_full_system.py`
- 최종 결과: `data/benchmarks/multihop/context_reader_aware/analysis/dual_value_full_system.json`
- 전체 22,398문항 예측: `data/benchmarks/multihop/context_reader_aware/analysis/dual_value_full_system.predictions.jsonl`
- 자동 테스트: 141개 통과
- Qwen3:8B: GPU unload 확인
