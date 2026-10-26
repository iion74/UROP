# NOVA UROP 저비용 Student 정책 개선 실험

## 1. 결론

중간 Qwen reader 호출을 제거하는 저비용 정책을 구현하고 HotpotQA·MuSiQue·2Wiki의
공식 validation 전체 22,398문항에서 평가했다. 모든 평가는 질문마다 제공되는 공식
distractor 문서 안에서 검색하는 **question-local closed-context** 조건이다.

- reader-free Student는 검색이 두 번 진행된 상태에서 세 번째 검색의 기대 이득을 예측한다.
- STOP이면 현재 근거로, CONTINUE이면 세 번째 검색 근거로 Qwen reader를 **최종 한 번만**
  호출한다.
- 5개 무작위 학습 표본 평균은 FIXED3보다 F1이 HotpotQA `-0.05점`, MuSiQue `-0.18점`,
  2Wiki `-0.06점`이었다. 세 신뢰구간은 모두 0을 포함해 유의한 차이는 아니었다.
- 평균 추가 검색은 `7.44~10.09%`, 추정 지연은 `4.77~6.59%` 감소했다.
- reader 생성 토큰은 FIXED3와 거의 같고, 기존 reader-aware 정책보다는 `73.16~74.33%`
  감소했다.
- 그러나 MuSiQue의 시드별 편차가 커서, 현재 Student를 정확도 개선 모델로 주장하거나
  특정 시드를 validation 결과만 보고 선택해 배포해서는 안 된다.

따라서 최종 판정은 **효율 후보로는 유망하지만 정확도 우월성은 미확인**이다. 기존
reader-aware 정책은 정확도 연구 후보로 유지하고, reader-free Student는 별도 미사용
holdout 또는 새 데이터셋에서 재확인하기 전까지 MVP 기본값으로 교체하지 않는다.

## 2. 구현한 방법

### Reader-free Student

- 의사결정 시점: 추가 검색 2회 이후 `STOP@2` 또는 `CONTINUE@3`
- 최종 reader 호출: 선택된 prefix에서 정확히 1회
- 입력 특징: 84개
  - 검색 점수·점수 간격·후보 수·새 후보 비율·기존 후보 생존율
  - 최초·1회·2회 검색의 시간적 변화
  - 질문과 검색 문서 제목의 token coverage·Jaccard·점수 분산
  - 질문 길이·WH 유형·비교/불리언/숫자 표현
- 금지 입력: 데이터셋 ID, gold hop, gold answer, gold support, 실제 EM/F1, reader 답변
- 학습 label: 훈련에서만 `0.5×EM + 0.5×F1`로 2회 종료와 3회 검색의 utility를 비교
- 검색 비용: 100점 기준 0.20점
- 후보 모델:
  - HistGradientBoosting 분류·회귀
  - 평균·절대손실과 조건부 80/90/95% 분위수 회귀
  - Extra-Trees 분류·회귀
- 희소 CONTINUE와 큰 손실 사례에 class balance 및 utility-gain 가중치 적용

### 검증 절차

- train prefix pool: 데이터셋별 train-only 무작위 2,000문항
- 시드 42~46별 선택: 데이터셋별 1,200문항, 총 3,600문항
- 임계값·모델 선택: train 안에서 5-fold out-of-fold 예측만 사용
- 공식 validation 전체: HotpotQA 7,405, MuSiQue 2,417, 2Wiki 12,576문항
- 통계: 데이터셋별 paired bootstrap 20,000회
- Qwen3:8B·embedding·reranker는 재학습하지 않음
- 기존 prefix cache를 재사용했으므로 이번 정책 학습·평가는 CPU만 사용

## 3. 5개 시드 평균 결과

모든 점수는 100점 기준이다.

| 데이터셋 | FIXED3 EM/F1 | Student EM/F1 | EM 변화 | F1 변화 | F1 변화 95% CI |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 57.73 / 71.78 | 57.65 / 71.73 | -0.08 | -0.05 | -0.22 ~ +0.12 |
| MuSiQue | 34.92 / 44.49 | 34.84 / 44.31 | -0.07 | -0.18 | -0.52 ~ +0.16 |
| 2Wiki | 57.16 / 64.01 | 57.09 / 63.95 | -0.07 | -0.06 | -0.23 ~ +0.10 |

| 데이터셋 | FIXED3 검색 | Student 검색 | 검색 절감 | FIXED3 지연 | Student 지연 | 지연 절감 |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 3.00 | 2.70 | 10.09% | 2.76초 | 2.58초 | 6.59% |
| MuSiQue | 3.00 | 2.78 | 7.44% | 3.10초 | 2.95초 | 4.77% |
| 2Wiki | 3.00 | 2.71 | 9.83% | 2.40초 | 2.26초 | 6.06% |

| 데이터셋 | FIXED3 reader 토큰 | Student reader 토큰 | 기존 reader-aware 토큰 | Student의 기존 정책 대비 절감 |
|---|---:|---:|---:|---:|
| HotpotQA | 51.02 | 50.86 | 189.45 | 73.16% |
| MuSiQue | 50.08 | 50.09 | 195.14 | 74.33% |
| 2Wiki | 60.92 | 60.96 | 228.04 | 73.27% |

Student는 FIXED3처럼 reader를 한 번만 호출하므로 생성 토큰은 FIXED3와 거의 같다. 비용
이득은 주로 세 번째 query rewrite·검색을 생략한 질문에서 발생한다. 표의 reader 토큰에는
기존 prefix가 기록하지 않은 query-rewriter 생성 토큰이 포함되지 않는다.

## 4. 기존 reader-aware 정책과 비교

지연 계산 오류를 수정한 기존 reader-aware 정책의 평균 지연은 HotpotQA 4.69초, MuSiQue
5.18초, 2Wiki 4.31초다.

| 데이터셋 | Student F1 변화 | 검색 수 변화 | 지연 감소 | reader 토큰 감소 |
|---|---:|---:|---:|---:|
| HotpotQA | -0.19 | 2.84 → 2.70 | 45.07% | 73.16% |
| MuSiQue | -0.21 | 2.92 → 2.78 | 42.96% | 74.33% |
| 2Wiki | -0.27 | 2.84 → 2.71 | 47.70% | 73.27% |

즉 reader-aware 정책의 작은 정확도 이득과 Student의 큰 생성비용 절감 사이에 명확한
trade-off가 있다. 현재 데이터만으로 둘 중 하나가 모든 지표를 지배한다고 말할 수 없다.

## 5. 앙상블과 시드 안정성

5개 Student의 다수결은 개별 시드 편차를 줄였지만 정확도 상승으로 이어지지는 않았다.

| 데이터셋 | 다수결 F1 변화 | 검색 절감 | 지연 절감 |
|---|---:|---:|---:|
| HotpotQA | -0.08 | 6.85% | 4.52% |
| MuSiQue | -0.14 | 4.11% | 2.54% |
| 2Wiki | -0.01 | 6.73% | 4.11% |

모든 모델이 STOP에 동의할 때만 멈추는 방식은 F1을 사실상 완전히 보존했지만 검색 절감이
`0.03~0.12%`에 불과해 실용적 이득이 없었다. 하나라도 STOP이면 멈추는 속도형은 검색을
`21.79~26.07%` 줄였지만 F1이 `0.07~0.26점` 낮아져 별도 speed mode 외에는 부적합하다.

seed 45 하나는 세 데이터셋에서 F1 변화가 `+0.01, +0.19, -0.01점`이고 검색을
`8.19~12.22%` 줄였지만, validation을 확인한 뒤 이 시드를 선택하면 사후 선택 편향이다.
따라서 최종 모델로 채택하지 않는다.

## 6. 추가 ablation

### 원문 근거 그래프 특징

질문-본문 coverage, 질문 엔티티 포함률, 문서 연결 성분, bridge entity, 제목 교차 언급을
추가해 총 120개 특징으로 만들었다. seed 42 단일 개발 실험에서 검색은
`10.84~16.28%`, 지연은 `6.96~10.49%` 줄었으나 F1 변화가 HotpotQA `+0.00`,
MuSiQue `-0.28`, 2Wiki `-0.11점`이었다. 단순 lexical graph 특징만으로 희소한 MuSiQue
추가검색 필요 사례를 안정적으로 구분하지 못해 다중 시드 단계로 승격하지 않았다.

### 현재 reader를 한 번 검사하는 선택적 정책

추가 검색 2회 시점의 답·근거 사슬을 한 번 생성해 grounding을 확인했다. F1 변화는
HotpotQA `+0.10`, MuSiQue `-0.26`, 2Wiki `+0.01점`이었지만, CONTINUE 질문에서 reader를
한 번 더 호출해 FIXED3 대비 reader 토큰이 `54.98~78.45%`, 지연이 `5.02~14.38%`
증가했다. 비용 목표를 충족하지 못해 기각했다.

### 분위수·Extra-Trees

희소 CONTINUE 사례의 평균이 아니라 상위 80/90/95% 예상 이득을 모델링하고 Extra-Trees도
비교했다. 일부 시드의 분류 AUC는 개선됐지만 시드 간 종료율과 MuSiQue 성능 편차를
해결하지 못했다. 핵심 병목은 모델 종류보다 추가 검색이 실제 답을 바꾸는 양성 궤적이
데이터셋별 약 3~6%로 희소하다는 점이다.

## 7. 지연 계산 수정

기존 분석은 prefix별 누적 query-rewrite 시간만 사용하면서 검색·후처리 잔여 시간을
누락했다. 수정 후에는 다음처럼 계산한다.

1. 기록된 누적 rewrite 시간은 그대로 사용한다.
2. `전체 planner 시간 - 전체 rewrite 시간`을 검색·후처리 잔여 시간으로 둔다.
3. 조기 종료 단계에 비례해 잔여 시간을 배분한다.
4. 3단계까지 실행하면 기록된 전체 planner 시간과 정확히 같아진다.

이 수정으로 기존 reader-aware 지연은 `4.38/4.61/4.06초`가 아니라
`4.69/5.18/4.31초`가 됐다. 정확도·검색 횟수·토큰 결과는 변하지 않는다.

## 8. 최종 결정과 다음 실험

- 정확도 연구 후보: 기존 reader-aware 정책 유지
- 서비스 기준선: 단순하고 안정적인 FIXED3 유지
- 효율 연구 후보: reader-free Student 유지, 아직 MVP 기본값으로 배포하지 않음
- 기각: single-reader 선택 정책
- 보류: lexical evidence-graph Student

다음 실험에서 실제로 필요한 것은 모델 종류를 더 바꾸는 것이 아니라, 희소한 CONTINUE
사례를 늘리는 것이다. 더 큰 train-only 궤적을 만들고 hard-case mining으로 검색 전후 답이
달라진 질문을 집중 수집한 뒤, 새로운 미사용 holdout에서 reader-free Student를 한 번만
최종 검증해야 한다. 또한 현재 평가는 question-local closed-context이므로 논문에서 일반
open-domain RAG 성능으로 표현해서는 안 되며, 별도의 corpus-wide 검색 평가가 필요하다.

## 9. 재현 파일

- 학습·평가: `experiments/train_reader_free_student_policy.py`
- 다중 시드 분석: `experiments/analyze_student_policy_multiseed.py`
- 5개 Student: `data/models/reader_free_student_multiseed/policy_seed{42..46}.joblib`
- 5개 학습 보고서: `data/models/reader_free_student_multiseed/policy_seed{42..46}.report.json`
- 전체 분석: `data/benchmarks/multihop/context_reader_aware/analysis/reader_free_student_multiseed.json`
- 근거 그래프 ablation: `data/models/reader_free_graph_student_2000/`
- single-reader ablation: `data/models/single_reader_student/`

## 10. 이중 가치 라우터와 적응형 추론 실험

추가 검색이 필요한 상태와 근거는 충분하지만 생성기가 틀릴 위험이 큰 상태를 하나의
STOP/CONTINUE 값으로 섞지 않고 분리했다.

- Evidence head: reader-free 검색 특징 84개로 세 번째 검색이 새 정답 근거를 추가할지 예측
- Answer-risk head: 두 번째 검색 시점의 답·근거 사슬 특징으로 현재 답이 틀릴 위험을 예측
- 행동: 근거 부족은 `SEARCH`, 답 위험은 `REASON`, 둘 다 낮으면 `STOP`
- 학습: train pool에서 데이터셋당 1,200개, 시드 42~46을 사용하고 validation gold는 학습에
  사용하지 않음
- 최종 라우팅: 유리한 시드 하나를 고르지 않고 5개 시드 다수결 사용

전체 validation의 재추론 전 라우팅 결과는 다음과 같다.

| 데이터셋 | F1 | FIXED3 대비 F1 | 평균 검색 | 검색 절감 | REASON 비율 |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 71.78 | -0.00 | 2.65 | 11.75% | 21.03% |
| MuSiQue | 44.40 | -0.09 | 2.95 | 1.74% | 4.34% |
| 2Wiki | 63.98 | -0.03 | 2.61 | 13.15% | 20.70% |

`REASON` 문항에는 같은 근거로 Qwen3:8B thinking 답을 하나 더 만들고, 기존 빠른 답과
thinking 답을 근거 기반 중재기가 다시 비교하도록 했다. 처음에는 서로 겹치지 않는
50개·50개·300개로 재현성을 확인했고, 이후 남은 REASON 문항까지 모두 생성했다. 따라서
아래 값은 표본 외삽이 아니라 validation 전체 22,398문항의 직접 평가 결과다.

| 데이터셋 | Direct F1 | FIXED3 F1 | 최종 F1 | FIXED3 대비 | F1 95% CI | 평균 검색 | 검색 절감 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 70.32 | 71.78 | 72.18 | +0.41 | [+0.11, +0.71] | 2.65 | 11.75% |
| MuSiQue | 39.30 | 44.49 | 44.57 | +0.08 | [-0.14, +0.31] | 2.95 | 1.74% |
| 2Wiki | 59.26 | 64.01 | 64.61 | +0.59 | [+0.23, +0.95] | 2.61 | 13.15% |

HotpotQA와 2Wiki는 검색 횟수를 줄이면서 F1을 유의하게 높였다. MuSiQue는 개선 폭이
작아 우월성은 입증하지 못했지만, F1 손실 없이 비열등성을 유지했다. 세 데이터셋 모두
평균 F1은 FIXED3보다 높고 평균 검색 횟수는 낮다.

추가 thinking 답을 무조건 채택하면 일부 표본에서 큰 퇴행이 발생했지만, 기존 답과 새 답
중 근거에 더 맞는 하나를 고르는 중재 단계가 이를 완화했다. 그 대가로 FIXED3 대비 추정
지연시간은 HotpotQA 4.60%, MuSiQue 7.06%, 2Wiki 3.93% 증가했고 reader 생성 토큰은
각각 22.17%, 6.01%, 24.07% 증가했다. 따라서 정확도·검색 효율 연구 결과로는 채택하되,
MVP 기본 정책 교체는 corpus-wide 및 NCS-domain 검증 이후로 보류한다.

재현 파일:

- 이중 라우터 학습: `experiments/train_dual_value_router.py`
- 5시드 다수결 분석: `experiments/analyze_dual_value_router_multiseed.py`
- 적응형 thinking: `experiments/evaluate_dual_value_reasoning_pilot.py`
- 후보 중재: `experiments/evaluate_dual_value_choice_judge.py`
- 반복 파일럿 분석: `experiments/analyze_dual_value_reasoning_replication.py`
- 전체 시스템 분석: `experiments/analyze_dual_value_full_system.py`
- 최종 분석: `data/benchmarks/multihop/context_reader_aware/analysis/dual_value_full_system.json`
- 전체 문항 예측: `data/benchmarks/multihop/context_reader_aware/analysis/dual_value_full_system.predictions.jsonl`
