# NOVA UROP Reader-aware 정책 공식 전체 검증 결과

> 후속 저비용 Student 개선 실험은 `STUDENT_POLICY_IMPROVEMENT_REPORT.md`에 정리했다.
> 아래 평가는 전체 위키 코퍼스가 아니라 질문별 공식 distractor 문서 안에서 검색하는
> question-local closed-context 조건이다.

## 1. 최종 결론

Qwen3:8B를 고정하고 검색 종료만 결정하는 reader-aware counterfactual sufficiency 정책을
HotpotQA, MuSiQue, 2Wiki의 공식 validation 전체에서 검증했다. 정책 학습 표본을 바꾸는
무작위 시드 42~46의 다섯 모델을 동일한 전체 validation에 적용했다.

- 적응형 정책의 평균 F1은 FIXED3보다 HotpotQA `+0.14점`, MuSiQue `+0.03점`, 2Wiki
  `+0.20점` 높았다.
- HotpotQA와 2Wiki의 F1 차이는 paired bootstrap 95% 신뢰구간 하한이 0보다 높았다.
- MuSiQue는 신뢰구간에 0이 포함되므로 우월성이 아니라 동률·비열등으로 해석해야 한다.
- 평균 추가 검색은 HotpotQA `2.84회`, MuSiQue `2.92회`, 2Wiki `2.84회`로 FIXED3의
  3회보다 적었다.
- 그러나 적응형 정책은 각 prefix에서 Qwen reader를 호출한다. 수정된 예상 온라인 지연은
  FIXED3보다 HotpotQA `1.93초`, MuSiQue `2.08초`, 2Wiki `1.91초` 길었다.
- 따라서 `정답 품질을 유지·개선하며 검색 횟수를 줄인다`는 가설은 지지되지만,
  `전체 생성 비용과 지연도 줄인다`는 가설은 현재 구조에서 지지되지 않는다.

최종 결정은 **검증된 연구 후보, MVP 미배포**다. 현재 정책을 논문의 핵심 기술로 보고하되,
중간 Qwen 호출을 대체할 저비용 sufficiency verifier를 구현하기 전에는 기존 MVP 정책을
교체하지 않는다.

## 2. 무엇을 학습했는가

- 동결 모델: Qwen3:8B reader와 query rewriter, multilingual-e5-base embedding,
  cross-encoder reranker
- 학습 모델: `HistGradientBoostingClassifier`와 `HistGradientBoostingRegressor`
- 분류 출력: 현재 prefix에서 STOP해도 될 확률
- 회귀 출력: 현재 STOP할 때 이후 검색 대비 예상 utility 손실(regret)
- 입력: 검색 점수·후보 변화·답변 안정성·근거 grounding·명시적 사실·비교 분기 완성도 등
  39개의 관찰 가능한 상태 특징
- 추론 금지 입력: 데이터셋 이름, reasoning type, gold hop, gold answer, 실제 EM/F1,
  gold supporting document

훈련 질문마다 최초 검색과 추가 검색 1·2·3회 prefix를 생성했다. 학습 시에만 각 prefix의
정답 EM/F1을 계산해 현재 STOP과 이후 CONTINUE의 utility를 비교했다. 실제 추론에서는
gold를 계산하지 않고 학습된 두 모델과 안전 조건만 사용한다.

현재 안전 설정은 답과 근거가 두 번의 변화 후에도 안정적인지 확인한 뒤 추가 검색 2회
시점에서 STOP을 허용한다. 추가 검색 3회는 목표가 아니라 안전 상한이다.

## 3. 데이터 분리와 반복 검증

- train prefix pool: 데이터셋별 train-only 무작위 2,000문항, seed 42
- 정책 학습 시드: 42, 43, 44, 45, 46
- 시드별·데이터셋별 선택: 1,200문항
- 시드별·데이터셋별 모델 fitting: 1,000문항
- 시드별·데이터셋별 threshold tuning: 200문항
- 시드별 전체 fitting/tuning: 3,000/600문항
- 공식 validation: HotpotQA 7,405, MuSiQue 2,417, 2Wiki 12,576문항
- train pool과 validation 질문 ID 중복: 세 데이터셋 모두 0건
- 생성 설정: Qwen3:8B, temperature 0, seed 42
- 통계: 데이터셋별 paired bootstrap 20,000회

다섯 시드의 fitting state는 각각 3,000개다. STOP state는 2,868~2,892개,
CONTINUE state는 108~132개로 불균형했으며, 분류 학습에 balanced class weight와
regret 기반 sample weight를 적용했다.

## 4. 공식 validation 전체 결과

모든 점수는 100점 기준이다. `±`는 다섯 학습 시드 사이의 표준편차다.

| 데이터셋 | 방식 | EM | F1 | 평균 추가 검색 | 예상 온라인 지연(초) |
|---|---|---:|---:|---:|---:|
| HotpotQA | Question-local Direct RAG | 56.60 | 70.32 | 0.00 | 0.89 |
| HotpotQA | Chain FIXED3 | 57.73 | 71.78 | 3.00 | 2.76 |
| HotpotQA | 적응형 5-seed 평균 | 57.86±0.15 | 71.92±0.13 | 2.84±0.15 | 4.69 |
| MuSiQue | Question-local Direct RAG | 30.29 | 39.30 | 0.00 | 1.00 |
| MuSiQue | Chain FIXED3 | 34.92 | 44.49 | 3.00 | 3.10 |
| MuSiQue | 적응형 5-seed 평균 | 34.95±0.05 | 44.52±0.06 | 2.92±0.08 | 5.18 |
| 2Wiki | Question-local Direct RAG | 52.66 | 59.26 | 0.00 | 0.85 |
| 2Wiki | Chain FIXED3 | 57.16 | 64.01 | 3.00 | 2.40 |
| 2Wiki | 적응형 5-seed 평균 | 57.40±0.23 | 64.22±0.21 | 2.84±0.16 | 4.31 |

### FIXED3 대비 적응형 정책

| 데이터셋 | EM 변화 | F1 변화 | F1 변화 95% CI | 검색 절감률 |
|---|---:|---:|---:|---:|
| HotpotQA | +0.13 | +0.14 | +0.06 ~ +0.22 | 5.45% |
| MuSiQue | +0.03 | +0.03 | -0.11 ~ +0.17 | 2.79% |
| 2Wiki | +0.23 | +0.20 | +0.12 ~ +0.29 | 5.46% |

### Direct 대비 적응형 정책

| 데이터셋 | EM 변화 | F1 변화 | F1 변화 95% CI |
|---|---:|---:|---:|
| HotpotQA | +1.26 | +1.60 | +0.99 ~ +2.21 |
| MuSiQue | +4.67 | +5.22 | +3.75 ~ +6.69 |
| 2Wiki | +4.73 | +4.96 | +4.28 ~ +5.65 |

## 5. 시드 안정성

- HotpotQA의 시드별 F1 변화는 FIXED3 대비 `-0.00~+0.29점`, 검색 절감률은
  `0.05~10.89%`였다.
- MuSiQue의 시드별 F1 변화는 `-0.01~+0.13점`, 검색 절감률은 `0.07~6.25%`였다.
- 2Wiki의 시드별 F1 변화는 `+0.00~+0.44점`, 검색 절감률은 `0.06~11.52%`였다.

정답 품질의 시드 표준편차는 작지만 검색 절감량은 시드에 민감했다. 특히 시드 45·46은
거의 항상 세 번째 검색까지 수행했다. 따라서 시드 42를 validation 결과만 보고 선택해
배포하는 것은 사후 선택 편향이므로 허용하지 않는다.

## 6. 지연과 생성 비용

| 데이터셋 | Direct 지연 | FIXED3 지연 | 적응형 지연 | Direct 생성 토큰 | FIXED3 생성 토큰 | 적응형 생성 토큰 |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 0.89초 | 2.76초 | 4.69초 | 49.29 | 51.02 | 189.45 |
| MuSiQue | 1.00초 | 3.10초 | 5.18초 | 50.51 | 50.08 | 195.14 |
| 2Wiki | 0.85초 | 2.40초 | 4.31초 | 58.41 | 60.92 | 228.04 |

FIXED3는 세 번 검색한 뒤 reader를 한 번 호출하지만, 현재 적응형 정책은 답변 안정성을
관찰하기 위해 여러 prefix의 reader 출력을 생성한다. 검색 호출 수 감소가 reader 호출
증가를 상쇄하지 못했으므로 전체 지연과 생성 토큰은 증가했다.

지연은 누적 rewrite 실측치에 `전체 planner 시간 - 전체 rewrite 시간`인 검색·후처리
잔여 시간을 단계 비례 배분해 추정한다. 종단 3단계는 기록된 전체 planner 시간과 정확히
일치한다. 이 수정은 이전 계산에서 빠졌던 잔여 시간을 반영하며 정확도·검색·토큰에는
영향을 주지 않는다.

## 7. 논문에서 가능한 주장

1. gold hop이나 정답 없이 관찰 가능한 reader 상태로 검색 종료를 결정할 수 있다.
2. 학습 표본을 다섯 번 바꿔도 세 데이터셋의 평균 F1은 강한 FIXED3 기준보다 낮지 않았다.
3. HotpotQA와 2Wiki에서는 작은 폭이지만 통계적으로 양의 F1 차이를 확인했다.
4. 정확도와 검색 호출 수만 보면 적응 정책의 효과가 있으나, 전체 생성 비용까지 절감하는
   시스템으로 주장해서는 안 된다.
5. NCS MVP 적용 전에는 Qwen prefix reader를 저비용 verifier로 증류하는 후속 개선이 필요하다.

## 8. 자격증 학습량 분석 도구 적용

멀티홉 QA 정책은 사람의 실제 학습시간을 직접 예측하지 않는다. NCS MVP에서는 검색
궤적에서 얻은 근거 경로 길이, 누락 선수개념 수, 추가 탐색 횟수를 학습 부담 특징으로
사용하고, 수험자의 오답률·반복 실패·가용시간과 결합하는 별도 학습량 산정 계층을 둔다.

논문의 핵심은 적응형 근거 탐색 정책의 일반화 검증이며, NCS 시스템은 이 기술을 자격증
학습자료에 적용하는 MVP다. 실제 학습시간 예측을 주장하려면 NCS 수험자 로그나 전문가
학습량 annotation을 이용한 별도 검증이 필요하다.

## 9. 재현 파일과 검증 상태

- 전체 prefix: `data/benchmarks/multihop/context_reader_aware/full_validation/`
- 정책 모델: `data/models/context_reader_aware_full_validation/policy_seed{42..46}.joblib`
- 시드별 학습 보고서: `data/models/context_reader_aware_full_validation/policy_seed{42..46}.report.json`
- 전체 통계: `data/benchmarks/multihop/context_reader_aware/analysis/full_validation_multiseed.json`
- 전체 실행기: `scripts/run_context_reader_aware_full_validation.sh`
- 자동 테스트: 138개 통과
- Qwen3:8B: 실험 종료 후 GPU unload 확인

## 10. 후속 최종 결과: 이중 가치 SEARCH/REASON/STOP

기존 reader-aware 정책의 비용 문제를 줄이고 정확도를 높이기 위해 검색 필요성과 답변 오류
위험을 두 개의 Extra-Trees head로 분리했다. 다섯 train 표본 시드의 다수결로 `SEARCH`,
`REASON`, `STOP`을 선택한다. `REASON`에는 Qwen3:8B thinking 답을 만들고 기존 답과 새
답을 evidence-grounded 중재기가 비교한다. REASON 큐 전체를 직접 생성하여 validation
22,398문항을 모두 평가했다.

| 데이터셋 | Direct EM/F1 | FIXED3 EM/F1 | 최종 EM/F1 | F1 변화 | F1 95% CI | 평균 검색 | 검색 절감 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 56.60/70.32 | 57.73/71.78 | 58.03/72.18 | +0.41 | [+0.11, +0.71] | 2.65 | 11.75% |
| MuSiQue | 30.29/39.30 | 34.92/44.49 | 34.96/44.57 | +0.08 | [-0.14, +0.31] | 2.95 | 1.74% |
| 2Wiki | 52.66/59.26 | 57.16/64.01 | 57.55/64.61 | +0.59 | [+0.23, +0.95] | 2.61 | 13.15% |

HotpotQA와 2Wiki는 FIXED3보다 검색을 줄이면서 F1을 유의하게 높였다. MuSiQue는 우월성
신뢰구간은 확보하지 못했지만 평균 F1이 높고 1점 비열등 확률은 100%였다. 반면 선택적
추론과 중재 때문에 FIXED3 대비 지연은 `3.93~7.06%`, reader 생성 토큰은
`6.01~24.07%` 증가했다. 따라서 논문에서는 정확도·검색 효율 개선으로 주장하되, 전체
생성 비용 절감으로 표현하지 않는다.

- 전체 분석: `data/benchmarks/multihop/context_reader_aware/analysis/dual_value_full_system.json`
- 전체 예측: `data/benchmarks/multihop/context_reader_aware/analysis/dual_value_full_system.predictions.jsonl`
- 자동 테스트: 141개 통과
- Qwen3:8B: 전체 실험 종료 후 GPU unload 확인
