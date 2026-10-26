# Multi-Hop QA 검색 정책 학습 실험 계획

> 이 문서는 논문 실험의 단일 기준 문서(single source of truth)다.
> 이후 실험 코드, 학습 명령, 평가 결과를 변경하거나 해석하기 전에 이 문서를 먼저 확인한다.

## 1. 연구 목표

이 연구의 최종 목표는 **생성/QA 모델을 튜닝하는 것이 아니라 검색 행동 정책만 학습하여 멀티홉 QA의 Answer EM/F1을 얼마나 향상시킬 수 있는지 검증하는 것**이다.

고정할 연구 질문은 다음과 같다.

> Direct 검색 후의 질문·근거 상태로 Graph 확장이 이득인 문항만 고르는 precision-gated 검색 정책이, 고정된 Direct 검색 및 always-Graph 검색보다 멀티홉 QA의 Answer EM/F1과 비용-성능 균형을 향상시킬 수 있는가?

### 1.1 유일하게 학습하는 요소

- 검색 행동을 선택하는 경량 policy model
- 필요하면 policy model의 상태 인코더 또는 작은 prediction head

### 1.2 동결하는 요소

- BM25 검색기
- Dense embedding model
- Cross-encoder reranker
- 문서 Graph 생성 규칙
- 답변 생성/QA 모델(reader)
- QA prompt와 decoding 설정
- 최종 reader 입력 문서 수와 token budget

Graph 생성 규칙이나 retriever를 개발 중에 개선할 수는 있지만, 최종 정책 실험을 시작하기 전에 버전을 고정해야 한다. 최종 비교 도중에는 정책 이외의 구성요소를 변경하지 않는다.

### 1.3 논문 범위에 포함하지 않는 것

- NCS 학습 도구 및 MVP 성능
- 퀴즈 생성 및 학습기록 분석
- QA/LLM fine-tuning
- embedding model 또는 reranker fine-tuning
- LLM 기반 Knowledge Graph 구축
- 서비스 UI 성능

NCS MVP와 논문 실험은 코드 일부를 공유할 수 있지만, 논문의 주장·데이터·결과에서는 분리한다.

## 2. 핵심 가설과 논문 기여

### 2.1 핵심 가설

1. 모든 멀티홉 질문에 동일한 검색 깊이를 적용하는 것은 최적이 아니다.
2. 질문 텍스트와 **Direct 검색 후 근거 상태**를 같이 사용하면 Graph 확장이 필요한 질문을 더 정확히 구별할 수 있다.
3. 순차 STOP/EXPAND는 예비 가설이었지만 3,000문항 pilot에서 one-shot을 안정적으로 넘지 못했다. 따라서 최종 주 방법은 precision-gated one-shot으로 고정하고 순차 정책은 보조 ablation으로 보고한다.
4. Graph 확장이 도움이 되지 않는 질문에서 STOP하면 distractor 유입과 불필요한 검색 비용을 함께 줄일 수 있다.

### 2.2 목표 차별점

논문에서 주장할 차별점은 단순한 adaptive RAG 또는 query complexity routing이 아니다.

- 질문만 보는 single-shot router가 아닌 **post-retrieval evidence-state policy**
- 최초 검색 시점에 2-hop까지 미리 고르는 방식이 아닌 **sequential STOP/EXPAND policy**
- 실제 1-hop 탐색에서 관측한 marginal gain을 다음 상태에 반영
- 의미 이웃과 title bridge로 구성된 경량 문서 Graph의 깊이와 예산을 동적으로 제어
- reader/retriever를 동결하고 policy만 학습하여 성능 변화의 원인을 분리
- supporting-document 성능이 아니라 최종 Answer EM/F1을 주 성능 기준으로 사용

논문의 임시 한 문장 설명은 다음과 같다.

> 고정된 검색기와 QA 모델 사이에서, 현재까지 검색된 근거의 상태를 바탕으로 Graph 탐색의 중단·깊이·예산을 순차적으로 결정하는 경량 정책을 학습하여 멀티홉 QA의 Answer EM/F1을 향상한다.

## 3. 용어 고정

논문과 로그에서 용어를 일관되게 사용한다.

- **Direct Hybrid Retrieval**: BM25 + dense retrieval + RRF/후보 결합 + cross-encoder reranking
- **Document Graph**: 청크 인접성, semantic-neighbor, 문서 제목 언급(title bridge) 간선으로 만든 경량 그래프
- **Graph expansion**: Direct 검색의 seed 문서에서 Document Graph를 따라 새 후보를 추가하는 행동
- **Reader**: 검색된 근거로 최종 답변을 생성하는 동결 QA 모델
- **Policy**: 현재 검색 상태를 입력받아 STOP 또는 Graph 확장을 고르는 유일한 학습 요소
- **Answer Oracle**: 각 질문에서 가능한 검색 행동을 모두 실행했을 때 Answer F1이 가장 높은 행동

현재 코드의 `vector` 행동은 순수 dense/vector 검색이 아니라 Hybrid Retrieval이므로, 논문 표와 설명에서는 `Direct Hybrid`로 표기한다. 코드 내부 ID는 호환성을 위해 당분간 `vector`를 유지할 수 있다.

## 4. 현재 구현의 한계

최종 실험 전에 다음 문제를 해결해야 한다.

### 4.1 corpus 크기가 학습 질문 수에 따라 변함

현재 통합 trainer는 선택한 질문의 문서만 사용해 인덱스를 만든다. 따라서 100문항 pilot과 2,000문항 실험은 질문 수뿐 아니라 corpus 크기와 검색 난이도도 다르다.

**수정 원칙:** 질문 sampling과 corpus construction을 분리한다. 동일 데이터셋/split에서는 학습 질문 수가 달라져도 동일한 고정 corpus와 동일한 Graph를 사용한다.

### 4.2 검색 깊이를 한 번에 선택함

현재 정책은 초기 상태에서 다음 5개 행동 중 하나를 바로 선택한다.

- `vector`
- `graph_1hop_small`
- `graph_1hop_large`
- `graph_2hop_small`
- `graph_2hop_large`

이 구조는 1-hop을 수행한 뒤 얻은 실제 새 근거를 관찰하지 않고 2-hop 여부까지 결정한다.

**수정 원칙:** 상태 전이가 있는 순차 정책으로 변경한다.

### 4.3 정책 목적함수가 최종 QA 성능과 직접 연결되지 않음

현재 utility는 All-Support Recall과 document Recall 중심이며 Answer EM/F1을 사용하지 않는다. 또한 선형 absolute-utility MSE는 행동 간 순위를 직접 최적화하지 않는다.

**수정 원칙:** retrieval supervision으로 사전학습한 뒤, 동결 reader의 counterfactual Answer F1으로 정책을 보정한다.

### 4.4 Graph 양성 행의 고정 8배 가중치

Graph가 조금 이기는 사례와 크게 이기는 사례를 동일하게 8배 처리하면, 일부 데이터셋에서 해로운 Graph 선택이 늘어날 수 있다.

**수정 원칙:** 고정 class weight 대신 행동 간 Answer F1 차이 또는 regret 크기를 학습 가중치로 사용한다.

### 4.5 현재 MVP 답변기는 벤치마크 evaluator가 아님

현재 답변기는 한국어 설명, 근거 번호, `temperature=0.2`를 사용한다.

**수정 원칙:** 영어 단답형 벤치마크 전용 reader/evaluator를 별도 구현한다.

## 5. 데이터와 분할

사용 데이터셋은 다음 세 가지다.

- HotpotQA distractor
- MuSiQue-Ans
- 2WikiMultiHopQA

### 5.1 주 학습/평가 설정

- 정책 학습: 세 데이터셋의 공식 train split
- threshold 및 hyperparameter 선택: train 내부 development split
- 최종 평가: 공식 validation/dev split
- 최종 validation은 최종 설정 확정 전까지 정책 선택이나 threshold 조정에 사용하지 않는다.

세 데이터셋 공동 학습이 논문의 주 실험이다. Leave-one-dataset-out은 unseen-dataset transfer를 보는 보조 실험으로 둔다.

### 5.2 최종 주 설정: 공식 question-local distractor context

최종 실험은 각 문항에 벤치마크가 함께 제공한 distractor 문맥만 사용한다. HotpotQA·2Wiki는 문항당 대략 10개, MuSiQue는 대략 20개 문단이며 다른 문항의 문서를 합치지 않는다.

1. 모든 공식 train 문항을 삭제·downsample 없이 Answer Oracle 라벨로 변환한다.
2. Direct는 고정 Top-K를 사용하고 Graph는 그 Direct frontier 밖의 후보만 추가한다.
3. Graph 간선은 청크 인접, semantic neighbor, title mention bridge만 사용한다.
4. gold support, answer, hop count, decomposition, relation evidence는 Graph 생성·policy feature에 사용하지 않는다.
5. train은 Oracle 생성에만 gold answer를 사용하고, validation은 동결 정책이 행동을 선택한 뒤 EM/F1 계산에만 gold를 사용한다.

2,000문항의 문서를 합친 `benchmark-derived pooled corpus` 결과는 행동 공간과 정책 후보를 검증한 **pilot**로만 보존한다. 전체 train의 문서를 무작정 pooled index로 확대하지 않는다.

### 5.3 전체 데이터셋 혼합

주 학습은 HotpotQA 90,447개, MuSiQue 19,938개, 2Wiki 167,454개, 총 277,839개 train 문항을 모두 사용한다. 가장 작은 데이터셋 크기에 맞춰 버리지 않는다.

주 혼합은 데이터셋별 총 가중치가 같은 `alpha=0` dataset-balanced weighting이다. 행 수가 `n_d`인 데이터셋의 행당 가중치는 `n_d^(alpha-1)`이다. `alpha=0.5` temperature mixing과 `alpha=1` example-proportional mixing은 ablation으로 둔다. 이 선택은 모든 행을 pooled하되 데이터셋별 기여를 균형화한 UnifiedQA와, equal/proportional/temperature mixing을 비교한 T5의 multi-task 설정을 따른다.

### 5.4 데이터 누수 점검

- train/validation 질문 ID 중복 확인
- 정규화된 질문의 exact duplicate 확인
- 정책 특징 정규화/PCA 통계는 train에서만 계산
- validation gold answer/support는 최종 평가에만 사용
- Graph와 index manifest에 corpus fingerprint 기록

## 6. 순차 검색 환경

### 6.1 상태

초기 상태 `s0`는 Direct Hybrid Retrieval 이후 생성한다.

두 번째 상태 `s1`은 선택한 1-hop 행동을 실제 실행하고 reranking한 뒤 다시 생성한다.

정책 입력에는 평가 시점에도 계산 가능한 정보만 사용한다.

- 고정 query embedding 또는 train-only 저차원 projection
- 상위 검색 점수 집중도와 margin
- 검색 결과 source 다양성
- 중복 청크/동일 제목 비율
- Graph connectivity
- Graph frontier 후보 수
- 새 source 후보 수
- title-bridge 수와 path score 통계
- Graph 후보와 Direct tail 후보의 점수 차
- 마지막 확장 전후 score/margin/source 변화
- 마지막 행동과 현재 depth/budget

gold hop count, gold support, gold decomposition, gold answer는 policy feature로 사용하지 않는다.

### 6.2 행동

초기 상태의 행동:

- `STOP`
- `EXPAND_1HOP_SMALL`
- `EXPAND_1HOP_LARGE`

1-hop 이후 상태의 행동:

- `STOP`
- `EXPAND_2HOP_SMALL`
- `EXPAND_2HOP_LARGE`

최대 깊이는 우선 2-hop으로 제한한다. 더 깊은 탐색은 2-hop 정책이 기준선을 안정적으로 넘은 뒤 별도 확장 실험으로 다룬다.

### 6.3 공정한 reader 입력

검색 행동별 후보 수와 reranking 비용은 달라질 수 있지만, reader에 전달하는 최종 조건은 동일해야 한다.

- 동일한 최종 Top-K
- 동일한 token budget
- 동일한 문서 정렬/절단 방식
- 동일한 prompt
- 동일한 decoding 설정

그래야 EM/F1 차이를 더 많은 context를 넣은 효과가 아니라 검색 정책의 효과로 해석할 수 있다.

### 6.4 One-shot과 sequential의 공유 terminal action space

두 정책의 정보 활용 능력을 비교하려면 최종 검색 결과 공간도 같아야 한다.

- One-shot은 초기 상태에서 `Direct / 1-hop small·large / 2-hop small·large` 중 하나를 선택한다.
- Sequential은 초기 상태에서 `STOP / 1-hop small·large`를 고르고, 실제 1-hop 상태를 본 뒤
  `STOP / 2-hop small·large`를 고른다.
- Sequential의 2-hop 종료 결과는 one-shot의 동일한 `graph_2hop_small/large` evidence를 사용한다.
- 따라서 두 방식의 Answer Oracle은 정의상 동일하다.
- 핵심 비교는 Oracle 크기가 아니라 학습된 Sequential Policy의 Answer F1, Policy Regret, Capture Ratio가
  학습된 One-shot Policy보다 좋은지 여부다.

Sequential 경로의 비용에는 1-hop 관측을 얻는 비용과 이어서 선택한 2-hop 행동 비용을 누적하여 기록한다.

## 7. 정책 학습

### 7.1 1단계: evidence-supervised 사전학습

세 데이터셋 train 질문에 대해 가능한 검색 행동을 실행하고 다음 지표를 저장한다.

- supporting-document Recall
- All-Support Recall
- distractor 비율 또는 evidence precision
- 추가 graph 후보 수
- cross-encoder 평가 문서 수
- retrieval latency

이 단계는 reader 호출 없이 대규모 상태-행동 데이터를 만드는 용도다.

### 7.2 2단계: answer-aware 보정

선택한 train 질문에서 모든 terminal retrieval trajectory를 실행한다. 각 terminal context를 동일한 동결 reader에 넣고 답변을 생성한다.

각 행동에 대해 다음을 캐시한다.

- predicted answer
- normalized predicted answer
- gold answer와 alias
- Answer EM
- Answer F1
- context document IDs
- retrieval cost와 latency
- reader model/prompt/config hash

### 7.3 학습 목표

EM은 0/1이라 학습 신호가 희소하므로 Answer F1을 주 학습 신호로 사용한다.

행동 이득은 다음과 같이 정의한다.

```text
advantage(s, a) = AnswerF1(s, a) - AnswerF1(s, STOP)
```

행동 선택 우선순위:

1. Answer F1 최대화
2. F1 차이가 실험 margin 이내이면 EM이 높은 행동
3. EM/F1이 사실상 같으면 비용이 낮은 행동

비용을 성능과 강하게 섞은 단일 임의 reward만 보고하지 않는다. 비용 계수를 바꾼 performance-cost Pareto curve를 별도로 보고한다.

### 7.4 모델 후보

현재 선형 utility regression은 baseline으로 유지한다. 제안 정책 후보는 다음과 같다.

- 작은 MLP
- Gradient Boosted Tree
- STOP 대비 Graph 이득을 예측하는 binary gate + conditional action selector
- 행동별 Q-value/advantage regression
- 행동 쌍의 순서를 학습하는 pairwise ranking model

최종 선택 기준은 action-label accuracy가 아니라 내부 development의 Macro Answer F1과 policy regret이다.

### 7.5 불균형 처리

- 데이터셋별 균형 batch 또는 macro loss
- Graph-positive/negative stratified sampling은 train에서만 사용
- 단순 고정 8배 가중치는 주 방법에서 제거
- pairwise loss는 `|F1_a - F1_b|` 또는 regret 크기로 가중
- reward 차이가 작은 행동은 tie-aware 처리
- 불확실한 예측은 보수적으로 STOP

## 8. 고정 QA reader와 Answer EM/F1 evaluator

논문용 evaluator는 MVP 답변기와 분리한다.

### 8.1 고정 조건

- 주 reader 후보: 현재 로컬 환경의 `qwen3:8b`
- 최종 실험 전에 정확한 model tag/hash 기록
- 영어 단답만 출력하는 benchmark prompt
- `temperature=0`
- reasoning/thinking 출력 비활성화
- deterministic seed를 지원하면 고정
- model, prompt, context, decoding config를 포함한 cache key 사용

가능하면 최종 정책을 재학습하지 않고 두 번째 동결 reader에도 적용하여 cross-reader transfer를 보조 실험으로 확인한다.

### 8.2 답변 평가

- 데이터셋 공식 normalization을 우선 재사용
- lowercase, punctuation, article, whitespace 처리 검증
- MuSiQue `answer_aliases` 반영
- yes/no 및 숫자/날짜 답변 회귀 테스트
- 데이터셋별 EM/F1과 세 데이터셋 Macro EM/F1 계산
- 전체 질문을 합친 Micro 점수는 보조 지표로만 사용

## 9. 가장 먼저 수행할 Answer Oracle audit

정책을 본격적으로 학습하기 전에 데이터셋별 약 300문항으로 action space의 최대 가능성을 확인한다.

```text
B = Direct Hybrid의 Answer F1
O = 가능한 검색 행동 중 최고 Answer F1
P = 학습 정책이 선택한 행동의 Answer F1

Oracle Potential = O - B
Policy Gain      = P - B
Policy Regret    = O - P
Capture Ratio    = (P - B) / (O - B)
```

`O - B`가 거의 0이면 policy training을 확대하지 않는다. 이 경우 Graph candidate generation, path scoring, reranking 또는 action space를 먼저 수정한다.

Audit에서 함께 확인할 값:

- Graph가 Direct보다 F1이 높은 질문 비율
- Graph가 Direct보다 F1이 낮은 질문 비율
- 데이터셋별 Oracle Potential
- hop/budget별 Oracle 행동 분포
- supporting evidence 개선이 실제 Answer F1 개선으로 이어지는 비율
- Gold-support context를 넣었을 때 reader의 EM/F1 상한

## 10. Baseline과 ablation

### 10.1 필수 baseline

- Direct Hybrid Retrieval
- Always 1-hop small
- Always 1-hop large
- Always 2-hop small
- Always 2-hop large
- 기존 rule-based router
- question-only router
- 현재 one-shot evidence-state linear policy
- 제안 sequential evidence-state policy
- Answer Oracle
- Gold-support context reader reference

외부 방법의 이름을 사용할 때는 설정을 충실히 재현한 경우에만 해당 방법의 재현 결과라고 표기한다. 단순히 비슷한 router를 만든 경우 `Adaptive-RAG-style` 또는 `question-only baseline`처럼 구분한다.

### 10.2 필수 ablation

- query feature 제거
- evidence-state feature 제거
- graph-topology feature 제거
- 1-hop 후 재관찰 없이 one-shot 선택
- answer-aware 보정 제거
- pairwise/ranking objective 제거
- cost tie-break 제거
- title-bridge 제거
- semantic-neighbor edge 제거
- small/large budget 구분 제거

## 11. 최종 평가 지표

### 11.1 주 지표

- Answer EM
- Answer F1
- 데이터셋별 EM/F1
- 세 데이터셋 Macro EM/F1

### 11.2 검색 및 정책 분석 지표

- supporting-document Recall@K
- All-Support Recall@K
- evidence precision 또는 distractor 비율
- Policy Regret
- Oracle action agreement는 보조 지표로만 사용
- Graph-beneficial 질문에 대한 Graph 선택 precision/recall
- harmful expansion rate
- missed beneficial expansion rate
- 행동 및 hop/budget 분포

### 11.3 효율 지표

- 평균 retrieval 단계 수
- 평균 Graph 확장 후보 수
- 평균 cross-encoder pair 수
- 질문당 retrieval latency
- 질문당 전체 QA latency
- performance-cost Pareto curve

## 12. 통계와 재현성

- 최소 3개 policy random seed
- 동일 질문에 대한 paired bootstrap 95% confidence interval
- 데이터셋별 점수를 먼저 계산한 뒤 Macro 평균
- 모든 실험 설정을 manifest에 저장
- corpus/index/Graph fingerprint 저장
- policy model, reader model, prompt, dependency version 기록
- 질문별 선택 행동, 상태 특징, 문서 ID, 답변, EM/F1을 JSONL로 저장
- 최종 표는 저장된 prediction JSONL에서 다시 계산 가능해야 함

## 13. 실험 단계와 중단 조건

### Phase 0. 실험 기반 고정

- [x] 고정 pooled corpus builder 구현
- [x] 질문 sampling과 corpus construction 분리
- [x] split/corpus/index manifest 구현
- [x] 공식 question-local context 스트리밍 Oracle 구현
- [x] 재시작 가능한 compact JSONL/manifest 구현
- [ ] Direct Hybrid 용어와 로그 정리
- [ ] 데이터 누수 검사 추가

### Phase 1. QA evaluator

- [x] benchmark 전용 reader prompt 구현
- [x] deterministic generation 설정
- [x] 데이터셋 공식 answer normalization/EM/F1 구현
- [x] alias/yes-no/숫자/날짜 단위 테스트
- [x] retrieval/answer cache 구현
- [x] 동일 Top-K 및 deterministic character budget pilot 구현
- [ ] 논문 최종 실행용 tokenizer 기반 동일 token budget 구현

### Phase 2. Answer Oracle pilot

- [x] 데이터셋별 300문항의 모든 one-shot terminal action 실행
- [x] Direct/Oracle/Gold-support EM/F1 계산 코드 구현
- [x] Graph-beneficial 비율 계산 코드 구현
- [x] Oracle Potential과 gold-support reader reference 분석

300문항×3 데이터셋 audit 결과에서 Macro Direct/Oracle F1은 `0.484/0.569`였고 Oracle Potential은
`+0.084`였다. 데이터셋별 Oracle F1 이득은 HotpotQA `+0.052`, MuSiQue `+0.061`, 2Wiki `+0.140`이며,
Graph-beneficial 문항은 900개 중 91개(`10.1%`)였다. stratified paired bootstrap 95% 구간은
약 `[+0.068, +0.102]`였다. Answer 개선 중 evidence recall도 상승한 비율은 전체 `4.9%`, recall이 같은
상태에서 reader 답변만 개선된 비율은 `5.2%`였다.

small/large 문맥은 약 71~84% 동일하고 F1도 95~99% 동일했지만, 행동을 제거한 축소 Oracle은 특히
2Wiki에서 F1을 최대 `0.043` 잃었다. 2-hop과 large 행동은 희소하지만 고유 이득이 있으므로 순차 audit까지
현재 행동 공간을 유지한다.

**중단 조건:** 세 데이터셋 Macro Oracle Potential이 의미 있게 양수가 아니면 policy 규모를 늘리지 않고 retrieval action space를 수정한다.

### Phase 3. 순차 policy 환경

- [x] 1-hop 실행 후 새 EvidenceProfile 생성
- [x] STOP/1-hop/2-hop state transition 구현
- [x] small/large budget별 trajectory 저장
- [x] 실제 marginal evidence 변화 feature 추가
- [x] 기존 one-shot executor를 baseline으로 유지
- [x] one-shot과 sequential의 terminal action/evidence 공간 통일
- [x] 공유 Terminal Oracle 일치 검증 및 회귀 테스트
- [x] 기존 one-shot cache에서 reader 재호출 없이 v6 순차 상태를 만드는 변환기 구현
- [x] 수정된 v6 순차 상태 데이터셋별 300문항 생성

초기 sequential pilot은 1-hop reader-visible 결과에서 별도의 2-hop 후보를 다시 생성하여 one-shot과 terminal
공간이 달랐다. 50문항 결과에서 HotpotQA `+0.016`, MuSiQue `+0.026`, 2Wiki `-0.040`으로 데이터셋별 변화가
엇갈리고 전체에서 우연히 `+0.001`만 남았다. 이 수치는 정책 우열이 아니라 서로 다른 검색 공간의 차이이므로
최종 결과에서 제외한다. v5부터 두 정책은 같은 terminal evidence를 공유하며, 실제 정책 학습 성능으로 비교한다.

### Phase 4. 정책 학습 pilot

- [ ] evidence-supervised pretraining 구현
- [x] Answer F1 advantage regression 기반 one-shot/sequential 정책 학습 구현
- [x] dataset-balanced sample weight 구현
- [x] fold 내부 STOP threshold 보정 및 최종 OOF threshold 구현
- [x] 데이터셋별 300문항 중첩 5-fold pilot
- [ ] 질문 특징 및 state feature 개선 후 MuSiQue 성능 붕괴 제거
- [ ] 데이터셋별 500문항 이상으로 확대

300문항×3 v6 상태에서 공유 Terminal Oracle은 one-shot과 정확히 일치했다. 기본 s1 행동에서
`RETURN_DIRECT`는 제외한다. Extra Trees 정책의 nested 5-fold
결과는 Macro F1 `Direct 0.484 / one-shot 0.490 / sequential 0.503 / Oracle 0.569`였다. Sequential gain은
Direct 대비 `+0.019`, one-shot 대비 `+0.013`, Oracle potential capture는 `22.1%`였고, 실제 2-hop 재결정의
ablation 기여는 `+0.005`였다. 데이터셋별 sequential gain은 HotpotQA `+0.006`, MuSiQue `-0.010`,
2Wiki `+0.060`이었다. 따라서 전체 가능성은 확인했지만 특정 데이터셋 붕괴가 없어야 한다는 Phase 4 통과
조건은 아직 충족하지 못했다. 다음 작업은 규모 확장보다 MuSiQue의 false expansion과 특징 부족 분석이다.

오류 분석에서 MuSiQue 하락 12건의 대부분은 Direct F1 1.0인 질문을 불필요하게 1-hop 확장해 F1 0으로 만든
경우였다. 이를 막기 위한 `RETURN_DIRECT` ablation은 Macro gain을 `+0.019`에서 `+0.012`로 낮췄고 MuSiQue도
`-0.009`로 개선하지 못해 기본 행동에서 제외했다. 질문 통계와 64차원 hashed lexical feature ablation도
Macro sequential gain을 약 `+0.004`로 낮추고 MuSiQue를 더 악화시켜 채택하지 않았다. 현재 Answer Oracle
Graph-positive는 HotpotQA 18개, MuSiQue 27개, 2Wiki 46개뿐이므로 다음 실험은 데이터셋별 1,000문항까지
Answer cache를 확장해 희소 positive 수를 먼저 늘린다.

데이터셋별 1,000문항으로 확장한 결과 전체 Oracle Potential은 `+0.082`, Graph-beneficial은 290/3,000
(`9.7%`)으로 300문항 audit과 거의 동일했다. raw advantage regression은 Macro gain `+0.013`이었지만
MuSiQue가 `-0.009`였고 sequential-one-shot 차이는 `+0.001`뿐이었다. 반면 질문 TF-IDF와 EvidenceProfile로
Graph-beneficial 여부를 먼저 분류하는 precision gate를 추가하자 다음 nested 5-fold 결과를 얻었다.

- HotpotQA Direct/Gated one-shot/Gated sequential/Oracle F1: `0.729/0.735/0.735/0.772`
- MuSiQue: `0.258/0.267/0.269/0.322`
- 2Wiki: `0.472/0.517/0.510/0.609`
- Macro: `0.486/0.506/0.505/0.568`

Gated one-shot의 Direct 대비 Macro gain은 `+0.020`, stratified paired bootstrap 95% CI는
`[+0.0124,+0.0274]`였다. 다섯 fold가 모두 양수이고 세 데이터셋 평균도 모두 향상됐다. Gated sequential의
Direct 대비 gain도 `+0.018`이지만, sequential-one-shot은 `-0.0017`, CI `[-0.0060,+0.0026]`로 유의한
차이가 없었다. 현재 데이터는 precision-gated retrieval policy의 효과는 지지하지만 sequential 우위 가설은
지지하지 않는다.

동일 설정의 policy seed `42/43/44` 반복 결과 Gated one-shot Macro gain은
`+0.0200/+0.0204/+0.0187`, Gated sequential은 `+0.0182/+0.0192/+0.0189`였다. 모든 seed의 모든
데이터셋과 nested fold에서 Direct 대비 gain이 양수였다. 반면 sequential-one-shot은 seed에 따라
`-0.0017/-0.0013/+0.0001`로 방향이 바뀌었다. 따라서 validation 전에 다음을 동결한다.

- 주 제안법: precision-gated **one-shot** retrieval policy
- 정책 후보 선택 데이터: pooled-corpus pilot train 1,000문항 × HotpotQA/MuSiQue/2Wiki
- 최종 정책 학습 데이터: 공식 question-local context의 세 train split 전체 277,839문항
- 정책 seed: 42/43/44를 각각 평가하고 평균±표준편차 보고
- gate/value 모델, threshold, 검색 action space, Qwen3:8B reader prompt를 validation 결과 확인 후 변경하지 않음
- 고정 Graph baseline: full train Macro에서 미리 선택한 always-Graph 행동; validation에서 다시 선택하지 않음
- sequential은 Direct 대비 보조 결과와 2-hop ablation으로만 보고하고 주 방법 우위 주장에 사용하지 않음

강한 정적 baseline 비교에서 `always graph_1hop_large`의 train Macro F1은 `0.50384`, Gated one-shot OOF는
`0.50630`으로 차이는 `+0.00246`이었다. 데이터셋별 정책−정적 차이는 HotpotQA `-0.00012`, MuSiQue
`+0.00734`, 2Wiki `+0.00016`이고, Graph 선택률은 정책 `34.4%` 대 정적 baseline `100%`다. 그러므로
Direct 대비 `+0.020`만으로 adaptive policy의 우위를 주장하지 않는다. validation에서는 다음 두 가설을 분리한다.

1. Gated policy가 Direct Hybrid보다 EM/F1을 높이는가.
2. Gated policy가 always-Graph와 비슷하거나 더 높은 F1을 더 낮은 Graph 호출률/latency로 달성하는가.

**통과 조건:** 내부 development에서 제안 정책의 Macro Answer F1이 Direct Hybrid보다 높고, 특정 데이터셋의 큰 성능 붕괴가 없어야 한다.

### Phase 4B. 증거 조건부 Bridge 경로 행동 파일럿

기존 Graph 행동은 확장 문서를 원래 질문에만 다시 대조하므로, 첫 문서에서 드러난 중간 엔티티와 두 번째
문서의 연결이 최종 재정렬에서 사라질 수 있었다. 이를 해결하기 위해 고정 retriever/reranker를 유지하면서
`질문 + 1차 근거`와 다음 후보를 cross-encoder로 평가하고, anchor-candidate 쌍을 reader 문맥 앞쪽에 함께
배치하는 두 행동을 별도 실험 공간에 추가했다.

- `bridge_path_1`: 가장 높은 경로 한 쌍과 Direct fill-in
- `bridge_path_2`: 서로 다른 상위 경로 최대 두 쌍과 Direct fill-in
- 고정 설정: seed 4개, seed별 후보 12개, 1-hop, anchor text 700자, Top-K 5
- gold 정답·support·decomposition은 행동 생성이나 점수 계산에 사용하지 않는다.

pooled-corpus train 100문항×3 retrieval screen의 All-Support Recall은 다음과 같다.

| Dataset | Direct | Old Graph 1-hop large | Bridge path 1 | Bridge path 2 |
|---|---:|---:|---:|---:|
| HotpotQA | 0.870 | 0.880 | **0.950** | 0.900 |
| MuSiQue | **0.290** | 0.210 | 0.250 | 0.180 |
| 2Wiki | 0.430 | 0.460 | 0.770 | **0.910** |
| Macro | 0.530 | 0.517 | 0.657 | **0.663** |

동일 100문항×3 frozen-reader Answer F1 결과는 다음과 같다.

| Dataset | Direct | Always Bridge 1 | Always Bridge 2 | Old Oracle | Expanded Oracle |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 0.815 | **0.841** | 0.836 | 0.860 | **0.926** |
| MuSiQue | 0.277 | **0.282** | 0.258 | 0.317 | **0.362** |
| 2Wiki | 0.558 | **0.725** | 0.691 | 0.669 | **0.838** |
| Macro | 0.550 | **0.616** | 0.595 | 0.615 | **0.709** |

새 행동을 포함한 Expanded Oracle은 Direct보다 `+0.159`, 기존 Oracle보다 `+0.094` 높았다. Graph-beneficial
라벨도 300개 중 55개(`18.3%`)로 기존 행동 공간보다 덜 희소해졌다.

질문 TF-IDF, 기존 EvidenceProfile, 정답을 사용하지 않는 Bridge path audit 특징으로 Direct/기존 Graph/Bridge를
선택하는 nested 5-fold 정책 결과는 다음과 같다.

- Direct Macro EM/F1 `0.500/0.550`
- 정책 Macro EM/F1 `0.547/0.598`, gain `+0.047/+0.049`
- Answer Oracle Macro F1 `0.709`
- 확장 실행률 `22.3%`
- stratified paired bootstrap 95% CI: EM `[+0.0233,+0.0733]`, F1 `[+0.0251,+0.0739]`
- seed 42/43/44 F1 gain `+0.0486/+0.0472/+0.0486`
- 데이터셋별 F1 gain: HotpotQA `-0.005`, MuSiQue `+0.013`, 2Wiki `+0.137`

Always Bridge 1의 Macro F1 `0.616`은 현재 정책 `0.598`보다 높지만, 정책은 질문의 약 22~24%만 확장한다.
그러므로 이 결과는 **+5점 수준의 실제 정책 향상 가능성**을 지지하지만, 아직 정적 Bridge보다 높은 정확도를
증명하지는 않는다. 표본이 작고 pooled corpus인 후보 선택 pilot이므로 논문 최종 결과로 사용하지 않는다.

다음 통과 순서는 다음과 같다.

1. 같은 고정 설정을 1,000문항×3까지 확장해 Answer F1, harmful expansion, 정적 Bridge 비교를 재확인한다.
2. 1,000문항 정책 seed 42/43/44의 OOF gain과 데이터셋별 붕괴 여부를 확인한다.
3. 통과하면 Bridge 행동을 공식 question-local context evaluator와 full-data trainer에 편입한다.
4. 공식 train 전체에서 설정·threshold를 동결한 뒤 공식 validation을 한 번 평가한다.

### Phase 4C. 홉 수를 모르는 상태 기반 STOP/CONTINUE 검색 정책

실제 입력의 필요 홉 수는 2·3·4 중 하나로 고정되지 않으며 5-hop, 10-hop 이상일 수도 있다. 따라서
`2-hop / 3-hop / 4-hop`을 직접 예측하는 다중 분류는 최종 정책으로 사용하지 않는다. 2-hop title
bridge만으로는 MuSiQue의 깊은 연결을 일반화했다고 주장할 수도 없다. 특히
`Peter Delorge → Belgium → Albert I/First World War → Italy`처럼 다음 문서의 **제목**이 아니라 현재
근거에서 얻은 **중간 답**으로 다음 검색을 수행해야 하는 깊은 문항이 있다.

최종 실행 단위는 다음 반복 상태 기계다.

1. 현재 질문, 검색 메모리, reader-visible 근거, 이전 검색어와 새 근거 점수를 관찰한다.
2. 정책이 `STOP` 또는 `CONTINUE`를 선택한다.
3. `CONTINUE`이면 고정 query planner가 다음 검색어를 만들고 corpus를 다시 검색해 메모리를 갱신한다.
4. 새 상태에서 1~3을 반복한다.
5. 주 실험의 안전 상한은 최대 evidence 4-hop, 즉 초기 검색 후 최대 3회 CONTINUE로 고정한다.

이를 위해 다음 두 실행기를 분리해 추가했다.

- `iterative_continue_t`: 원 질문과 누적 근거로 고정 hybrid retriever를 다시 호출하고 graph 후보와 합치는 전이
- `planned_continue_t`: 고정 Qwen3:8B가 누적 근거로 다음 검색어만 재작성하는 전이
- 검색 메모리는 reader Top-K와 분리해 누적하며, 10-step도 동일 실행기와 행동 `CONTINUE`를 반복 사용
- gold hop은 균형 표본 추출과 사후 분석에만 사용하며 정책 상태에는 포함하지 않음
- planner/retriever는 동결하며 gold answer, gold support, MuSiQue decomposition을 행동 생성에 사용하지 않음
- 2/3/4-hop을 같은 개수로 추출해 2-hop 다수 표본이 깊은 홉 실패를 가리지 못하게 함

제목/semantic 확장만 사용한 최초 20문항/hop screen은 4-hop All-Support가 모든 행동에서 `0.000`이었다.
이는 단순히 graph depth를 늘리는 방식이 실패한다는 진단 결과다. 고정 query reformulator를 추가한 동일
20문항/hop screen은 다음과 같다.

| Gold hop | Direct Recall/All | Planned 2-hop | Planned 3-hop | Planned 4-hop | Depth Oracle |
|---|---:|---:|---:|---:|---:|
| 2 | 0.725/0.450 | 0.725/0.450 | 0.675/0.350 | 0.625/0.300 | 0.775/0.550 |
| 3 | 0.567/0.100 | 0.600/0.050 | 0.600/0.150 | **0.583/0.250** | 0.700/0.300 |
| 4 | 0.362/0.000 | 0.338/0.000 | 0.375/0.050 | **0.400/0.100** | 0.425/0.100 |
| Macro-hop | 0.551/0.183 | 0.554/0.167 | 0.550/0.183 | **0.536/0.217** | 0.633/0.317 |

3-hop은 Planned 4-hop All-Support가 Direct보다 `+0.150`, 4-hop은 `+0.100`이었고, 전체 Depth Oracle은
`0.183 → 0.317`로 `+0.134`였다. 반대로 2-hop 문항에 불필요하게 3/4-hop을 강제하면 성능이 하락했다.
따라서 이 결과의 의미는 always-expand가 좋다는 것이 아니라, **중간 답 기반 재검색이 깊은 홉에서 실제 새
근거를 만들며 STOP/CONTINUE 정책을 학습할 가치가 있다**는 것이다. 여기서 `Planned 2/3/4-hop` 표기는
각각 검색을 1/2/3회 수행한 prefix의 과거 진단 이름일 뿐 최종 정책의 class가 아니다. n=20/hop retrieval screen이므로 논문
성능 결과로 사용하지 않는다.

실행기의 일반성을 확인하기 위해 `max_search_steps=10`으로도 10개 prefix와 각 상태의 STOP/CONTINUE
offline label이 생성됨을 확인했다. 다만 논문 주 실험은 계산량과 사용 벤치마크의 범위를 고려해 최대 4-hop으로
고정하며, 실제 종료 시점은 그 범위 안에서 학습 정책이 결정한다.

다음 통과 순서는 다음과 같다.

1. MuSiQue 100문항/hop, 최대 evidence 4-hop으로 trajectory screen을 확장하고 harmful continuation을 계산한다.
2. 같은 trajectory의 모든 prefix에 고정 reader를 적용해 Answer EM/F1 stop-step Oracle을 측정한다.
3. 각 상태에서 `STOP / GRAPH_CONTINUE / REQUERY_CONTINUE`의 Answer-F1 advantage label을 생성한다.
4. gold hop을 feature로 쓰지 않는 sequential policy를 train 전체에서 학습한다.
5. HotpotQA·2Wiki는 2-hop baseline/전이 평가, MuSiQue는 2/3/4-hop 주 깊이 일반화 평가로 보고한다.

### Phase 5. 규모 확장과 학습 곡선

- [x] 전체 train 스트리밍 공식-context Oracle 코드 구현
- [x] 데이터셋 잘라내기 없이 dataset-balanced/temperature/proportional 혼합 구현
- [x] full-data one-shot nested OOF trainer 구현
- [ ] 데이터셋별 100/1,000/5,000/전체 질문 학습 곡선
- [ ] Graph-positive 수와 성능 포화점 확인
- [x] 세 데이터셋 공동 학습
- [x] policy seed 3회 이상 반복
- [ ] 주요 ablation 실행

최종 주 결과는 모든 raw train 행을 사용한다. 실행은 오래 걸리므로 행당 즉시 저장, manifest 일치 검사, SQLite reader cache로 중단 후 이어하기를 보장한다. 학습 곡선은 최종 전체 실험을 대체하지 않고 sample efficiency 분석으로만 보고한다.

### Phase 6. 최종 평가

- [ ] full-data OOF로 설정, checkpoint, threshold 동결
- [ ] 공식 validation에 최초 최종 실행
- [ ] 모든 baseline과 동일 reader 조건 비교
- [ ] 데이터셋별 및 Macro EM/F1 계산
- [x] validation용 동결 정책 evaluator와 paired bootstrap CI 구현

1. 공식 train 전체에서 문항별 다섯 검색 행동의 Answer EM/F1을 한 번 생성한다.
2. 모든 train 행을 사용하고 dataset-balanced weighting을 적용해 seed 42/43/44 정책을 학습한다.
3. train nested OOF에서 항상-Graph 행동과 gate threshold를 고정한다.
4. 공식 validation 전체도 동일한 question-local context/동결 reader로 다섯 행동을 한 번 생성한다.
5. `evaluate_gated_policy`가 저장된 seed 42/43/44 모델과 threshold를 재학습 없이 그대로 적용한다.
6. train-validation ID 중복과 데이터셋별 reader signature 일치를 검사한다.
7. 데이터셋별 EM/F1, Macro EM/F1, Graph 선택률, Oracle potential capture, seed 평균±표준편차를 보고한다.
8. 주 지표 Macro F1의 Policy−Direct 및 Policy−고정 Graph 차이에 stratified paired bootstrap 20,000회를 적용한다.
- [ ] latency/cost 측정
- [ ] error analysis 및 결과 표 생성

### Phase 7. 보조 일반화 실험

- [ ] HotpotQA + MuSiQue 학습 -> 2Wiki 평가
- [ ] HotpotQA + 2Wiki 학습 -> MuSiQue 평가
- [ ] MuSiQue + 2Wiki 학습 -> HotpotQA 평가
- [ ] 가능하면 두 번째 frozen reader transfer 평가

Leave-one-dataset-out은 주 결과가 아니라 cross-dataset transfer를 분석하는 보조 결과다.

## 14. 최종 성공 기준

정책 학습 완료나 높은 oracle-action accuracy를 성공으로 보지 않는다. 다음 조건으로 판단한다.

1. 제안 정책의 Macro Answer F1이 Direct Hybrid보다 높다.
2. 제안 정책의 Macro Answer EM이 Direct Hybrid보다 높다.
3. 최소 2개 데이터셋에서 F1이 향상되고, 나머지 데이터셋에서 큰 붕괴가 없다.
4. Always-Graph baseline보다 성능 또는 비용-성능 균형이 좋다.
5. 순차 policy는 one-shot을 넘을 때만 확장 기여로 주장하고, 그렇지 않으면 보조 음성 결과로 보고한다.
6. paired bootstrap 및 여러 policy seed에서 개선이 안정적이다.
7. Oracle Potential 중 의미 있는 비율을 policy가 회수한다.

목표 결과 표의 핵심 열은 다음과 같다.

| Method | Hotpot EM/F1 | MuSiQue EM/F1 | 2Wiki EM/F1 | Macro EM/F1 | Steps | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Direct Hybrid |  |  |  |  |  |  |
| Always 1-hop |  |  |  |  |  |  |
| Always 2-hop |  |  |  |  |  |  |
| Rule Router |  |  |  |  |  |  |
| One-shot Policy |  |  |  |  |  |  |
| Sequential Policy |  |  |  |  |  |  |
| Answer Oracle |  |  |  |  |  |  |
| Gold Support |  |  |  |  |  |  |

## 15. 결과 해석 원칙

- Retrieval Recall만 오르고 Answer EM/F1이 오르지 않으면 최종 목표 달성으로 보지 않는다.
- 정책이 Direct보다 낮으면 데이터 규모부터 늘리지 않고 Oracle Potential과 선택 오류를 먼저 분석한다.
- Oracle도 Direct보다 낮거나 비슷하면 정책이 아니라 Graph 검색 공간을 수정한다.
- 행동 분포가 균형적인 것을 목표로 하지 않는다. Direct가 실제로 최선이면 STOP 비율이 높은 것이 맞다.
- Graph 선택 비율보다 `Policy Gain`, `Policy Regret`, `harmful expansion rate`를 우선한다.
- validation 결과를 본 뒤 retriever, prompt 또는 threshold를 바꿨다면 해당 결과는 최종 결과가 아니며 새로운 평가 split 또는 명시적 반복 검증이 필요하다.

## 16. 바로 다음 코드 작업

다음 순서를 바꾸지 않는다.

1. **고정 pooled corpus/index 생성 방식 구현** — 완료
2. **벤치마크 전용 Answer EM/F1 evaluator 구현** — 완료
3. **데이터셋별 Answer Oracle pilot 실행** — 완료
4. **동일 terminal action space의 순차 STOP/EXPAND 상태 구현** — 완료
5. **Answer-aware one-shot/sequential 정책 학습 및 내부 비교** — 300문항 pilot 완료
6. **MuSiQue false expansion 및 query/RETURN_DIRECT ablation** — 완료, 두 ablation 모두 기각
7. **Answer-aware 행을 데이터셋별 1,000문항으로 확장** — 완료
8. **precision-gated one-shot/sequential 정책 비교** — 완료, gated one-shot이 현재 최선
9. **policy seed 반복과 설정 동결 후 공식 validation 평가** — 기존 행동 공간 pilot 완료
10. **증거 조건부 Bridge retrieval/Answer pilot 100문항×3** — 완료, 정책 OOF F1 `+0.049`
11. **Bridge retrieval pilot을 1,000문항×3으로 확장** — 완료, 구조 편향 진단
12. **MuSiQue 2/3/4-hop 균형 query-planner screen** — 20문항/hop 완료, 깊은 문항 개선 확인
13. **임의 길이 STOP/CONTINUE 실행기와 10-step trajectory 검증** — 완료
14. **Answer EM/F1 prefix screen과 상태별 STOP/CONTINUE 라벨 생성** — 현재 다음 단계
15. **통과 시 공식 question-local full-data 행동 공간으로 편입** — 대기

현재 우선 작업은 가변 깊이 행동을 MuSiQue의 2/3/4-hop 균형 표본에서 frozen-reader Answer EM/F1로
검증하는 것이다. 이 검증 전에는 전체 공식-context 학습을 시작하지 않는다.

## 17. 결정 기록

### 확정된 결정

- 논문에서 NCS 및 MVP는 다루지 않는다.
- 세 공개 멀티홉 QA 데이터셋을 공동 학습에 사용한다.
- 논문의 최종 주 성능 기준은 Answer EM/F1이다.
- QA 모델 자체는 튜닝하지 않는다.
- 검색 정책만 학습한다.
- 세 데이터셋 공동 학습이 주 실험이다.
- leave-one-dataset-out은 보조 일반화 실험이다.
- 순차 evidence-state policy를 제안 방법으로 개발한다.

### 구현 후 확정할 항목

- 주 reader의 정확한 모델 tag/hash
- 최종 Top-K와 reader token budget
- F1 tie margin
- small/large expansion budget
- 최종 policy architecture
- answer-aware 학습 질문 수
- 비용 계수별 Pareto 평가 범위

이 항목들은 train 내부 development와 Oracle pilot으로 결정하고, 공식 validation 결과를 이용해 선택하지 않는다.
