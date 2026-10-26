# NOVA Study Map

자격증·NCS 학습 자료를 로컬에서 검색하고, 근거 기반 설명과 확인 퀴즈를 제공하는 Streamlit 시스템입니다. 문서와 질의는 외부 API로 전송하지 않으며, 생성 모델은 Ollama를 통해 로컬에서 실행합니다.

## GitHub 동기화

코드 저장소는 [iion74/UROP](https://github.com/iion74/UROP)입니다. 수정·검증을 마친 작업 파일만
커밋·푸시하며, 데이터셋·모델·논문 원본·인증정보는 기본 업로드 대상에서 제외합니다.
최초 로그인과 사용 방법은 [GitHub 동기화 안내](docs/GITHUB_SYNC.md)를 참고하세요.

## 시작하기

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen3:8b
streamlit run UROP.py
```

퀴즈 생성 속도는 `.env`에서 조절할 수 있습니다. 기본값은 근거 2개와 최대 420토큰이며,
더 빠르게 하려면 `QUIZ_MAX_TOKENS=320`으로 낮출 수 있습니다. `OLLAMA_KEEP_ALIVE=30m`은
모델을 메모리에 유지해 연속 생성의 대기 시간을 줄입니다.

`qwen3:8b`의 양자화 모델은 24 GB VRAM에서 구동하기 적절한 시작점입니다. 시스템 메모리와 생성 품질이 충분하면 `qwen3:14b`도 후속 비교 대상으로 사용할 수 있습니다.

## 자료 넣기

`data/` 폴더 아래의 원하는 하위 폴더에 다음 파일을 넣습니다. 현재 넣어둔 NCS 자료처럼
`data/01.정보기술/07.인공지능/...` 구조를 그대로 유지해도 됩니다. `data/index/`는 검색
인덱스 저장용이므로 원본 자료를 넣지 않습니다.

- PDF: NCS 학습모듈, 출제기준, 공개 학습 자료
- TXT/MD: 정리한 개념 노트
- CSV: 개념 목록 또는 문제·해설 표

앱의 **자료 읽기 및 인덱스 생성** 버튼을 누르면 검색 인덱스가 `data/index/`에 만들어집니다. 인덱스는 원본 자료가 아니므로 삭제 후 다시 생성해도 됩니다.

## 현재 기능

- PDF/TXT/MD/CSV 문서 로드 및 청킹
- BM25 + 다국어 임베딩을 RRF로 결합한 Hybrid Retrieval
- 초기 검색 근거의 불확실성·출처 분산·그래프 연결성 기반 적응형 라우팅
- 문서 인접성·의미 유사도 기반 경량 청크 그래프와 근거-증가량 기반 예산형 확장
- 검색 근거를 제한한 Ollama 기반 질문 답변
- 검색 근거를 제한한 확인 퀴즈 생성
- 자료별 학습 단위 확인 화면

## 논문 실험과의 관계

이 앱의 검색 모듈은 이후 BM25, Dense Retrieval, Hybrid RRF, 그래프 기반 재정렬을 비교하는 실험으로 확장할 수 있습니다. 서비스용 NCS 자료와 논문용 공개 벤치마크(MIRACL Korean)는 분리해 관리합니다.

## 연구 모듈

`src/routing.py`의 경량 라우터는 질문 문구만으로 난이도를 추정하지 않습니다. 초기 Hybrid Retrieval
결과의 점수 집중도·점수 차·출처 분산도·근거 중복도와 청크 그래프의 연결성을 측정한 뒤, 그래프 확장으로
얻을 예상 근거량이 비용보다 클 때만 `종료 / 1-hop / 2-hop`을 선택합니다. 확장 시에도 추가 청크 수를
3개 또는 6개로 제한하고, 새 출처·연결 근거·의미적 새로움을 많이 더하는 청크를 우선 선택합니다.

그래프는 인덱싱 시 `data/index/chunk_graph.json`에 저장되며 문서 인접 청크, FAISS 의미 이웃, 그리고
문단에서 명시적으로 언급된 문서 제목을 잇는 title-bridge 간선으로 생성됩니다. title bridge는 HotpotQA의
영문 Wikipedia 문단에서 첫 근거 문서가 언급한 다른 문서로 이동하게 하는 경량 관계 신호입니다. 즉 LLM
엔티티 추출형 Knowledge Graph가 아니라, 공개 다중-hop QA 벤치마크에서도 비용과 지연시간을 비교할 수
있는 경량 청크 그래프입니다. 라우터와 retriever는 UI와 분리되어 있으므로
HotpotQA·MuSiQue·2WikiMultiHopQA용 실험 코드에서 그대로 재사용할 수 있습니다.

그래프 정책은 `.env`에서 `ROUTER_SEED_COUNT`, `GRAPH_SMALL_BUDGET`, `GRAPH_LARGE_BUDGET`,
`GRAPH_UTILITY_THRESHOLD`, `GRAPH_EXPANSION_DECAY`, `GRAPH_SEMANTIC_NEIGHBORS`로 조절할 수 있습니다.
`GRAPH_SEMANTIC_NEIGHBORS`의 기본값은 24이며, 정책 학습에서 직접 검색 후보(15개) 밖의 새 근거를
그래프가 제안할 수 있게 합니다. `GRAPH_TITLE_BRIDGES=true`는 문서 제목 언급 기반 bridge 간선을 켜며,
`GRAPH_MAX_TITLE_BRIDGES_PER_CHUNK=8`로 청크별 bridge 수를 제한합니다. 논문 실험에서는 답변 정확도뿐
아니라 Evidence Recall, 질의당 추가 reranking 수, 지연시간을 동일한 검색 예산에서 보고합니다.

### HotpotQA 정책 학습 준비

먼저 HotpotQA distractor 설정을 서비스용 NCS 자료와 분리된 폴더에 저장합니다.

```bash
.venv/bin/python -m experiments.download_hotpotqa \
  --output-dir data/benchmarks/hotpotqa/distractor
```

이후 아래처럼 **검색 행동 정책만** 학습할 수 있습니다. 생성 LLM은 호출하지 않습니다. 각 질문에 대해
`Vector`, `1-hop/2-hop × 추가 후보 3/6개`를 실행하고, supporting document 회수율에서 추가 reranking
비용을 뺀 **행동별 효용 전체**를 학습합니다. 그래프 행동이 실제로 Vector보다 이긴 희소 질문은 기본 8배
가중해, 정책이 단순히 Vector만 선택하는 현상을 줄입니다.

```bash
.venv/bin/python -m experiments.train_hotpot_policy \
  --hotpot data/benchmarks/hotpotqa/distractor \
  --split train \
  --index-dir data/benchmarks/hotpot/index \
  --model-out data/models/hotpot_action_policy.npz \
  --max-examples 2000
```

실행 중에는 다음이 터미널에 즉시 출력됩니다. 새 인덱스 생성 시에는 임베딩 진행 바도 표시되고, 라벨 생성
단계에서는 질문 수·질문당 시간·ETA·oracle Evidence Recall@5·현재 행동 분포가, 정책 학습 단계에서는
epoch별 효용 예측 loss·oracle 행동 일치율·선택 행동의 실제 train 효용·ETA가 보입니다. 기본 로그 간격은 라벨 25문항, epoch 5회이며 필요하면
`--label-log-every 10 --epoch-log-every 1`처럼 더 촘촘하게 바꿀 수 있습니다.

```text
[3/5] 라벨 250/2,000 (12.5%) | 1.82초/질문 | ETA 53분 05초 | oracle Recall@5=0.714 | ...
[4/5] epoch  20/80 (25%) | train loss=0.9821 | train action-accuracy=0.604 | 경과 2초 | ETA 6초
```

처음에는 `--max-examples 2000`으로 파이프라인을 확인한 뒤, 데이터·VRAM·시간 여유에 맞춰 늘립니다.
HotpotQA 벤치마크 인덱스는 서비스용 `data/index/`와 분리해야 합니다. 라벨 행은
`data/benchmarks/hotpot/policy_rows.jsonl`에 저장되며, 모델의 개발 분할 효용·규칙 정책 효용·oracle
효용을 함께 출력합니다.

생성된 모델을 MVP에 적용하려면 `.env`에 다음을 추가합니다.

```bash
POLICY_MODEL_PATH=data/models/hotpot_action_policy.npz
POLICY_MIN_EXPECTED_ADVANTAGE=0.01
```

효용 정책은 Graph 행동의 예상 효용이 Vector보다 최소 0.01 이상 높을 때만 Graph를 선택합니다. 모델 파일을 읽을 수 없으면 기존 규칙 기반 정책으로 자동 폴백합니다.
HotpotQA는 영어 Wikipedia 기반이므로, NCS 자료에 적용한 결과는 교차 도메인 실험일 뿐 성능 보장은 아닙니다.

### 추가 멀티홉 벤치마크

MuSiQue v1.0과 2WikiMultiHopQA의 공식 수정 배포본은 다음 명령으로 함께 받을 수 있습니다.

```bash
.venv/bin/python -m experiments.download_multihop_datasets --dataset all
```

각각 `data/benchmarks/musique`, `data/benchmarks/2wikimultihopqa`에 저장됩니다. MuSiQue 정책 실험에는
골드 supporting paragraph가 있는 `musique_ans_v1.0_train/dev.jsonl`을 사용하고, 2Wiki는 `train/dev.json`을
사용합니다. test split은 골드 근거가 없으므로 로컬 정책 효용 평가에 사용하지 않습니다.

세 데이터셋을 동일한 검색 정책 코드에서 사용하려면 먼저 공통 스키마로 전처리합니다. 기본 명령은
HotpotQA distractor, MuSiQue-Ans, 2Wiki의 train과 validation(dev) 분할을 모두 변환합니다.

```bash
.venv/bin/python -m experiments.preprocess_multihop \
  --dataset all \
  --split all \
  --output-root data/benchmarks/processed
```

전체 실행 전에 데이터셋마다 100개씩 구조와 로그를 확인하려면 별도 출력 폴더에서 다음처럼 실행합니다.
일부 문항만 지정하면 원본 앞부분이 아니라 전체 split에서 `--seed` 기준으로 균일 표본을 추출합니다.

```bash
.venv/bin/python -m experiments.preprocess_multihop \
  --dataset all \
  --split all \
  --max-examples 100 \
  --output-root data/benchmarks/processed_pilot
```

출력은 `dataset/split.jsonl`과 같은 이름의 manifest로 구성됩니다. 공통 레코드는 질문·정답·hop·질문 유형,
원문 문서와 문장, 정답 문서 후보군, 정답 문장 번호, 데이터셋별 추론 경로를 보존합니다. `doc_id`는 질문 안의
문서 위치를 정확하게 나타내고, `content_id`는 이후 pooled corpus에서 동일 내용을 안전하게 중복 제거할 때
사용합니다. MuSiQue는 동일 제목의 서로 다른 문단이 많으므로 제목 대신 `paragraph_idx`로 정답을 연결합니다.
2Wiki 관계 트리플과 MuSiQue 질문 분해는 감독·평가 필드일 뿐 평가 인덱스의 그래프 간선으로 사용하지 않습니다.
원본에 범위를 벗어난 supporting sentence 번호가 있으면 문서 정답은 유지하고 해당 번호를
`invalid_sentence_ids`로 격리하며, manifest에 발견 개수를 기록합니다.

### 세 데이터셋 통합 정책 학습

공통 전처리가 끝나면 데이터셋마다 2,000문항을 균일 표본 추출해 하나의 검색 행동 정책을 학습할 수 있습니다.
각 데이터셋은 독립된 검색 인덱스를 사용하고, 행동별 효용이 기록된 정책 행만 합칩니다. 따라서 한 데이터셋의
문서가 다른 데이터셋의 검색 난이도나 그래프를 바꾸지 않습니다.

기본 효용은 `0.7 × All-Support Recall + 0.3 × 문서 Recall - 추가 reranking 비용`입니다. 2Wiki처럼
네 개의 정답 문서가 필요한 질문에서 일부 근거만 찾은 행동이 과대평가되지 않게 합니다. 2Wiki의 데이터량이
정책을 지배하지 않도록 데이터셋별 학습 가중치 합은 동일하게 맞추며, 그래프 행동이 Vector보다 이긴 희소 행은
추가 가중합니다. 행동 라벨은 데이터셋별 JSONL에 질문마다 즉시 저장되므로 중단 후 같은 명령을 실행하면 이어서
진행합니다. 검색 설정이나 표본이 달라지면 안전하게 중단하며, 의도적으로 다시 만들 때만 `--relabel`을 사용합니다.

이 명령의 내부 개발 분할은 학습 상태 확인용입니다. 논문 최종 성능은 train으로 학습한 모델을 공식 validation에
별도로 적용해 Vector-only·규칙 정책·고정 Graph·Oracle과 비교해야 합니다.

인덱스는 `data/index/`에 유지됩니다. 자료를 바꾸지 않았다면 앱을 다시 실행해도 재인덱싱하지 않습니다.
파일명·크기·수정 시각, 청크 설정, 임베딩 모델을 기록한 manifest로 자료 변경을 감지합니다.
