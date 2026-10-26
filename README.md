# NOVA UROP — 자격증 학습량 AI 분석 도구 (NOVA Study Map)

NCS 자격증 수험자의 학습 메모·질문·퀴즈를 NCS 지식 지도에 투사해
**실제로 확보한 지식량**을 측정하는 **로컬 웹 앱**과, 그 AI 학습 도우미의 검색 엔진으로
들어간 논문 「다중 홉 질의응답을 위한 답변 이득 예측 기반 단계별 검색 제어」의
연구 코드·원고·결과를 함께 보관합니다.

> 논문의 105차원 STOP/SEARCH 정책은 AI 학습 도우미의 검색 제어로 탑재되어 있습니다.
> 다만 정책은 영어 다중 홉 벤치마크로 학습·검증되었고, 한국어 NCS 자료에서의 성능은
> 따로 검증하지 않았습니다. 일부 어휘 특징은 영어 토큰 기준이라 한국어 질문에서는 0이 됩니다.
> 실제 NCS 9개 모듈로 확인한 질문 7개에서는 질문-단원 매핑이 7개 모두 정확했지만, 정책은
> 6개 질문에서 최대 4회까지 검색했습니다(평균 3.9회). 영어 벤치마크에서 보정한 임곗값이
> 한국어 상태에서는 SEARCH 쪽으로 치우치므로, 검색 감소 효과는 이 앱에서 기대하지 않습니다.

## 처음 보는 사람을 위한 안내

| 목적 | 읽을 곳 |
|---|---|
| 웹 앱 기능·실행 | 아래 **학습량 분석 웹 앱** |
| 최종 정책의 구조·학습·105차원 입력 | [연구 방법](docs/RESEARCH_METHOD.md) |
| 결과 확인·오프라인 재생·전체 재생성 | [재현 가이드](docs/REPRODUCIBILITY.md) |
| 코드별 역할과 과거 실험 구분 | [실험 코드 목록](docs/EXPERIMENT_CATALOG.md) |
| 논문 HWP/PDF·그림·신뢰구간 | [논문 안내](paper/README.md) |
| 전체 파일 목록·공개/제외 사유 | [자료 안내](docs/DATA_AND_ARTIFACTS.md) |
| 수정 후 GitHub 반영 | [동기화 안내](docs/GITHUB_SYNC.md) |

## 최종 연구: 무엇을 학습했나

Qwen3-8B는 고정하고 **히스토그램 기반 그래디언트 부스팅 회귀 모델**만 학습합니다.
필수 최초 검색 이후 현재 질문·검색 점수·근거 변화·답변 상태를 105차원으로
구성하고, 후속 단계의 최대 F1에서 현재 F1을 뺀 이득을 예측합니다.
단계별 임곗값으로 STOP/SEARCH를 결정하며, 최초 검색을 포함해 최대 4회 검색합니다.

```text
질문 → 최초 검색 → 현재 답변 생성 → 상태 특징 → 이득 예측
                      ↑                         │
                      └── 질의 갱신·추가 검색 ← SEARCH
                                                └─ STOP → 현재 답변 반환
```

- 시드 42~46마다 각 데이터셋 train 2,000문항 재추출: 시드당 총 6,000질문.
- 질문당 3개 상태: 시드당 18,000개 학습 쌍.
- 질문 단위 3분할 OOF로 단계별 임곗값 보정.
- 전체 validation: HotpotQA 7,405 / MuSiQue-Ans 2,417 / 2Wiki 12,576문항.
- 고정 1~4회 및 검색 깊이 분포 일치 무작위 종료와 비교.
- 평가는 저장된 궤적 재생이며 온라인 종단 간 실행 측정이 아닙니다.

## 최종 5개 학습 표본 시드 결과

F1·EM은 100점 기준, `±`는 시드 간 표본 표준편차입니다.
검색 횟수에는 최초 검색을 포함합니다.

| 데이터셋 | 고정 4회 F1 | 정책 F1 | 정책 EM | 평균 검색 | 고정 4회 대비 검색 감소 |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 71.78 | 72.05 ± 0.14 | 58.01 ± 0.13 | 2.90 ± 0.51 | 27.40 ± 12.76% |
| MuSiQue | 44.49 | 44.97 ± 0.48 | 35.18 ± 0.51 | 3.17 ± 0.34 | 20.79 ± 8.59% |
| 2WikiMultiHopQA | 64.01 | 64.81 ± 0.48 | 58.01 ± 0.46 | 2.84 ± 0.36 | 28.99 ± 9.05% |

동일 검색 깊이 분포의 무작위 종료 기대 F1 대비 차이는 각각
**+0.55 / +1.36 / +1.63점**입니다.

**한계와 비용도 함께 읽어야 합니다.** 고정 4회 대비 모든 시드에서 통계적
우위가 확인된 것은 아닙니다. 중간 답변 생성 때문에 답변 출력 토큰은
177.49 / 213.98 / 178.91%, 추정 지연시간은 25.29 / 33.36 / 31.41%
증가했습니다. 입력·질의 갱신 토큰은 이 출력 토큰 수에 포함하지 않습니다.
질문별 제한된 후보 문서 환경이고 validation은 개발 중에도 사용했습니다.
따라서 검색 배분의 효과이지 전체 비용 절감이나 독립 test에서의 보편적 우위 주장이 아닙니다.

원본: [5개 시드 보고서](data/benchmarks/multihop/context_reader_aware/analysis/sequential_multiseed/),
[집계 JSON](data/benchmarks/multihop/context_reader_aware/analysis/sequential_answer_gain_multiseed.json),
[시드별 신뢰구간](paper/evidence/seed_confidence_intervals.md).

## 학습량 분석 웹 앱

NCS 인공지능 플랫폼 구축(LM2001070101~109) 9개 모듈·63개 세부 단원을
표준 지식 그래프로 삼습니다. 모든 처리는 로컬(Ollama·GPU)에서 끝나며
학습 기록은 `data/index/study_progress.sqlite3`에만 저장됩니다.

| 화면 | 기능 |
|---|---|
| 대시보드 | 지식 커버리지·의미적 학습 밀도·시험일 유효 지식량·지식 연결성, 모듈별 숙련도, AI 학습 페이스메이커 |
| 학습 기록 | 단원·상태·시간·메모 기록. 메모와 NCS 원문의 의미 일치도를 즉시 표시 |
| 지식 맵 | 모듈 선수관계 그래프 위에 단원별 숙련도·기억 유지 상태를 색으로 투사, 단원별 지표 표 |
| AI 학습 도우미 | 논문 정책이 검색을 더 할지 단계마다 판단(검색 1~4회)하고, 그 근거 안에서만 한국어로 답변 |
| 확인 퀴즈 | 단원 원문 근거로 객관식 3문항 생성·채점, 결과를 숙련도와 망각 예측에 반영 |

**분석 지표 계산** ([learning_analytics.py](src/learning_analytics.py))

- 지식 커버리지: 학습 기록·질문·퀴즈가 닿은 단원 / 전체 단원. 질문은 임베딩 유사도로 단원에 매핑합니다.
- 의미적 학습 밀도: 학습 메모와 해당 단원 NCS 원문 구절의 cross-encoder 관련도(0~1) 평균.
- 숙련도: 학습 상태 0.5 · 학습 밀도 0.3 · 퀴즈 정답률 0.2의 가중 평균(없는 항목은 제외).
- 기억 유지율: `exp(-경과일 / 안정도)`. 안정도는 1.5일에서 시작해 복습한 날마다 두 배,
  퀴즈 정답률에 비례해 커지는 간이 망각 곡선이며, 시험일까지 연장해 유효 지식량을 계산합니다.
- AI 페이스메이커: 정답률·학습 밀도·유지율이 낮거나 선수 모듈이 비어 있는 단원을 찾아
  요약 생성·퀴즈·학습 기록으로 바로 연결합니다.

### 실행 (Windows)

Python 3.12 이상, [Ollama](https://ollama.com), NVIDIA GPU(권장)를 사용합니다.

```powershell
git clone https://github.com/iion74/UROP.git
cd UROP
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
ollama pull qwen3:8b
.venv\Scripts\streamlit run UROP.py
```

브라우저에서 `http://localhost:8501`이 열립니다. 같은 네트워크의 다른 기기에서 보려면
`streamlit run UROP.py --server.address 0.0.0.0`으로 실행합니다.
최초 실행 시 임베딩·재정렬 모델(약 3GB)을 내려받습니다.

**NCS 학습모듈 준비:** [NCS 학습모듈 파일검색](https://www.ncs.go.kr/unity/th03/ncsModuleFileSearch.do)에서
세분류 `20010701`(인공지능플랫폼구축)의 학습모듈 9개 PDF를 내려받아 `data/` 아래 아무 폴더에 넣고,
사이드바의 **자료 읽기 및 인덱스 생성**을 누릅니다. 파일명에 `LM2001070101` 같은 모듈 코드가
있어야 단원에 연결됩니다. NCS 원본은 재배포하지 않으므로 저장소에 포함하지 않습니다.

`SEARCH_CONTROL=graph`로 바꾸면 이전 그래프 라우팅으로 답합니다.
[기존 MVP 상세 설명](docs/LEGACY_MVP_GUIDE.md)은 개발 이력으로 보존했습니다.

## 저장소 구조

```text
UROP.py                 Streamlit 앱 진입점
src/                    검색·생성·상태 특징·평가 공통 모듈
experiments/            최종 정책과 이전 탐색 실험
scripts/                그림·동기화·공개 자료 생성·재생
tests/                  단위 테스트
docs/                   방법·재현·자료·실험 목록
docs/history/           이전 단계 실험 계획·보고서
paper/                  최종 원고 HWP·계획서·그림·신뢰구간 증빙
data/benchmarks/...     검토한 요약 JSON·manifest
data/models/...         최종 5개 경량 정책 모델
```

`docs/history/`의 과거 보고서는 파일명에 FINAL이 있어도 현재 최종 실험이
아닐 수 있으므로 위의 최종 JSON을 기준으로 읽으세요.

## 대용량 자료와 검증

[공개 snapshot Release](https://github.com/iion74/UROP/releases/tag/workspace-2026-10-26)에
정책 모델 묶음과 문서 전문을 제외한 수치형 재생 자료를 제공합니다.
원본 벤치마크와 NCS 자료는 원저작자 배포 조건을 따르며 저장소에 그대로 올리지 않습니다.
인증정보·가상환경·개인 학습기록·재생성 가능한 캐시도 제외합니다.
파일별 구분은 [전체 목록](docs/workspace_inventory.csv)을 확인하세요.

```bash
pip install -r requirements-research.txt
python -m pytest -q
```

이 저장소 공개는 논문 게재나 소프트웨어 사용권 자동 부여를 의미하지 않습니다.
코드·모델·원고와 외부 데이터의 권리 및 재사용 조건을 각각 확인하세요.
