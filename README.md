# NOVA UROP — 적응형 검색 제어와 학습 지원 MVP

자격증·NCS 학습을 지원하는 **로컬 RAG 웹 앱**과, 다중 홉 질의응답에서
**추가 검색의 답변 이득을 예측하는 학습 정책**의 연구 코드·원고·결과를 함께 보관합니다.

> 먼저 구분하세요: 앱의 하이브리드·청크 그래프 라우팅과 논문의 105차원
> 단계별 검색 정책은 서로 다른 경로입니다. 논문 정책이 현재 앱에 그대로
> 배포되어 있다고 해석하지 마세요. 이전 실험도 삭제하지 않고 보존했습니다.

## 처음 보는 사람을 위한 안내

| 목적 | 읽을 곳 |
|---|---|
| 웹 앱 실행 | 아래 **MVP 빠른 시작** |
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

## MVP 빠른 시작

Python 3.12 가상환경과 별도 설치된 Ollama를 사용합니다.

```bash
git clone https://github.com/iion74/UROP.git
cd UROP
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen3:8b
streamlit run UROP.py
```

Ollama 서버가 실행 중이어야 합니다. 기본 주소는 `http://127.0.0.1:11434`입니다.
최초 임베딩·재정렬 모델 다운로드에는 네트워크가 필요합니다.
실행은 GPU 또는 CPU 환경과 모델 크기에 따라 속도·메모리 요구가 달라집니다.

재배포 권한이 있는 PDF·TXT·MD·CSV 학습 자료를 `data/`에 넣고,
사이드바의 **자료 읽기 및 인덱스 생성**을 누릅니다.
자료가 바뀌면 인덱스를 갱신해야 하며, 변경이 없으면 기존 인덱스를 재사용합니다.
실험용 원문 자료와 학습용 문서를 같은 데이터 폴더에 넣으면 함께 검색될 수
있으므로 MVP와 실험은 별도 작업 복사본을 권장합니다.

MVP는 BM25·벡터 하이브리드 검색, 경량 청크 그래프, 근거 기반 답변·퀴즈,
학습 기록을 제공합니다. [기존 MVP 상세 설명](docs/LEGACY_MVP_GUIDE.md)은
개발 이력으로 보존했으며 최종 논문 정책 설명과 구분합니다.

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
