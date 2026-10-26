# 결과 확인과 재현

프로젝트 루트에서 Python 3.12 가상환경을 사용하는 것을 권장한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-research.txt
```

`requirements-research-snapshot.txt`는 이 공개 정리 시점의 주요 패키지
설치 버전이다. 실험 당시 모든 의존성을 고정한 lockfile은 아니며,
Ollama 모델의 동일성은 각 원본 manifest와 보고서를 확인한다.

## 1. 보고된 결과 확인: GPU·생성 모델 불필요

```bash
python -m experiments.analyze_sequential_answer_gain_multiseed \
  --reports data/benchmarks/multihop/context_reader_aware/analysis/sequential_multiseed/policy_seed{42,43,44,45,46}.json \
  --output /tmp/urop-five-seed-summary.json
python -m experiments.export_paper_seed_confidence_intervals
python scripts/plot_kci_results.py
```

이는 저장된 보고서 집계와 그림 생성이지 독립적인 신규 실험이 아니다.

## 2. 공개 수치형 기록으로 정책 선택 재생: CPU만 사용

[Release](https://github.com/iion74/UROP/releases/tag/workspace-2026-10-26)의
`urop-numeric-replay.tar.gz`를 내려받아 프로젝트 루트에 해제한다.
압축 안에는 질문 ID, 105차원 상태, 단계별 EM/F1·토큰·시간이 들어 있으며,
원문 질문·문서 전문·정답 문자열은 포함하지 않는다.

```bash
gh release download workspace-2026-10-26 --repo iion74/UROP \
  --pattern urop-numeric-replay.tar.gz --dir /tmp/urop-release
tar -xzf /tmp/urop-release/urop-numeric-replay.tar.gz -C .
python scripts/replay_published_policy.py --seed 42
```

이 재생은 저장된 모델·상태로 최종 선택과 평균 점수를 재계산한다.
부트스트랩을 새로 수행하거나 온라인 지연시간을 측정하지 않는다.
출력은 원본 보고서와 수치 비교하며 정책 성능이 달라지면 실패한다.

## 3. 원문 데이터부터 궤적 재생성: 다운로드와 모델 실행 필요

다음은 시간이 오래 걸릴 수 있고 생성 단계에서 GPU를 사용할 수 있다.
저장소를 정리하거나 결과를 읽기 위해 실행할 필요는 없다.

```bash
python -m experiments.download_hotpotqa --output-dir data/benchmarks/hotpotqa/distractor
python -m experiments.download_multihop_datasets --dataset all
python -m experiments.preprocess_multihop --dataset all --split all
ollama pull qwen3:8b
python -m experiments.evaluate_context_reader_aware_prefixes \
  --datasets hotpotqa musique 2wiki --split validation \
  --max-queries-per-dataset 0 \
  --output-root data/benchmarks/multihop/context_reader_aware/full_validation
python -m experiments.run_sequential_answer_gain_multiseed
```

마지막 명령은 시드 42~46을 순차 실행하고, 부족한 train 궤적을 생성한 뒤
정책을 학습·평가한다. 완료된 원본 JSON을 보존하려면 새 작업 복사본에서
수행한다. 초기 다운로드·궤적 생성은 네트워크와 로컬 모델이 필요하다.

## 해석상 주의

- train은 데이터셋별 시드별 2,000문항 재추출, validation은 전체다.
- 최초 검색 포함 총 검색 1~4회이며 gold hop 예측이 아니다.
- 단위는 F1·EM 100점, `±`는 5개 시드의 표본 표준편차다.
- 지연시간은 저장 기록 기반 추정값이다. 토큰은 답변 출력만 센다.
- 고정 4회 대비 모든 시드에서 유의하게 우수한 것은 아니다.
- 공개 snapshot으로 점수를 재생하는 것과 새로운 모델 호출로 원래
  생성 답변을 완전히 재현하는 것은 다르다.
