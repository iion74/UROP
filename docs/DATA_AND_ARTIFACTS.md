# 자료·모델·공개 범위

이 저장소는 공개 저장소다. 코드와 함께 소유자가 공개 요청한 원고·그림·
검토 메모·실험 보고서·최종 정책 모델을 포함한다. 문서는 연구 진행
기록이며 게재·심사 완료를 뜻하지 않는다.

## 저장 위치

| 자료 | 위치 |
|---|---|
| 원고 HWP/PDF와 계획서 | 프로젝트 루트, [논문 안내](../paper/README.md) |
| 그림·시드별 신뢰구간 | [paper](../paper/) |
| 최종 5개 시드 보고서 | [sequential_multiseed](../data/benchmarks/multihop/context_reader_aware/analysis/sequential_multiseed/) |
| 최종 집계 JSON | [sequential_answer_gain_multiseed.json](../data/benchmarks/multihop/context_reader_aware/analysis/sequential_answer_gain_multiseed.json) |
| 최종 5개 정책 | [sequential_answer_gain_multiseed](../data/models/sequential_answer_gain_multiseed/) |
| 대용량 정책 모델·수치형 재생 입력 | [GitHub Releases](https://github.com/iion74/UROP/releases/tag/workspace-2026-10-26) |
| 파일별 공개 위치·용량 | [workspace_inventory.csv](workspace_inventory.csv) |

파일을 보기 좋게 정리하되 기존 코드가 참조하는 경로는 바꾸지 않았다.
Release 압축파일도 기존 상대경로를 유지한다. `artifacts/`는 로컬 생성
폴더이며 Git에는 포함하지 않는다.

## 제외한 것

- `.env`, 인증정보, `.git` 내부 정보, 개인 학습기록 SQLite.
- 가상환경, 파이썬 캐시, 임베딩 캐시, 재생성 가능한 검색 인덱스.
- 외부 벤치마크 원본과 전처리 문서 전문: 공식 경로에서 내려받는다.
- NCS 원본 PDF: 프로젝트 코드의 저작물이 아니며 재배포 조건을 별도로
  확인해야 하므로 업로드하지 않는다. MVP 실행자는 사용 권한이 있는 자료를 넣는다.
- 문서 전문을 포함하는 중간 검색 기록: 그대로 재배포하는 대신 최종
  연구의 수치형 특징·점수·비용 기록을 Release로 제공한다.

일반 Git에는 100 MiB 초과 파일을 넣을 수 없다. 이 작업은 Git LFS
유료 저장소를 설정하거나 유료 서비스를 구매하지 않는다.
[GitHub 크기 안내](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

## 외부 데이터 출처

- [HotpotQA](https://hotpotqa.github.io/): CC BY-SA 4.0.
- [MuSiQue](https://github.com/StonyBrookNLP/musique): CC BY 4.0.
- [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop): 원저자 배포본과
  사용 조건을 따른다. 저장소 코드의 Apache-2.0을 모든 원문 데이터의
  라이선스로 확대 해석하지 않는다.

원본과 전처리 문서의 권리는 원저작자에게 있다. 이 프로젝트에 별도의
오픈소스 라이선스를 임의로 부여하지 않았다. 코드·모델·원고의 재사용
허락과 데이터별 조건은 각각 확인해야 한다.

## 모델 안전성

`joblib`/pickle 및 일부 PyTorch 파일은 로드 시 코드를 실행할 수 있다.
신뢰하는 출처의 파일만 사용하고 Release SHA-256을 확인한다.
수치형 재생 NPZ는 `allow_pickle=False`로 읽도록 구성한다.
