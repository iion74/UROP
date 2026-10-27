# 논문과 연구 기록

## 우선 읽을 파일

- [최종 원고 HWP](<다중 홉 질의응답을 위한 답변 이득 예측 기반 단계별 검색 제어_이정호.hwp>):
  「다중 홉 질의응답을 위한 답변 이득 예측 기반 단계별 검색 제어」. 게재 확정본을 뜻하지 않는다.
- [NOVA UROP 계획서 HWP](<2026년 NOVA UROP 계획서_이정호.hwp>) ·
  [PDF](<2026년 NOVA UROP 계획서_이정호.pdf>): 초기 연구 계획.
- [최종 연구 방법](../docs/RESEARCH_METHOD.md): 현재 기준으로 읽어야 할 기술 설명.
- [실험 재현](../docs/REPRODUCIBILITY.md): 결과 확인·재생·재학습의 차이.

## 그림과 증빙

- [figures/](figures/): PNG·PDF·SVG. 그림 파일과 원고 삽입 버전은 다를 수 있다.
- [시드별 신뢰구간](evidence/seed_confidence_intervals.md)
- [정밀도 유지 CSV](evidence/seed_confidence_intervals.csv)
- [원본 보고서 SHA-256](evidence/seed_confidence_intervals.sources.json)

## 결과 수치의 기준

원고의 표·그림 수치는 [최종 5개 시드 보고서](../data/benchmarks/multihop/context_reader_aware/analysis/sequential_multiseed/)와
[집계 JSON](../data/benchmarks/multihop/context_reader_aware/analysis/sequential_answer_gain_multiseed.json)을
기준으로 한다. 검색 횟수는 최초 검색을 포함한 총 1~4회다.
