# 실험 코드 안내

## 최종 논문 경로

최종 방법은 `sequential_best_future_f1_gain_hgb`다. 아래 여섯 개가 주 경로다.

| 파일 | 역할 |
|---|---|
| `evaluate_context_reader_aware_prefixes.py` | 질문별 공식 후보 문서에서 단계별 기록 생성 |
| `evaluate_sequential_answer_gain_policy.py` | 105차원 이득 회귀 정책 학습·평가 |
| `run_sequential_answer_gain_multiseed.py` | 5개 표본 시드 순차 실행 |
| `analyze_sequential_answer_gain_multiseed.py` | 시드·문항 수 감사와 집계 |
| `export_paper_seed_confidence_intervals.py` | 기존 결과의 신뢰구간 증빙 추출 |
| `scripts/replay_published_policy.py` | 공개 수치형 입력으로 정책 점수 재생 |

## 전체 실험 모듈

원래 파일과 이력을 보존한다. 아래 목록의 이전 실험은 최종 비교군이나 채택된 방법을 뜻하지 않는다.

| 모듈 | 파일에 기록된 설명 |
|---|---|
| [analyze_causal_evidence_reader.py](../experiments/analyze_causal_evidence_reader.py) | Summarize paired causal-evidence reader results without pooling datasets. |
| [analyze_context_reader_aware_multiseed.py](../experiments/analyze_context_reader_aware_multiseed.py) | Aggregate full-validation reader-aware policies across training seeds. |
| [analyze_dual_value_full_system.py](../experiments/analyze_dual_value_full_system.py) | Materialise and score the complete dual-value adaptive RAG system. |
| [analyze_dual_value_reasoning_replication.py](../experiments/analyze_dual_value_reasoning_replication.py) | Combine development and disjoint holdout adaptive-reasoning pilots. |
| [analyze_dual_value_router_multiseed.py](../experiments/analyze_dual_value_router_multiseed.py) | Aggregate dual-value router runs without selecting a favourable seed. |
| [analyze_kci_required_baselines.py](../experiments/analyze_kci_required_baselines.py) | Evaluate fixed-depth and equal-search-budget baselines on cached prefixes. |
| [analyze_reader_aware_sufficiency.py](../experiments/analyze_reader_aware_sufficiency.py) | Paired analysis for the frozen reader-aware sufficiency candidate. |
| [analyze_repeated_seed_validation.py](../experiments/analyze_repeated_seed_validation.py) | Aggregate repeated 1,000-question validation cohorts from a frozen 2,000 pool. |
| [analyze_sequential_answer_gain_multiseed.py](../experiments/analyze_sequential_answer_gain_multiseed.py) | Audit and aggregate independent train-sample-seed policy evaluations. |
| [analyze_student_policy_multiseed.py](../experiments/analyze_student_policy_multiseed.py) | Aggregate reader-free or single-reader student policies across seeds. |
| [audit_query_memory_recovery.py](../experiments/audit_query_memory_recovery.py) | Audit whether persistent query memory recovers missing proof documents. |
| [build_sequential_answer_states.py](../experiments/build_sequential_answer_states.py) | Build fair sequential-policy states from cached one-shot Answer Oracle rows. |
| [compare_policy_predictions.py](../experiments/compare_policy_predictions.py) | Paired, per-dataset comparison of two adaptive STOP policies. |
| [download_hotpotqa.py](../experiments/download_hotpotqa.py) | Download and persist an official HotpotQA configuration for local experiments. |
| [download_multihop_datasets.py](../experiments/download_multihop_datasets.py) | Download official MuSiQue and 2WikiMultiHopQA benchmark releases. |
| [evaluate_adaptive_proof_reader.py](../experiments/evaluate_adaptive_proof_reader.py) | Evaluate adaptive proof-completeness retrieval with the frozen reader. |
| [evaluate_answer_gain_search_policy.py](../experiments/evaluate_answer_gain_search_policy.py) | Test whether learning *answer* gain improves the third-search decision. |
| [evaluate_answer_gain_three_way_holdout.py](../experiments/evaluate_answer_gain_three_way_holdout.py) | Evaluate frozen answer-gain SEARCH heads with the existing REASON head. |
| [evaluate_answer_oracle.py](../experiments/evaluate_answer_oracle.py) | Measure answer-level retrieval-action potential with one frozen QA reader. |
| [evaluate_bridge_answer_oracle.py](../experiments/evaluate_bridge_answer_oracle.py) | Answer-F1 pilot for evidence-conditioned bridge retrieval actions. |
| [evaluate_bridge_policy.py](../experiments/evaluate_bridge_policy.py) | Evaluate a frozen evidence-conditioned Bridge policy on an unseen split. |
| [evaluate_bridge_retrieval.py](../experiments/evaluate_bridge_retrieval.py) | Cheap gold-evidence screen for evidence-conditioned bridge actions. |
| [evaluate_causal_evidence_reader.py](../experiments/evaluate_causal_evidence_reader.py) | Evaluate a learned compact evidence bottleneck with the frozen QA reader. |
| [evaluate_context_answer_oracle.py](../experiments/evaluate_context_answer_oracle.py) | Create answer-level action labels in each benchmark's official context. |
| [evaluate_context_reader_aware_prefixes.py](../experiments/evaluate_context_reader_aware_prefixes.py) | Generate reader-aware search prefixes in each question's official context. |
| [evaluate_dual_value_choice_judge.py](../experiments/evaluate_dual_value_choice_judge.py) | Arbitrate baseline and adaptive-thinking answers using evidence only. |
| [evaluate_dual_value_reasoning_pilot.py](../experiments/evaluate_dual_value_reasoning_pilot.py) | Evaluate adaptive reasoning compute on the dual-value router's REASON queue. |
| [evaluate_frozen_action_gain_full.py](../experiments/evaluate_frozen_action_gain_full.py) | Evaluate frozen, directly learned action gains on all validation questions. |
| [evaluate_frozen_dual_value_router.py](../experiments/evaluate_frozen_dual_value_router.py) | Apply saved five-seed dual-value router to a disjoint cached prefix set. |
| [evaluate_gated_policy.py](../experiments/evaluate_gated_policy.py) | Evaluate frozen precision-gated one-shot retrieval policies. |
| [evaluate_graph_repair_reader.py](../experiments/evaluate_graph_repair_reader.py) | Evaluate local graph proof repair against the frozen topology reader. |
| [evaluate_grounded_answer_verifier.py](../experiments/evaluate_grounded_answer_verifier.py) | Evaluate one uniform conservative draft verifier on every QA dataset. |
| [evaluate_iterative_hop_retrieval.py](../experiments/evaluate_iterative_hop_retrieval.py) | Evaluate arbitrary-length STOP/CONTINUE search trajectories without a reader. |
| [evaluate_llm_stop_policy.py](../experiments/evaluate_llm_stop_policy.py) | Evaluate a frozen LLM self-judgment STOP/CONTINUE baseline. |
| [evaluate_proof_program_gold.py](../experiments/evaluate_proof_program_gold.py) | Pilot executable proof programs on exactly the gold-support reader context. |
| [evaluate_reader_aware_proof_prefixes.py](../experiments/evaluate_reader_aware_proof_prefixes.py) | Evaluate every compact-proof prefix with an explicit evidence-chain reader. |
| [evaluate_reader_routed_policy.py](../experiments/evaluate_reader_routed_policy.py) | Merge a frozen STOP policy with a validation-gated reader prompt. |
| [evaluate_reason_gain_gate.py](../experiments/evaluate_reason_gain_gate.py) | Gate existing REASON routes by predicted *actual* answer-F1 benefit. |
| [evaluate_reason_gain_gate_reverse.py](../experiments/evaluate_reason_gain_gate_reverse.py) | Cross-split check: train REASON benefit on all-reason holdout, test full validation. |
| [evaluate_rule_stop_policy.py](../experiments/evaluate_rule_stop_policy.py) | Calibrate and evaluate a transparent retrieval-score STOP rule. |
| [evaluate_sequential_answer_gain_policy.py](../experiments/evaluate_sequential_answer_gain_policy.py) | Train and evaluate a gold-free, every-prefix STOP/SEARCH policy. |
| [evaluate_stop_answer_oracle.py](../experiments/evaluate_stop_answer_oracle.py) | Score every 4-hop STOP prefix with one frozen QA reader. |
| [evaluate_stop_value_ensemble.py](../experiments/evaluate_stop_value_ensemble.py) | Calibrate and evaluate a seed-vote ensemble of STOP value policies. |
| [evaluate_stop_value_policy.py](../experiments/evaluate_stop_value_policy.py) | Evaluate a frozen STOP/CONTINUE value policy on held-out trajectories. |
| [export_paper_seed_confidence_intervals.py](../experiments/export_paper_seed_confidence_intervals.py) | Export previously computed seed-wise confidence intervals for the paper. |
| [materialize_adaptive_proof_predictions.py](../experiments/materialize_adaptive_proof_predictions.py) | Materialize evidence for a new threshold from cached prefix probabilities. |
| [preprocess_multihop.py](../experiments/preprocess_multihop.py) | Preprocess HotpotQA, MuSiQue, and 2Wiki into one validated schema. |
| [run_kci_disjoint_holdout.py](../experiments/run_kci_disjoint_holdout.py) | Finish the three-dataset, frozen-policy holdout sequentially and unload Qwen. |
| [run_kci_three_way_counterfactual.py](../experiments/run_kci_three_way_counterfactual.py) | Evaluate a uniform SEARCH/REASON/STOP control at the learned policy's budgets. |
| [run_sequential_answer_gain_multiseed.py](../experiments/run_sequential_answer_gain_multiseed.py) | Run five independent train-sample-seed policy experiments sequentially. |
| [train_adaptive_proof_completion_policy.py](../experiments/train_adaptive_proof_completion_policy.py) | Train an adaptive proof-completeness policy over retrieval prefixes. |
| [train_answer_policy.py](../experiments/train_answer_policy.py) | Train and audit one-shot versus sequential answer-aware retrieval policies. |
| [train_bridge_policy.py](../experiments/train_bridge_policy.py) | Train a pilot policy over Direct, legacy Graph, and Bridge-path actions. |
| [train_counterfactual_proof_policy.py](../experiments/train_counterfactual_proof_policy.py) | Train and audit a counterfactual proof-completion evidence policy. |
| [train_dual_value_router.py](../experiments/train_dual_value_router.py) | Train a dual-value SEARCH/REASON/STOP router on cached prefix trajectories. |
| [train_evidence_intervention_router.py](../experiments/train_evidence_intervention_router.py) | Route between Direct RAG and proof-repaired evidence before generation. |
| [train_evidence_width_policy.py](../experiments/train_evidence_width_policy.py) | Learn dataset-blind proof width from question and evidence-score structure. |
| [train_factorized_graph_repair_policy.py](../experiments/train_factorized_graph_repair_policy.py) | Train separate WHERE-to-repair and WHETHER-to-repair policies. |
| [train_full_gated_policy.py](../experiments/train_full_gated_policy.py) | Train the final one-shot retrieval policy from all benchmark train rows. |
| [train_gated_answer_policy.py](../experiments/train_gated_answer_policy.py) | Train a precision-gated answer-aware one-shot/sequential retrieval policy. |
| [train_graph_repair_policy.py](../experiments/train_graph_repair_policy.py) | Train a local graph-edge policy to repair missing multi-hop evidence. |
| [train_hotpot_policy.py](../experiments/train_hotpot_policy.py) | Train a retrieval-action policy from an official HotpotQA JSON split. |
| [train_multihop_policy.py](../experiments/train_multihop_policy.py) | Train one adaptive retrieval policy from three multi-hop QA benchmarks. |
| [train_reader_aware_sufficiency_policy.py](../experiments/train_reader_aware_sufficiency_policy.py) | Train a reader-aware, gold-free-at-inference proof stopping policy. |
| [train_reader_free_student_policy.py](../experiments/train_reader_free_student_policy.py) | Train a reader-free student that chooses between search prefixes 2 and 3. |
| [train_stable_stop_ensemble.py](../experiments/train_stable_stop_ensemble.py) | Train a stable tree ensemble for rare next-search gains. |
| [train_stop_policy.py](../experiments/train_stop_policy.py) | Train and nested-OOF evaluate a 4-hop Answer-F1 STOP/CONTINUE policy. |
| [train_stop_value_policy.py](../experiments/train_stop_value_policy.py) | Train a cost-aware long-term value policy for adaptive retrieval stopping. |
