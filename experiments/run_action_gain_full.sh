#!/usr/bin/env bash
set -euo pipefail

cd /home/lys/Desktop/UROP
python_bin=/home/lys/Desktop/UROP/.venv/bin/python
root=data/benchmarks/multihop/context_reader_aware
analysis_root="$root/analysis"
exec > >(tee -a "$analysis_root/action_gain_full_run.log") 2>&1

# Ollama had no loaded model before this run. Release our Qwen session whether
# the sequential pipeline succeeds or stops on an error.
trap 'ollama stop qwen3:8b || true' EXIT

echo "[$(date --iso-8601=seconds)] Stage 1/3: full REASON answer generation"
"$python_bin" -u -m experiments.evaluate_dual_value_reasoning_pilot \
  --split validation \
  --prefix-root "$root/full_validation" \
  --all-questions --rows-per-dataset 0 \
  --datasets hotpotqa musique 2wiki \
  --output-root "$analysis_root/action_gain_full_validation_reason" \
  --ollama-keep-alive 30m --log-every 100

echo "[$(date --iso-8601=seconds)] Stage 2/3: full REASON answer arbitration"
"$python_bin" -u -m experiments.evaluate_dual_value_choice_judge \
  --split validation \
  --pilot-root "$analysis_root/action_gain_full_validation_reason" \
  --prefix-root "$root/full_validation" \
  --datasets hotpotqa musique 2wiki \
  --output-root "$analysis_root/action_gain_full_validation_judge" \
  --ollama-keep-alive 30m --skip-identical --log-every 100

echo "[$(date --iso-8601=seconds)] Stage 3/3: frozen-policy and baseline evaluation"
"$python_bin" -u -m experiments.evaluate_frozen_action_gain_full \
  --prefix-root "$root/full_validation" \
  --judge-root "$analysis_root/action_gain_full_validation_judge" \
  --output-root "$analysis_root/frozen_action_gain_full_validation"

echo "[$(date --iso-8601=seconds)] All stages complete"
