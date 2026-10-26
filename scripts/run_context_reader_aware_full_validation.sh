#!/usr/bin/env bash
set -eu

cd /home/lys/Desktop/UROP
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python_bin=.venv/bin/python
train_root=data/benchmarks/multihop/context_reader_aware/train_pool_seed42_2000
validation_root=data/benchmarks/multihop/context_reader_aware/full_validation
model_root=data/models/context_reader_aware_full_validation
analysis_root=data/benchmarks/multihop/context_reader_aware/analysis

mkdir -p "$model_root" "$analysis_root"

echo "[1/9] train-only 무작위 2,000문항 pool 생성 시작"
"$python_bin" -m experiments.evaluate_context_reader_aware_prefixes \
  --datasets hotpotqa musique 2wiki \
  --split train \
  --max-queries-per-dataset 2000 \
  --query-seed 42 \
  --output-root "$train_root" \
  --log-every 25

echo "[2/9] HotpotQA 전체 validation 시작"
"$python_bin" -m experiments.evaluate_context_reader_aware_prefixes \
  --datasets hotpotqa \
  --split validation \
  --max-queries-per-dataset 0 \
  --query-seed 42 \
  --output-root "$validation_root" \
  --log-every 25

echo "[3/9] MuSiQue 전체 validation 시작"
"$python_bin" -m experiments.evaluate_context_reader_aware_prefixes \
  --datasets musique \
  --split validation \
  --max-queries-per-dataset 0 \
  --query-seed 42 \
  --output-root "$validation_root" \
  --log-every 25

echo "[4/9] 2Wiki 전체 validation 시작"
"$python_bin" -m experiments.evaluate_context_reader_aware_prefixes \
  --datasets 2wiki \
  --split validation \
  --max-queries-per-dataset 0 \
  --query-seed 42 \
  --output-root "$validation_root" \
  --log-every 25

echo "[5/9] seed=42 정책 학습·전체 validation 적용"
"$python_bin" -m experiments.train_reader_aware_sufficiency_policy \
  --train-root "$train_root" \
  --validation-root "$validation_root" \
  --max-train-rows-per-dataset 1200 \
  --train-sample-seed 42 \
  --tune-every 6 \
  --max-validation-rows-per-dataset 0 \
  --minimum-stop-step 2 \
  --seed 42 \
  --model-out "$model_root/policy_seed42.joblib" \
  --report-out "$model_root/policy_seed42.report.json"

echo "[6/9] seed=43 정책 학습·전체 validation 적용"
"$python_bin" -m experiments.train_reader_aware_sufficiency_policy \
  --train-root "$train_root" \
  --validation-root "$validation_root" \
  --max-train-rows-per-dataset 1200 \
  --train-sample-seed 43 \
  --tune-every 6 \
  --max-validation-rows-per-dataset 0 \
  --minimum-stop-step 2 \
  --seed 43 \
  --model-out "$model_root/policy_seed43.joblib" \
  --report-out "$model_root/policy_seed43.report.json"

echo "[7/9] seed=44~46 정책 학습·전체 validation 적용"
for policy_seed in 44 45 46
do
  "$python_bin" -m experiments.train_reader_aware_sufficiency_policy \
    --train-root "$train_root" \
    --validation-root "$validation_root" \
    --max-train-rows-per-dataset 1200 \
    --train-sample-seed "$policy_seed" \
    --tune-every 6 \
    --max-validation-rows-per-dataset 0 \
    --minimum-stop-step 2 \
    --seed "$policy_seed" \
    --model-out "$model_root/policy_seed${policy_seed}.joblib" \
    --report-out "$model_root/policy_seed${policy_seed}.report.json"
done

echo "[8/9] 데이터셋별 다중-seed 통계 분석"
"$python_bin" -m experiments.analyze_context_reader_aware_multiseed \
  --prefix-root "$validation_root" \
  --reports \
    "$model_root/policy_seed42.report.json" \
    "$model_root/policy_seed43.report.json" \
    "$model_root/policy_seed44.report.json" \
    "$model_root/policy_seed45.report.json" \
    "$model_root/policy_seed46.report.json" \
  --bootstrap-samples 20000 \
  --output "$analysis_root/full_validation_multiseed.json"

echo "[9/9] 전체 자동 테스트"
"$python_bin" -m unittest discover -s tests

echo "Qwen3:8B unload"
curl -sS http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3:8b","keep_alive":0}'

echo "FULL_VALIDATION_COMPLETE"
