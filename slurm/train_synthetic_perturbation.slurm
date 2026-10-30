#!/bin/bash
#SBATCH --job-name=synth_perturb
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

PROFILE="${PROFILE:-all8}"
MAGNITUDE_MODE="${MAGNITUDE_MODE:-continuous}"
NEGATIVE_COUNT="${NEGATIVE_COUNT:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TARGET_FLOOR="${TARGET_FLOOR:-0.0}"
RUN_NAME="${RUN_NAME:-synthetic_perturb_${PROFILE}_v1}"
OUTPUT_DIR="${DATA_DIR}/derived/experiments/${RUN_NAME}"

EXTRA_ARGS=()
if [[ -n "${END_STEP:-}" ]]; then
  EXTRA_ARGS+=(--end-step "${END_STEP}")
fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi

python scripts/train_synthetic_perturbation.py \
  --profile "${PROFILE}" \
  --magnitude-mode "${MAGNITUDE_MODE}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --negative-count "${NEGATIVE_COUNT}" \
  --patch-size 256 \
  --temperature 0.10 \
  --target-floor "${TARGET_FLOOR}" \
  --lambda-global 1.0 \
  --lambda-penalty 1.0 \
  --symmetric-global \
  --wet-threshold-mm 0.1 \
  --min-anchor-wet-fraction 0.001 \
  --learning-rate 3e-4 \
  --warmup-start-lr 3e-5 \
  --warmup-fraction 0.05 \
  --min-learning-rate 3e-6 \
  --weight-decay 1e-4 \
  --validation-samples 512 \
  --validate-every 500 \
  --checkpoint-every 1500 \
  --num-workers 8 \
  --prefetch-factor 2 \
  "${EXTRA_ARGS[@]}"

python scripts/plot_synthetic_perturbation_training.py \
  --experiment-dir "${OUTPUT_DIR}"

