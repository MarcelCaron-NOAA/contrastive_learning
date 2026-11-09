#!/bin/bash
#SBATCH --job-name=nfv
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUN_NAME="${RUN_NAME:-hard_ranking_v3_full_v1}"

# Capture bootstrap failures too.  submit_nfv.sh normally exports LOG_DIR after
# sourcing load_env.sh; the fallback only matters for a direct sbatch launch.
BOOTSTRAP_LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${BOOTSTRAP_LOG_DIR}"
LOG_FILE="${BOOTSTRAP_LOG_DIR}/${RUN_NAME}_${SLURM_JOB_ID}.log"
exec >"${LOG_FILE}" 2>&1

cd "${PROJECT_DIR}"
source load_env.sh

OUTPUT_DIR="${DATA_DIR}/derived/experiments/${RUN_NAME}"

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo

EXTRA_ARGS=()
if [[ -n "${END_STEP:-}" ]]; then
    EXTRA_ARGS+=(--end-step "${END_STEP}")
fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    EXTRA_ARGS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi

python scripts/train_nfv.py \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --negative-count 4 \
    --patch-size 256 \
    --lambda-global 1.0 \
    --lambda-tl 1.0 \
    --lambda-evolution 1.0 \
    --global-temperature 0.10 \
    --ranking-temperature 0.10 \
    --learning-rate 3e-4 \
    --warmup-start-lr 3e-5 \
    --warmup-fraction 0.05 \
    --min-learning-rate 3e-6 \
    --weight-decay 1e-4 \
    --validation-samples 512 \
    --validate-every 500 \
    --checkpoint-every 2500 \
    --num-workers 8 \
    --prefetch-factor 2 \
    "${EXTRA_ARGS[@]}"

# Training intentionally stops here. Evaluation and plotting are separate
# workflows so they can carry their own data/grid requirements.
echo "Finished training: $(date)"
