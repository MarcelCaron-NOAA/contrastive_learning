#!/usr/bin/env bash
# Submit the final Natural Forecast Variability (NFV) training workflow on URSA.
#
# Optional environment overrides:
#   RUN_NAME, WALLTIME, END_STEP, BATCH_SIZE, RESUME_CHECKPOINT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
source load_env.sh

RUN_NAME="${RUN_NAME:-hard_ranking_v3_full_v1}"
WALLTIME="${WALLTIME:-16:00:00}"

EXPORTS="ALL,RUN_NAME=${RUN_NAME},BATCH_SIZE=${BATCH_SIZE:-8},END_STEP=${END_STEP:-},RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}"

jid="$(sbatch --parsable \
    --job-name=nfv \
    --time="${WALLTIME}" \
    --export="${EXPORTS}" \
    slurm/train_nfv.slurm)"

echo "Submitted NFV run: ${jid}"
echo "Run name: ${RUN_NAME}"
echo "Log: ${LOG_DIR}/${RUN_NAME}_${jid}.log"
