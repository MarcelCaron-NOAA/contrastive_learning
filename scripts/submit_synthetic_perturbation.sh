#!/usr/bin/env bash
set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

PROFILE="${PROFILE:-all8}"
RUN_NAME="${RUN_NAME:-synthetic_perturb_${PROFILE}_v1}"

jid="$(sbatch --parsable \
  --job-name="synth_${PROFILE}" \
  --output="${LOG_DIR}/${RUN_NAME}_%j.out" \
  --error="${LOG_DIR}/${RUN_NAME}_%j.err" \
  --export=ALL,PROFILE="${PROFILE}",RUN_NAME="${RUN_NAME}",MAGNITUDE_MODE="${MAGNITUDE_MODE:-continuous}",NEGATIVE_COUNT="${NEGATIVE_COUNT:-4}",BATCH_SIZE="${BATCH_SIZE:-8}",TARGET_FLOOR="${TARGET_FLOOR:-0.0}",END_STEP="${END_STEP:-}",RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}" \
  slurm/train_synthetic_perturbation.slurm)"

echo "Submitted synthetic perturbation run: ${jid}"
echo "Profile : ${PROFILE}"
echo "Run name: ${RUN_NAME}"
echo "squeue -j ${jid}"
echo "Logs:"
echo "  ${LOG_DIR}/${RUN_NAME}_${jid}.out"
echo "  ${LOG_DIR}/${RUN_NAME}_${jid}.err"
