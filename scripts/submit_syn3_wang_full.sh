#!/usr/bin/env bash
set -euo pipefail
cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh
RUN_NAME="${RUN_NAME:-synthetic_perturb_syn3_v2}"
WALLTIME="${WALLTIME:-05:00:00}"
EXPORTS="ALL,RUN_NAME=$RUN_NAME,BATCH_SIZE=${BATCH_SIZE:-8},END_STEP="
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXPORTS="${EXPORTS},RESUME_CHECKPOINT=${RESUME_CHECKPOINT}"
else
  EXPORTS="${EXPORTS},RESUME_CHECKPOINT="
fi
jid="$(sbatch --parsable \
  --job-name=syn3_wang \
  --time="$WALLTIME" \
  --export="$EXPORTS" \
  slurm/train_syn3_wang.slurm)"
echo "Submitted revised SYN-3 full run: $jid"
echo "Run name: $RUN_NAME"
echo "Log: $LOG_DIR/${RUN_NAME}_${jid}.log"

