#!/usr/bin/env bash
set -euo pipefail
cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh
RUN_NAME="${RUN_NAME:-synthetic_perturb_syn8_v2_smoke}"
WALLTIME="${WALLTIME:-00:45:00}"
END_STEP="${END_STEP:-10}"
EXPORTS="ALL,RUN_NAME=$RUN_NAME,BATCH_SIZE=${BATCH_SIZE:-8},END_STEP=$END_STEP,RESUME_CHECKPOINT="
jid="$(sbatch --parsable \
  --job-name=syn8_wang_smoke \
  --time="$WALLTIME" \
  --export="$EXPORTS" \
  slurm/train_syn8_wang.slurm)"
echo "Submitted revised SYN-8 smoke test: $jid"
echo "Run name: $RUN_NAME"
echo "Log: $LOG_DIR/${RUN_NAME}_${jid}.log"

