#!/usr/bin/env bash
set -euo pipefail
cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh
RUN_NAME="${RUN_NAME:-synthetic_perturb_syn3_v2_smoke}"
END_STEP="${END_STEP:-10}"
WALLTIME="${WALLTIME:-00:45:00}"
jid="$(sbatch --parsable \
  --job-name=syn3w_smoke \
  --time="$WALLTIME" \
  --export=ALL,RUN_NAME="$RUN_NAME",END_STEP="$END_STEP",BATCH_SIZE="${BATCH_SIZE:-8}",RESUME_CHECKPOINT="" \
  slurm/train_syn3_wang.slurm)"
echo "Submitted revised SYN-3 smoke test: $jid"
echo "Run name: $RUN_NAME"
echo "Log: $LOG_DIR/${RUN_NAME}_${jid}.log"

