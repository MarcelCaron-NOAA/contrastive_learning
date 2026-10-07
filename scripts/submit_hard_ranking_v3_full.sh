#!/usr/bin/env bash
set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

jid="$(sbatch --parsable slurm/train_hard_ranking_v3_full.slurm)"
echo "Submitted hard-ranking V3 full run: ${jid}"
echo "squeue -j ${jid}"
echo "Logs:"
echo "  ${LOG_DIR}/hard_rank_v3_${jid}.out"
echo "  ${LOG_DIR}/hard_rank_v3_${jid}.err"

