#!/bin/bash
#SBATCH --job-name=hard_rank_v3
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/hard_rank_v3_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/hard_rank_v3_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

OUTPUT_DIR="${DATA_DIR}/derived/experiments/hard_ranking_v3_full_v1"

EXTRA_ARGS=()
if [[ -n "${END_STEP:-}" ]]; then
  EXTRA_ARGS+=(--end-step "${END_STEP}")
fi

python scripts/train_hard_ranking_v3_full.py \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 8 \
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

python scripts/plot_hard_ranking_v3_full.py \
  --experiment-dir "${OUTPUT_DIR}"

