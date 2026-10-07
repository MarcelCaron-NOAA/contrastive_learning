#!/bin/bash
#SBATCH --job-name=hard_rank_ft
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/hard_rank_ft_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/hard_rank_ft_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

SOURCE_CHECKPOINT="${DATA_DIR}/derived/experiments/groupnorm_pilot_v1/checkpoint_best.pt"
OUTPUT_DIR="${DATA_DIR}/derived/experiments/hard_ranking_finetune_v1"

python scripts/train_hard_ranking_finetune.py \
  --source-checkpoint "${SOURCE_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --steps 12000 \
  --batch-size 8 \
  --negative-count 4 \
  --patch-size 256 \
  --lambda-global 0.25 \
  --lambda-tl 1.0 \
  --lambda-evolution 1.0 \
  --global-temperature 0.10 \
  --ranking-temperature 0.10 \
  --learning-rate 5e-5 \
  --warmup-start-lr 1e-5 \
  --warmup-steps 500 \
  --min-learning-rate 5e-6 \
  --weight-decay 1e-4 \
  --validation-samples 512 \
  --validate-every 500 \
  --checkpoint-every 1000 \
  --num-workers 8 \
  --prefetch-factor 2

python scripts/plot_hard_ranking_finetune.py \
  --experiment-dir "${OUTPUT_DIR}"

