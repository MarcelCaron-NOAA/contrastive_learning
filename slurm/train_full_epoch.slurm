#!/bin/bash
#SBATCH --job-name=contrast_epoch1
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/full_epoch_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/full_epoch_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/train_full_epoch.py \
  --epochs 1 \
  --batch-size 8 \
  --negative-count 8 \
  --patch-size 256 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --validate-every 1000 \
  --print-every 100 \
  --checkpoint-every 2500 \
  --validation-samples 512 \
  --validation-split validation \
  --temperature 0.10 \
  --learning-rate 3e-4 \
  --warmup-start-lr 3e-5 \
  --warmup-fraction 0.05 \
  --min-learning-rate 3e-6 \
  --weight-decay 1e-4

