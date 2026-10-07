#!/bin/bash
#SBATCH --job-name=epoch1_lastdiag
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/full_epoch_lastdiag_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/full_epoch_lastdiag_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

EXP="$DATA_DIR/derived/experiments/full_epoch_train_validation_v1"

python scripts/evaluate_full_epoch_failure_tail.py \
  --checkpoint "$EXP/checkpoint_last.pt" \
  --batch-size 8 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --wet-threshold-mm 0.1 \
  --failure-gallery-count 6

python scripts/evaluate_full_epoch_timelag_fp32.py \
  --checkpoint "$EXP/checkpoint_last.pt" \
  --split validation \
  --anchor-fhr 12 \
  --sample-count 1000 \
  --patch-size 256 \
  --batch-size 8 \
  --num-workers 8 \
  --prefetch-factor 2

