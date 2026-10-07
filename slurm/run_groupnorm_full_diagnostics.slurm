#!/bin/bash
#SBATCH --job-name=gn_full_diag
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/groupnorm_diag_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/groupnorm_diag_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

EXP="$DATA_DIR/derived/experiments/groupnorm_pilot_v1"

for CKPT in "$EXP/checkpoint_best.pt" "$EXP/checkpoint_last.pt"
do
  echo
  echo "========================================================================"
  echo "FP32 failure tail: $CKPT"
  echo "========================================================================"
  python scripts/evaluate_failure_tail.py \
    --experiment-dir "$EXP" \
    --checkpoint "$CKPT" \
    --batch-size 8 \
    --num-workers 8 \
    --prefetch-factor 2 \
    --wet-threshold-mm 0.1 \
    --failure-gallery-count 6

  echo
  echo "========================================================================"
  echo "FP32 1/3/6-h time-lag gradation: $CKPT"
  echo "========================================================================"
  python scripts/evaluate_timelag_fp32.py \
    --experiment-dir "$EXP" \
    --checkpoint "$CKPT" \
    --split validation \
    --anchor-fhr 12 \
    --sample-count 1000 \
    --patch-size 256 \
    --batch-size 8 \
    --num-workers 8 \
    --prefetch-factor 2
done

