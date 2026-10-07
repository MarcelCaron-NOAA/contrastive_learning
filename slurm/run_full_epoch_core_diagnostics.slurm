#!/bin/bash
#SBATCH --job-name=epoch1_diag
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/full_epoch_diag_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/full_epoch_diag_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

EXP="$DATA_DIR/derived/experiments/full_epoch_train_validation_v1"

echo "============================================================"
echo "1) Plot full-epoch history"
echo "============================================================"
python scripts/plot_full_epoch_history.py

echo
echo "============================================================"
echo "2) Diagnose BatchNorm running-stat oscillation"
echo "============================================================"
python scripts/diagnose_batchnorm_swap.py \
  --batch-size 8 \
  --num-workers 8 \
  --prefetch-factor 2

echo
echo "============================================================"
echo "3) FP32 failure tail for BEST checkpoint"
echo "============================================================"
python scripts/evaluate_full_epoch_failure_tail.py \
  --checkpoint "$EXP/checkpoint_best.pt" \
  --batch-size 8 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --wet-threshold-mm 0.1 \
  --failure-gallery-count 6

echo
echo "============================================================"
echo "4) FP32 1/3/6-h gradation for BEST checkpoint"
echo "============================================================"
python scripts/evaluate_full_epoch_timelag_fp32.py \
  --checkpoint "$EXP/checkpoint_best.pt" \
  --split validation \
  --anchor-fhr 12 \
  --sample-count 1000 \
  --patch-size 256 \
  --batch-size 8 \
  --num-workers 8 \
  --prefetch-factor 2

