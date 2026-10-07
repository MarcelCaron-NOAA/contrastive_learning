#!/bin/bash
#SBATCH --job-name=bnrecal_phys
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/bnrecal_physical_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/bnrecal_physical_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

EXP="$DATA_DIR/derived/experiments/full_epoch_train_validation_v1"
RECAL="$EXP/bn_recalibration"

for CKPT in \
  "$RECAL/checkpoint_best_bnrecal.pt" \
  "$RECAL/checkpoint_last_bnrecal.pt"
do
  echo
  echo "======================================================================"
  echo "Failure tail: $CKPT"
  echo "======================================================================"
  python scripts/evaluate_bnrecal_failure_tail.py \
    --checkpoint "$CKPT" \
    --batch-size 8 \
    --num-workers 8 \
    --prefetch-factor 2 \
    --wet-threshold-mm 0.1 \
    --failure-gallery-count 6

  echo
  echo "======================================================================"
  echo "1/3/6-h time-lag gradation: $CKPT"
  echo "======================================================================"
  python scripts/evaluate_bnrecal_timelag_fp32.py \
    --checkpoint "$CKPT" \
    --split validation \
    --anchor-fhr 12 \
    --sample-count 1000 \
    --patch-size 256 \
    --batch-size 8 \
    --num-workers 8 \
    --prefetch-factor 2
done

