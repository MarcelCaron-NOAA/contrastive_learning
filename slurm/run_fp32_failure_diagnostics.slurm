#!/bin/bash
#SBATCH --job-name=contrast_fp32diag
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/fp32_diag_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/fp32_diag_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "1) FP32 fixed-validation failure-tail + traditional metrics"
echo "============================================================"
python scripts/evaluate_fp32_failure_tail.py \
  --batch-size 8 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --wet-threshold-mm 0.1 \
  --failure-gallery-count 6

echo
echo "============================================================"
echo "2) FP32 +/-1 / +/-3 / +/-6 h gradation"
echo "============================================================"
python scripts/evaluate_timelag_gradation_fp32.py \
  --split validation \
  --anchor-fhr 12 \
  --sample-count 1000 \
  --patch-size 256 \
  --batch-size 8 \
  --num-workers 8 \
  --prefetch-factor 2

