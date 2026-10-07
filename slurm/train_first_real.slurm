#!/bin/bash
#SBATCH --job-name=contrast_train1
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/first_real_train_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/first_real_train_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

# One numerical-library thread per process. The eight DataLoader workers are
# where we want CPU-side parallelism.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/train_first_real.py \
  --batch-size 8 \
  --negative-count 8 \
  --patch-size 256 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --steps 5000 \
  --validate-every 500 \
  --print-every 50 \
  --validation-samples 512 \
  --validation-split validation \
  --temperature 0.10 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4

