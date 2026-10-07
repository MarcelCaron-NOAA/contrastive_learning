#!/bin/bash
#SBATCH --job-name=contrast_diag
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/validation_diag_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/validation_diag_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/evaluate_fixed_validation_diagnostics.py \
  --batch-size 8 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --wet-threshold-mm 0.1

python scripts/plot_validation_by_group.py

