#!/bin/bash
#SBATCH --job-name=sim_diag2
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/similarity_diag2_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/similarity_diag2_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "1) Fixed-validation A/P and A/N similarity decomposition"
echo "============================================================"
python scripts/evaluate_validation_similarity_pairs.py \
  --batch-size 8 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --wet-threshold-mm 0.1

python scripts/plot_similarity_decomposition.py

echo
echo "============================================================"
echo "2) Unseen +/-1 / +/-3 / +/-6 h similarity gradation test"
echo "============================================================"
python scripts/evaluate_timelag_gradation.py \
  --split validation \
  --anchor-fhr 12 \
  --sample-count 1000 \
  --patch-size 256 \
  --batch-size 8 \
  --num-workers 8 \
  --prefetch-factor 2

echo
echo "============================================================"
echo "3) Small physical case gallery"
echo "============================================================"
python scripts/make_similarity_case_gallery.py \
  --cases-per-category 3

