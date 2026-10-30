#!/bin/bash
#SBATCH --job-name=morph_v2_v3
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/morph_v2_v3_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/morph_v2_v3_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

OUT_DIR="${DATA_DIR}/derived/experiments/hard_ranking_v3_full_v1/evaluation/morphology_v2_v3"

python scripts/evaluate_morphology_v2_v3.py \
  --output-dir "${OUT_DIR}" \
  --case-count 500 \
  --case-seed 20261007 \
  --batch-size 32 \
  --threshold-mm 2.54

python scripts/plot_morphology_v2_v3_wang_style.py \
  "${OUT_DIR}/summary_by_bias.csv" \
  --output-dir "${OUT_DIR}"
