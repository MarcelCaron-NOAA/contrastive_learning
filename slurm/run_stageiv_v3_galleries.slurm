#!/bin/bash
#SBATCH --job-name=stageiv_v3_gallery
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/stageiv_v3_gallery_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/stageiv_v3_gallery_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

OUT_DIR="${DATA_DIR}/derived/experiments/hard_ranking_v3_full_v1/evaluation/stageiv_v1_v2_v3"
GALLERY_DIR="${DATA_DIR}/derived/experiments/hard_ranking_v3_full_v1/evaluation/stageiv_case_examples_v3"

python scripts/evaluate_stageiv_v1_v2_v3.py \
  --output-dir "${OUT_DIR}" \
  --batch-size 32

python scripts/plot_stageiv_case_examples_v3.py \
  --archive-dir "${DATA_DIR}/derived/evaluation/stageiv_reference_v1" \
  --metrics-csv "${OUT_DIR}/per_case_metrics.csv.gz" \
  --output-dir "${GALLERY_DIR}" \
  --n-per-bucket 5 \
  --min-wet-fraction-0p1in 0.001 \
  --min-max-in 0.10
