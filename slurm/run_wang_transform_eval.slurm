#!/bin/bash
#SBATCH --job-name=wang_transform
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/wang_transform_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/wang_transform_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

: "${WANG_THRESHOLD_MM:?Set WANG_THRESHOLD_MM after auditing the Wang released verification code. If using a project-specific threshold instead, document it explicitly.}"

#python scripts/build_wang_transform_archive.py \
#  --target-cases 2000 \
#  --patch-size 256 \
#  --wet-threshold-mm 0.1

python scripts/evaluate_wang_transformations.py \
  --threshold-mm "$WANG_THRESHOLD_MM" \
  --batch-size 32

RESULT="$DATA_DIR/derived/evaluation/wang_transform_v1/results"

python scripts/plot_wang_style.py \
  "$RESULT/summary_by_bias.csv"

python scripts/plot_wang_raw_scatter.py \
  "$RESULT/per_case_metrics.csv.gz"

