#!/bin/bash
#SBATCH --job-name=wang_v2
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/wang_v2_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/wang_v2_%j.err
set -euo pipefail
cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh
python scripts/evaluate_wang_v2.py --threshold-mm 2.54 --batch-size 32
OUT="$DATA_DIR/derived/experiments/hard_ranking_finetune_v1/evaluation/wang_v1_vs_v2"
python scripts/plot_wang_v2.py "$OUT/summary_by_bias.csv" --output-dir "$OUT"

