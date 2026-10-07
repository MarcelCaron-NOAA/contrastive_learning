#!/bin/bash
#SBATCH --job-name=morph_v2
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/morph_v2_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/morph_v2_%j.err
set -euo pipefail
cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh
python scripts/evaluate_signed_morphology_v2.py --case-count 500 --case-seed 20261007 --batch-size 32 --threshold-mm 2.54
OUT="$DATA_DIR/derived/experiments/hard_ranking_finetune_v1/evaluation/signed_morphology_v1_vs_v2"
python scripts/plot_signed_morphology_wang_style.py "$OUT/signed_morphology_summary.csv" --output-dir "$OUT"

