#!/bin/bash
#SBATCH --job-name=syn3_diag
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning"
cd "$PROJECT_DIR"

source load_env.sh

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/syn3_diagnostics_${SLURM_JOB_ID}.log"
exec >"$LOG_FILE" 2>&1

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Log: $LOG_FILE"
echo

SYN3_DIR="$DATA_DIR/derived/experiments/synthetic_perturb_syn3_v1"
SYN8_DIR="$DATA_DIR/derived/experiments/synthetic_perturb_all8_v1"

SYN3_CKPT="$SYN3_DIR/checkpoint_best.pt"

echo "============================================================"
echo "1. SYN-3 Stage-IV evaluation"
echo "============================================================"

python scripts/evaluate_stageiv_synthetic_vs_v3.py \
  --synthetic-checkpoint "$SYN3_CKPT" \
  --output-dir "$SYN3_DIR/evaluation/stageiv_vs_v3"

echo
echo "============================================================"
echo "2. SYN-3 Stage-IV galleries"
echo "============================================================"

python scripts/plot_stageiv_syn3_galleries.py \
  --primary-evaluation-dir "$SYN3_DIR/evaluation/stageiv_vs_v3" \
  --comparison-evaluation-dir "$SYN8_DIR/evaluation/stageiv_vs_v3" \
  --primary-label "SYN-3" \
  --comparison-label "SYN-8" \
  --natural-label "NFV" \
  --count 5

echo
echo "============================================================"
echo "3. SYN-3 morphology evaluation"
echo "============================================================"

python scripts/evaluate_morphology_synthetic_vs_v3.py \
  --synthetic-checkpoint "$SYN3_CKPT" \
  --output-dir "$SYN3_DIR/evaluation/morphology_vs_v3" \
  --case-count 500 \
  --case-seed 20261007 \
  --batch-size 32 \
  --threshold-mm 2.54

echo
echo "============================================================"
echo "4. SYN-3 vs traditional morphology plot"
echo "============================================================"

python scripts/plot_morphology_syn3_traditional.py \
  --evaluation-dir "$SYN3_DIR/evaluation/morphology_vs_v3" \
  --synthetic-label "SYN-3"

echo
echo "============================================================"
echo "5. SYN-3 vs SYN-8 morphology plot"
echo "============================================================"

python scripts/plot_morphology_syn3_vs_syn8.py \
  --syn3-evaluation-dir "$SYN3_DIR/evaluation/morphology_vs_v3" \
  --syn8-evaluation-dir "$SYN8_DIR/evaluation/morphology_vs_v3" \
  --syn3-label "SYN-3" \
  --syn8-label "SYN-8"

echo
echo "============================================================"
echo "SYN-3 DIAGNOSTICS COMPLETE"
echo "============================================================"
echo "Finished: $(date)"
echo
echo "Stage-IV galleries:"
echo "  $SYN3_DIR/evaluation/stageiv_vs_v3/galleries_syn3_vs_syn8"
echo
echo "Morphology:"
echo "  $SYN3_DIR/evaluation/morphology_vs_v3"
