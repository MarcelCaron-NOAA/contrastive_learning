#!/bin/bash
#SBATCH --job-name=final_inches
#SBATCH --partition=u1-compute
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
PROJECT_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning"
cd "$PROJECT_DIR"
source load_env.sh
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/final_inches_${SLURM_JOB_ID}.log"
exec >"$LOG_FILE" 2>&1

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Log: $LOG_FILE"

echo "=== Stage-IV inch metrics ==="
python scripts/recompute_final_stageiv_metrics_inches.py

echo "=== Stage-IV galleries ==="
python scripts/plot_final_stageiv_buckets.py

echo "=== Morphology traditional metrics at 0.1 in ==="
python scripts/recompute_final_morphology_traditional_inches.py

echo "=== Morphology plots ==="
python scripts/plot_final_morphology_models.py

echo "=== Area-size audit ==="
python scripts/audit_area_size_response.py

echo "Finished: $(date)"
