#!/bin/bash
#SBATCH --job-name=contrast_gn_full
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/groupnorm_full_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/groupnorm_full_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

EXP="$DATA_DIR/derived/experiments/groupnorm_pilot_v1"

# Override at submit time if recovering from a later periodic checkpoint:
#   sbatch --export=ALL,RESUME_CHECKPOINT=/path/to/checkpoint_stepXXXXX.pt \
#          slurm/continue_groupnorm_full_epoch.slurm
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$EXP/checkpoint_last.pt}"

if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "ERROR: resume checkpoint does not exist: $RESUME_CHECKPOINT" >&2
  exit 2
fi

echo "Continuing GroupNorm experiment"
echo "  EXP               = $EXP"
echo "  RESUME_CHECKPOINT = $RESUME_CHECKPOINT"
echo "  target             = complete deterministic epoch"

python scripts/train_groupnorm.py \
  --output-dir "$EXP" \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --batch-size 8 \
  --negative-count 8 \
  --patch-size 256 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --validate-every 500 \
  --print-every 100 \
  --checkpoint-every 2500 \
  --validation-samples 512 \
  --validation-split validation \
  --temperature 0.10 \
  --learning-rate 3e-4 \
  --warmup-start-lr 3e-5 \
  --warmup-fraction 0.05 \
  --min-learning-rate 3e-6 \
  --weight-decay 1e-4

# Refresh the BN-vs-GN trajectory plots through the completed epoch.
python scripts/plot_bn_vs_groupnorm.py

