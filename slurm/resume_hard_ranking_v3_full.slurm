#!/bin/bash
#SBATCH --job-name=hard_rank_v3_r
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/hard_rank_v3_resume_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/hard_rank_v3_resume_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

OUTPUT_DIR="${DATA_DIR}/derived/experiments/hard_ranking_v3_full_v1"

# You can override this at submission time:
#   sbatch --export=ALL,RESUME_CHECKPOINT=/path/to/checkpoint_step25000.pt \
#     slurm/resume_hard_ranking_v3_full.slurm
if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
  RESUME_CHECKPOINT="$(
    find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'checkpoint_step*.pt' \
      | sort -V | tail -1
  )"

  # A deliberately partial run (for example END_STEP=500) writes
  # checkpoint_last.pt even if it has not yet reached the first 2500-step
  # periodic checkpoint. Use that as a fallback.
  if [[ -z "${RESUME_CHECKPOINT}" && -f "${OUTPUT_DIR}/checkpoint_last.pt" ]]; then
    RESUME_CHECKPOINT="${OUTPUT_DIR}/checkpoint_last.pt"
  fi
fi

if [[ -z "${RESUME_CHECKPOINT}" || ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "ERROR: Could not find a V3 checkpoint to resume." >&2
  echo "Set RESUME_CHECKPOINT explicitly if needed." >&2
  exit 2
fi

echo "Resuming from: ${RESUME_CHECKPOINT}"

EXTRA_ARGS=()
if [[ -n "${END_STEP:-}" ]]; then
  EXTRA_ARGS+=(--end-step "${END_STEP}")
fi

python scripts/train_hard_ranking_v3_full.py \
  --resume-checkpoint "${RESUME_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 8 \
  --negative-count 4 \
  --patch-size 256 \
  --lambda-global 1.0 \
  --lambda-tl 1.0 \
  --lambda-evolution 1.0 \
  --global-temperature 0.10 \
  --ranking-temperature 0.10 \
  --learning-rate 3e-4 \
  --warmup-start-lr 3e-5 \
  --warmup-fraction 0.05 \
  --min-learning-rate 3e-6 \
  --weight-decay 1e-4 \
  --validation-samples 512 \
  --validate-every 500 \
  --checkpoint-every 2500 \
  --num-workers 8 \
  --prefetch-factor 2 \
  "${EXTRA_ARGS[@]}"

python scripts/plot_hard_ranking_v3_full.py \
  --experiment-dir "${OUTPUT_DIR}"

