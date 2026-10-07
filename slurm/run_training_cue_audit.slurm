#!/bin/bash
#SBATCH --job-name=train_cue_audit
#SBATCH --partition=u1-compute
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/train_cue_audit_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/train_cue_audit_%j.err

set -euo pipefail
cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh

python scripts/audit_training_cues.py \
  --sample-count 3000 \
  --batch-size 8 \
  --negative-count 8 \
  --patch-size 256 \
  --num-workers 8 \
  --seed 20260930 \
  --epoch 0

