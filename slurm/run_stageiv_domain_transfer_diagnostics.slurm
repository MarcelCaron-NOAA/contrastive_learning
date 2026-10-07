#!/bin/bash
#SBATCH --job-name=st4_domain_diag
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/stageiv_domain_diag_%j.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/stageiv_domain_diag_%j.err

set -euo pipefail

cd /scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning
source load_env.sh
PYTHON_EXE="$(which python)"

echo "Python: $PYTHON_EXE"
"$PYTHON_EXE" -c 'import torch; print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())'

"$PYTHON_EXE" scripts/diagnose_stageiv_domain_transfer.py \
  --batch-size 32 \
  --batch-invariance-count 64 \
  --random-negatives 20 \
  --transform-sample-count 500 \
  --shift-cells 0 1 3 6

