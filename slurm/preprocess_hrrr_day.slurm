#!/usr/bin/env bash
#SBATCH --job-name=hrrr_qpf6
#SBATCH --partition=u1-service
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/hrrr_qpf6_%A_%a.out
#SBATCH --error=/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning/logs/hrrr_qpf6_%A_%a.err
set -euo pipefail
PROJECT_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning"
DATE_LIST="${1:?Usage: sbatch ... preprocess_hrrr_day.slurm DATE_LIST}"
source "${PROJECT_DIR}/load_env.sh"
export BLOSC_NTHREADS="${SLURM_CPUS_PER_TASK:-4}"
DATE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${DATE_LIST}")"
[[ -n "${DATE}" ]] || { echo "No date for task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2; }
echo "DATE=${DATE} HOST=$(hostname) JOB=${SLURM_JOB_ID} TASK=${SLURM_ARRAY_TASK_ID}"
python "${PROJECT_DIR}/scripts/preprocess_hrrr_day.py" --date "${DATE}"

