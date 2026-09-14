#!/usr/bin/env bash

# Project paths
export PROJECT_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/contrastive_learning"
export WORK_DIR="/scratch4/NCEPDEV/stmp/Marcel.Caron/mach_learn/contrastive_learning"

export DATA_DIR="${WORK_DIR}/data"
export LOG_DIR="${WORK_DIR}/logs"
export SCRATCH_DIR="${WORK_DIR}/scratch"

export ENV_DIR="/scratch4/NCEPDEV/ovp/Marcel.Caron/mach_learn/envs/contrastive_learning"

# Create working directories
mkdir -p "${DATA_DIR}" "${LOG_DIR}" "${SCRATCH_DIR}"

# Load maintained Conda
module load rdhpcs-conda/25.11.0

# Activate environment
conda activate "${ENV_DIR}"

# Avoid oversubscribing CPU threads in batch jobs unless explicitly requested
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Make project modules importable
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

cd "${PROJECT_DIR}"

echo "============================================================"
echo " Contrastive Learning Project Environment"
echo "============================================================"
echo "PROJECT_DIR : ${PROJECT_DIR}"
echo "DATA_DIR    : ${DATA_DIR}"
echo "LOG_DIR     : ${LOG_DIR}"
echo "SCRATCH_DIR : ${SCRATCH_DIR}"
echo "Python      : $(which python)"
echo "Version     : $(python --version 2>&1)"
echo "============================================================"
