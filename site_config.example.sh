#!/usr/bin/env bash
# Local URSA configuration for the contrastive-learning project.
#
# Copy this file to site_config.sh and edit the values for your account:
#   cp site_config.example.sh site_config.sh
#
# site_config.sh is intentionally ignored by Git. PROJECT_DIR is detected
# automatically from the location of load_env.sh, so it does not belong here.

# Large working-data area. Keep training data, experiment outputs, logs, and
# scratch files outside the Git repository.
export WORK_DIR="/scratch4/NCEPDEV/stmp/${USER}/mach_learn/contrastive_learning"

# Conda environment created from environment.yml.
export ENV_DIR="/scratch4/NCEPDEV/ovp/${USER}/mach_learn/envs/contrastive_learning"

# Usually leave these derived from WORK_DIR. Override only if desired.
export DATA_DIR="${WORK_DIR}/data"
export LOG_DIR="${WORK_DIR}/logs"
export SCRATCH_DIR="${WORK_DIR}/scratch"

# URSA maintained Conda module used by this project.
export CONDA_MODULE="rdhpcs-conda/25.11.0"

# Optional: shared/internal archive location for restoring the frozen dataset.
# NOAA/EMC users can copy the group-readable snapshot path from docs/DATA.md.
# Leave blank when HPSS is not being used.
export HPSS_DATA_ROOT=""
