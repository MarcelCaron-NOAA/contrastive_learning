#!/usr/bin/env bash
# Source this file from the repository root (or from anywhere):
#   source load_env.sh
#
# Machine/user-specific paths live only in site_config.sh. The tracked example
# is site_config.example.sh.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: load_env.sh must be sourced, not executed." >&2
    echo "Use: source load_env.sh" >&2
    exit 2
fi

_CL_LOAD_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR="${_CL_LOAD_ENV_DIR}"
unset _CL_LOAD_ENV_DIR

export SITE_CONFIG="${CL_SITE_CONFIG:-${PROJECT_DIR}/site_config.sh}"

if [[ ! -f "${SITE_CONFIG}" ]]; then
    echo "ERROR: local site configuration not found:" >&2
    echo "  ${SITE_CONFIG}" >&2
    echo >&2
    echo "Create it once with:" >&2
    echo "  cp \"${PROJECT_DIR}/site_config.example.sh\" \"${PROJECT_DIR}/site_config.sh\"" >&2
    echo "and edit WORK_DIR / ENV_DIR for your URSA account." >&2
    return 2
fi

# shellcheck source=/dev/null
source "${SITE_CONFIG}"

: "${WORK_DIR:?WORK_DIR must be set in site_config.sh}"
: "${ENV_DIR:?ENV_DIR must be set in site_config.sh}"

export DATA_DIR="${DATA_DIR:-${WORK_DIR}/data}"
export LOG_DIR="${LOG_DIR:-${WORK_DIR}/logs}"
export SCRATCH_DIR="${SCRATCH_DIR:-${WORK_DIR}/scratch}"
export CONDA_MODULE="${CONDA_MODULE:-rdhpcs-conda/25.11.0}"

mkdir -p "${DATA_DIR}" "${LOG_DIR}" "${SCRATCH_DIR}"

# URSA uses a maintained Conda module. Allow an already-initialized shell to
# skip module loading, while keeping the normal batch/interactive path simple.
if command -v module >/dev/null 2>&1 && [[ -n "${CONDA_MODULE}" ]]; then
    module load "${CONDA_MODULE}"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is unavailable after loading ${CONDA_MODULE}." >&2
    return 2
fi

# `conda activate` is a shell function and may need explicit initialization in
# non-interactive shells after module load.
if ! type conda 2>/dev/null | grep -q 'function'; then
    _CL_CONDA_BASE="$(conda info --base 2>/dev/null)"
    if [[ -f "${_CL_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "${_CL_CONDA_BASE}/etc/profile.d/conda.sh"
    fi
    unset _CL_CONDA_BASE
fi

conda activate "${ENV_DIR}" || return 2

# Avoid oversubscribing CPU threads in batch jobs unless explicitly requested.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

# The repository currently exposes `src` as a top-level package. A later
# packaging refactor may replace this with an editable install.
case ":${PYTHONPATH:-}:" in
    *":${PROJECT_DIR}:"*) ;;
    *) export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

cat <<EOF2
============================================================
 Contrastive Learning Project Environment
============================================================
PROJECT_DIR : ${PROJECT_DIR}
WORK_DIR    : ${WORK_DIR}
DATA_DIR    : ${DATA_DIR}
LOG_DIR     : ${LOG_DIR}
SCRATCH_DIR : ${SCRATCH_DIR}
ENV_DIR     : ${ENV_DIR}
Python      : $(command -v python)
Version     : $(python --version 2>&1)
============================================================
EOF2
