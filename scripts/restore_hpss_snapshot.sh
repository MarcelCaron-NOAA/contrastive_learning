#!/usr/bin/env bash
# Restore the frozen 2026-10-30 NOAA/EMC dataset snapshot from HPSS.
#
# Usage:
#   ./scripts/restore_hpss_snapshot.sh list
#   ./scripts/restore_hpss_snapshot.sh support
#   ./scripts/restore_hpss_snapshot.sh native [YEAR ...]
#   ./scripts/restore_hpss_snapshot.sh extended [YEAR ...]
#   ./scripts/restore_hpss_snapshot.sh all
#
# Set HPSS_DATA_ROOT in site_config.sh to the group-readable archive root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
SITE_CONFIG="${CL_SITE_CONFIG:-${PROJECT_DIR}/site_config.sh}"
if [[ ! -f "${SITE_CONFIG}" ]]; then
    echo "ERROR: ${SITE_CONFIG} does not exist." >&2
    echo "Copy site_config.example.sh to site_config.sh and edit it first." >&2
    exit 2
fi
# shellcheck source=/dev/null
source "${SITE_CONFIG}"
: "${WORK_DIR:?WORK_DIR must be set in site_config.sh}"
DATA_DIR="${DATA_DIR:-${WORK_DIR}/data}"
: "${HPSS_DATA_ROOT:?Set HPSS_DATA_ROOT in site_config.sh; see docs/DATA.md}"

if ! command -v htar >/dev/null 2>&1; then
    echo "ERROR: htar is not available in this shell." >&2
    exit 2
fi

MODE="${1:-}"
shift || true
DRY_RUN="${DRY_RUN:-0}"

native_years=(2020 2021 2022 2023 2024 2025 2026)
extended_years=(2024 2025)

run_htar() {
    local archive="$1"
    echo
    echo "Restoring: ${HPSS_DATA_ROOT}/${archive}"
    echo "Into     : ${DATA_DIR}"
    if [[ "${DRY_RUN}" == 1 ]]; then
        printf 'DRY RUN: (cd %q && htar -xvf %q)\n' "${DATA_DIR}" "${HPSS_DATA_ROOT}/${archive}"
    else
        mkdir -p "${DATA_DIR}"
        (cd "${DATA_DIR}" && htar -xvf "${HPSS_DATA_ROOT}/${archive}")
    fi
}

list_archives() {
    echo "HPSS snapshot: ${HPSS_DATA_ROOT}"
    if command -v hsi >/dev/null 2>&1; then
        hsi "ls -l ${HPSS_DATA_ROOT}"
    else
        echo "hsi is unavailable; expected archives are:"
        echo "  project_support_20261030.tar"
        for y in "${native_years[@]}"; do echo "  hrrr_qpf6_native_${y}.tar"; done
        for y in "${extended_years[@]}"; do echo "  hrrr_qpf6_extended_${y}.tar"; done
    fi
}

restore_native() {
    local years=("$@")
    [[ ${#years[@]} -gt 0 ]] || years=("${native_years[@]}")
    for y in "${years[@]}"; do
        [[ " ${native_years[*]} " == *" ${y} "* ]] || { echo "Unsupported native year: ${y}" >&2; exit 2; }
        run_htar "hrrr_qpf6_native_${y}.tar"
    done
}

restore_extended() {
    local years=("$@")
    [[ ${#years[@]} -gt 0 ]] || years=("${extended_years[@]}")
    for y in "${years[@]}"; do
        [[ " ${extended_years[*]} " == *" ${y} "* ]] || { echo "Unsupported extended year: ${y}" >&2; exit 2; }
        run_htar "hrrr_qpf6_extended_${y}.tar"
    done
}

case "${MODE}" in
    list)
        list_archives
        ;;
    support)
        run_htar "project_support_20261030.tar"
        ;;
    native)
        restore_native "$@"
        ;;
    extended)
        restore_extended "$@"
        ;;
    all)
        [[ $# -eq 0 ]] || { echo "'all' takes no year arguments" >&2; exit 2; }
        run_htar "project_support_20261030.tar"
        restore_native
        restore_extended
        ;;
    *)
        cat >&2 <<'EOF'
Usage:
  ./scripts/restore_hpss_snapshot.sh list
  ./scripts/restore_hpss_snapshot.sh support
  ./scripts/restore_hpss_snapshot.sh native [YEAR ...]
  ./scripts/restore_hpss_snapshot.sh extended [YEAR ...]
  ./scripts/restore_hpss_snapshot.sh all
EOF
        exit 2
        ;;
esac
