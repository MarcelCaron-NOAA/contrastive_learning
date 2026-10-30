#!/usr/bin/env bash
# Small, explicit front door for the supported reproduction workflows.
# This script dispatches to documented commands; it intentionally does not
# silently chain long Slurm jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

usage() {
    cat <<'EOF'
Usage:
  ./scripts/reproduce.sh preflight [setup|support|training|diagnostics|all] [--verify-stores]
  ./scripts/reproduce.sh restore {list|support|native|extended|all} [YEAR ...]
  ./scripts/reproduce.sh train {syn3|syn8|nfv} [full|smoke]
  ./scripts/reproduce.sh diagnostic {morphology|stageiv|forecast-hour|inversion|timelag}
  ./scripts/reproduce.sh plot {morphology|stageiv|forecast-hour}
  ./scripts/reproduce.sh plan

Examples:
  ./scripts/reproduce.sh restore support
  ./scripts/reproduce.sh preflight diagnostics
  ./scripts/reproduce.sh train syn3 smoke
  ./scripts/reproduce.sh diagnostic morphology
  CASE_INDEX=299 N_STEPS=1200 ./scripts/reproduce.sh diagnostic inversion
EOF
}

plan() {
    cat <<'EOF'
Recommended order
=================

A. Recreate final diagnostics using the frozen checkpoints/products
  1. ./scripts/reproduce.sh restore support
  2. source load_env.sh
  3. ./scripts/reproduce.sh preflight diagnostics
  4. ./scripts/reproduce.sh diagnostic morphology
  5. ./scripts/reproduce.sh diagnostic stageiv
  6. ./scripts/reproduce.sh diagnostic forecast-hour
  7. CASE_INDEX=299 N_STEPS=1200 ./scripts/reproduce.sh diagnostic inversion

B. Retrain the three final learned metrics
  1. ./scripts/reproduce.sh restore support
  2. ./scripts/reproduce.sh restore native
  3. source load_env.sh
  4. ./scripts/reproduce.sh preflight training --verify-stores
  5. ./scripts/reproduce.sh train syn3 full
  6. ./scripts/reproduce.sh train syn8 full
  7. ./scripts/reproduce.sh train nfv full

Jobs are intentionally submitted separately. Check Slurm/log status before
starting downstream diagnostics; reproduce.sh does not silently chain jobs.
EOF
}

cmd="${1:-}"
shift || true

# These commands do not need the Conda runtime.
case "${cmd}" in
    restore)
        exec ./scripts/restore_hpss_snapshot.sh "$@"
        ;;
    plan)
        plan
        exit 0
        ;;
    -h|--help|help|"")
        usage
        exit 0
        ;;
esac

source load_env.sh >/dev/null

case "${cmd}" in
    preflight)
        scope="${1:-all}"
        [[ $# -gt 0 ]] && shift
        exec python scripts/preflight.py "${scope}" "$@"
        ;;
    train)
        model="${1:-}"
        mode="${2:-full}"
        case "${model}" in
            syn3|syn8) exec ./scripts/submit_syn.sh "${model}" "${mode}" ;;
            nfv) exec ./scripts/submit_nfv.sh "${mode}" ;;
            *) echo "train model must be syn3, syn8, or nfv" >&2; exit 2 ;;
        esac
        ;;
    diagnostic)
        name="${1:-}"
        case "${name}" in
            morphology|stageiv|forecast-hour|forecast_hour|inversion)
                exec ./scripts/submit_diagnostic.sh "${name}"
                ;;
            timelag)
                exec ./scripts/submit_timelag_experiment.sh
                ;;
            *) echo "unknown diagnostic: ${name}" >&2; exit 2 ;;
        esac
        ;;
    plot)
        name="${1:-}"
        case "${name}" in
            morphology) exec python scripts/plot/morphology.py ;;
            stageiv) exec python scripts/plot/stageiv_buckets.py ;;
            forecast-hour|forecast_hour) exec python scripts/plot/forecast_hour.py ;;
            *) echo "plot target must be morphology, stageiv, or forecast-hour" >&2; exit 2 ;;
        esac
        ;;
    *)
        echo "Unknown command: ${cmd}" >&2
        usage >&2
        exit 2
        ;;
esac
