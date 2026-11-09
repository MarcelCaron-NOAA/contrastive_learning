#!/usr/bin/env bash

set -euo pipefail

DATE="${1:?Usage: $0 YYYYMMDD}"

if [[ ! "${DATE}" =~ ^[0-9]{8}$ ]]; then
    echo "DATE must be YYYYMMDD" >&2
    exit 2
fi

: "${DATA_DIR:?Run: source load_env.sh}"
: "${SCRATCH_DIR:?Run: source load_env.sh}"

YEAR="${DATE:0:4}"
YYYYMM="${DATE:0:6}"

OUTDIR="${DATA_DIR}/observations/stage4/raw/${YEAR}/${YYYYMM}"
WORKDIR="${SCRATCH_DIR}/stage4/${DATE}.$$"

mkdir -p "${OUTDIR}" "${WORKDIR}"

cleanup() {
    rm -rf "${WORKDIR}"
}
trap cleanup EXIT

# Skip dates already completed.
complete=1

for HH in 00 06 12 18; do
    FILE="${OUTDIR}/st4_conus.${DATE}${HH}.06h.grb2"

    if [[ ! -s "${FILE}" ]]; then
        complete=0
        break
    fi
done

if [[ "${complete}" == 1 ]]; then
    echo "${DATE}: all four Stage IV files already exist; skipping"
    exit 0
fi

if ! command -v htar >/dev/null 2>&1; then
    module load hpss >/dev/null 2>&1 || true
fi

if ! command -v htar >/dev/null 2>&1; then
    echo "ERROR: htar is unavailable" >&2
    exit 3
fi

BASE="/NCEPPROD/hpssprod/runhistory/rh${YEAR}/${YYYYMM}/${DATE}"

CANDIDATES=(
    "${BASE}/com_pcpanl_v4.1_pcpanl.${DATE}.tar"
    "${BASE}/com_pcpanl_prod_pcpanl.${DATE}.tar"
)

MEMBERS=(
    "./st4_conus.${DATE}00.06h.grb2"
    "./st4_conus.${DATE}06.06h.grb2"
    "./st4_conus.${DATE}12.06h.grb2"
    "./st4_conus.${DATE}18.06h.grb2"
)

ARCHIVE=""

for CAND in "${CANDIDATES[@]}"; do
    echo "Checking:"
    echo "  ${CAND}"

    if htar -tf "${CAND}" > "${WORKDIR}/index.txt" 2>/dev/null; then
        if grep -q "st4_conus.${DATE}00.06h.grb2" "${WORKDIR}/index.txt"; then
            ARCHIVE="${CAND}"
            break
        fi
    fi
done

if [[ -z "${ARCHIVE}" ]]; then
    echo "ERROR: Could not locate Stage IV archive for ${DATE}" >&2
    exit 4
fi

echo "Using:"
echo "  ${ARCHIVE}"

cd "${WORKDIR}"

htar -xvf "${ARCHIVE}" "${MEMBERS[@]}"

for HH in 00 06 12 18; do

    SRC="${WORKDIR}/st4_conus.${DATE}${HH}.06h.grb2"
    DST="${OUTDIR}/st4_conus.${DATE}${HH}.06h.grb2"

    if [[ ! -s "${SRC}" ]]; then
        echo "ERROR: Missing extracted file ${SRC}" >&2
        exit 5
    fi

    MAGIC="$(head -c 4 "${SRC}")"

    if [[ "${MAGIC}" != "GRIB" ]]; then
        echo "ERROR: ${SRC} does not begin with GRIB" >&2
        exit 6
    fi

    mv "${SRC}" "${DST}"
done

echo
echo "${DATE}: Stage IV extraction complete"
ls -lh "${OUTDIR}"/st4_conus."${DATE}"*.06h.grb2
