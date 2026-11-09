# HRRR archive audit + Stage IV ingestion

## 1. Audit the completed HRRR archive

This is intentionally a *lightweight but broad* audit.  It opens every daily
ZipStore, validates consolidated Zarr structure and metadata, checks grid/time
consistency, and reads one deterministic data patch from each day.

It does **not** run full ZIP CRC over 2.2 TiB by default.

```bash
source load_env.sh

python scripts/validate_hrrr_archive.py \
  --root "$DATA_DIR/forecasts/hrrr/qpf6_native" \
  --date-list "$DATA_DIR/metadata/date_lists/hrrr_20201202_20260922.txt" \
  --sample-size 64 \
  --report "$LOG_DIR/hrrr_archive_audit.txt"
```

Expected:
* 2,121 stores
* 661,752 logical fields
* 0 missing dates
* ideally 0 unavailable fields
* 0 errors

`--full-crc` exists but should **not** be used casually: `ZipFile.testzip()`
would reread essentially the whole archive.

## 2. Stage IV canonical raw layout

```text
$DATA_DIR/
└── observations/
    └── stage4/
        ├── raw/
        │   └── YYYY/
        │       └── YYYYMM/
        │           ├── st4_conus.YYYYMMDD00.06h.grb2
        │           ├── st4_conus.YYYYMMDD06.06h.grb2
        │           ├── st4_conus.YYYYMMDD12.06h.grb2
        │           └── st4_conus.YYYYMMDD18.06h.grb2
        └── metadata/
```

The original native GRIB2 files are retained.  No regridding is done during
ingestion.

## 3. One-day Stage IV test first

```bash
source load_env.sh
./scripts/stage4_extract_day.sh 20201202

ls -lh \
  "$DATA_DIR/observations/stage4/raw/2020/202012/"
```

The historical runhistory tar name is discovered automatically from these
known candidates:

```text
com_pcpanl_v4.1_pcpanl.YYYYMMDD.tar
com_pcpanl_prod_pcpanl.YYYYMMDD.tar
```

Only the four native six-hour members ending at 00/06/12/18 UTC are extracted.

## 4. Bulk Stage IV retrieval

Start conservatively because HPSS retrieval can be dominated by tape staging:

```bash
MAX_CONCURRENT=2 DAYS_PER_TASK=7 \
  ./scripts/submit_stage4_download.sh 20201202 20260922
```

This creates 303 array tasks rather than 2,121 one-day jobs.  Each task
continues to its remaining dates if one date fails, then exits nonzero with a
list of failed dates.

Completed files are skipped, so resubmitting the same range is safe.

## 5. Audit Stage IV after retrieval

```bash
python scripts/audit_stage4_raw.py \
  --root "$DATA_DIR/observations/stage4/raw" \
  --start 20201202 \
  --end 20260922
```

For a complete archive the expected count is:

```text
2,121 days × 4 synoptic analyses/day = 8,484 files
```

The audit checks that each file exists, begins with the GRIB magic bytes and
ends with the GRIB end marker, and creates:

```text
$DATA_DIR/observations/stage4/metadata/stage4_6h_manifest.csv.gz
```

## Important scientific choice

Stage IV remains in its native grid here.  Regridding Stage IV and HRRR onto a
common verification grid is a later methodological choice and is intentionally
not baked into the raw archive.

