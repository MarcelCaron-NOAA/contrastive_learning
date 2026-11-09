# HRRR QPF6 preprocessing

Canonical layout:

```text
$DATA_DIR/
├── forecasts/hrrr/qpf6_native/
│   ├── daily/YYYY/YYYYMM/hrrr_qpf6_YYYYMMDD.zarr.zip
│   └── metadata/hrrr_qpf6_manifest.csv.gz
├── observations/stage4/
│   ├── raw/
│   └── qpf6_native/
├── derived/
└── metadata/date_lists/
```

Each daily HRRR store contains `qpf6(init_hour=24, forecast_hour=13, y=1059, x=1799)` for F06-F18 on the native grid. Storage is float32. Chunks are `(1, 1, 256, 256)`, matching the planned training patch support while allowing arbitrary patch origins.

The store is a daily Zarr ZipStore so millions of logical Zarr chunks do not become millions of filesystem files. It is treated as write-once: build in a temporary file, write each chunk once, consolidate metadata, close, then atomically move into the canonical location.

## First test

```bash
source load_env.sh
MAX_CONCURRENT=1 ./scripts/submit_hrrr_preprocessing.sh 20210105 20210105
```

Then benchmark training-like random reads:

```bash
```

Do not launch the multi-year archive until this one-day test confirms acceptable runtime, file size, and random-patch read performance.

## Bulk example

```bash
MAX_CONCURRENT=12 ./scripts/submit_hrrr_preprocessing.sh 20201202 20260921
```

Choose the end date based on confirmed source availability.

## Manifest

```bash
python scripts/build_hrrr_manifest.py
```

Train/validation/test assignments are intentionally not baked into the canonical data. They can be created later from the manifest with whatever chronological split and gap we ultimately choose.
