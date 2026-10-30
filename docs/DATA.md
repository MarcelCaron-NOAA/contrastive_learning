# Data layout and frozen snapshot

Large datasets are stored outside Git. Runtime locations come from the single
local `site_config.sh` file.

## Canonical local layout

```text
$DATA_DIR/
├── forecasts/
│   └── hrrr/
│       ├── qpf6_native/
│       │   ├── daily/YYYY/YYYYMM/hrrr_qpf6_YYYYMMDD.zarr.zip
│       │   └── metadata/hrrr_qpf6_manifest.csv.gz
│       └── qpf6_extended_synoptic/
│           ├── daily/YYYY/YYYYMM/hrrr_qpf6_YYYYMMDD.zarr.zip
│           └── metadata/hrrr_qpf6_manifest.csv.gz
├── observations/
├── metadata/
└── derived/
```

The HRRR products are daily Zarr-v2 stores packaged as ZIP files. The manifests
are the authoritative indices from logical samples to daily stores.

## Frozen NOAA/EMC snapshot (2026-10-30)

The group-readable internal HPSS location is:

```text
/NCEPDEV/emc-global/2year/marcel.caron/contrastive_learning
```

Set that path once in your untracked `site_config.sh`:

```bash
export HPSS_DATA_ROOT="/NCEPDEV/emc-global/2year/marcel.caron/contrastive_learning"
```

The directory contains:

```text
hrrr_qpf6_native_2020.tar
hrrr_qpf6_native_2021.tar
hrrr_qpf6_native_2022.tar
hrrr_qpf6_native_2023.tar
hrrr_qpf6_native_2024.tar
hrrr_qpf6_native_2025.tar
hrrr_qpf6_native_2026.tar
hrrr_qpf6_extended_2024.tar
hrrr_qpf6_extended_2025.tar
project_support_20261030.tar
```

Each HTAR archive also has a matching `.idx` file. At archive time, the native
manifest referenced **2,121/2,121** present daily stores and the extended
manifest referenced **357/357** present stores.

## Restore tiers

The helper restores directly into `$DATA_DIR` and preserves the archived
relative directory structure.

### Project support (~37 GB)

Contains `metadata/`, `observations/`, `derived/`, and the HRRR metadata trees.
This includes the final checkpoints and processed evaluation datasets, so it is
the normal starting point for reproducing final diagnostics without retraining.

```bash
./scripts/reproduce.sh restore support
```

### Native HRRR (~1.1 TB)

Required for retraining SYN-3, SYN-8, or NFV. Restore all years:

```bash
./scripts/reproduce.sh restore native
```

Or selected years:

```bash
./scripts/reproduce.sh restore native 2024 2025
```

The final training split spans the archived multi-year period, so exact
retraining requires the complete native set.

### Extended-synoptic HRRR (~24 GB)

Needed only to rebuild the forecast-hour archive from source products; it is not
required when the processed forecast-hour archive is already restored from
`project_support_20261030.tar`.

```bash
./scripts/reproduce.sh restore extended
```

### Everything

```bash
./scripts/reproduce.sh restore all
```

Use `DRY_RUN=1` with the restore helper to print HTAR extraction commands without
executing them.

## Preflight after restore

Processed/final diagnostic inputs:

```bash
source load_env.sh
./scripts/reproduce.sh preflight diagnostics
```

Training archive, including every unique store referenced by the manifest:

```bash
./scripts/reproduce.sh preflight training --verify-stores
```

## Stage IV note

The raw Stage IV GRIB2 download cache is **not** part of the irreplaceable
snapshot. The processed 1,000-case Stage-IV reference arrays and the final
forecast-hour evaluation archive are stored in `derived/` and are included in
`project_support_20261030.tar`. Raw Stage IV can be reacquired if either archive
must be rebuilt from source.

To rebuild the 1,000-case reference archive from raw Stage IV:

```bash
export STAGEIV_ROOT=/path/to/stage4/6h/files
export HRRR_GRID_TEMPLATE=/path/to/a/native/hrrr/grid/file
sbatch slurm/run_stageiv_reference_build.slurm
```

The data-rebuild utilities remain in the repository, but rebuilding public/raw
source data is deliberately separate from the normal frozen-snapshot
reproduction path.
