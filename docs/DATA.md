# Data layout and archived snapshot

Large datasets are stored outside Git. All runtime paths are derived from the
local `site_config.sh` loaded by `load_env.sh`.

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

The HRRR stores are daily Zarr-v2 archives packaged as ZIP files. The manifest
is the authoritative index from logical samples to daily stores.

## Frozen NOAA/EMC snapshot (2026-10-30)

An internal snapshot was archived to the group-readable HPSS location:

```text
/NCEPDEV/emc-global/2year/marcel.caron/contrastive_learning
```

The snapshot contains:

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

Each HTAR archive has a matching `.idx` file. Permissions are owner read/write,
`emc-global` group read, and no world access.

At archive time the authoritative HRRR manifests referenced:

- 2,121 / 2,121 present native daily stores;
- 357 / 357 present extended-synoptic daily stores.

The project-support archive contains the smaller `metadata/`, `observations/`,
`derived/`, and HRRR metadata trees.

## Stage IV note

The raw Stage IV GRIB2 download cache is not considered part of the irreplaceable
snapshot. The 1,000-case processed evaluation arrays and their case metadata are
preserved under `derived/evaluation/stageiv_reference_v1/`. Raw Stage IV inputs
can be reacquired when needed.

## Rebuilding the processed Stage IV reference archive

The frozen project-support archive already contains the validated 1,000-case
Stage-IV reference dataset. If it must be rebuilt from raw Stage IV files, use:

```bash
export STAGEIV_ROOT=/path/to/stage4/6h/files
export HRRR_GRID_TEMPLATE=/path/to/a/native/hrrr/grid/file
sbatch slurm/run_stageiv_reference_build.slurm
```

This rebuilds the case manifest and the HRRR/Stage-IV 256x256 arrays under
`$DATA_DIR/derived/evaluation/stageiv_reference_v1/`. Raw Stage IV is treated as
reacquirable source data rather than a permanent repository asset.

## Restoring data

Restore only the years/products required for the experiment you intend to run.
For example, inspect an archive first with HTAR before extracting it. Exact
restore commands are intentionally kept separate from model code so the same
repository can work with either restored or freshly preprocessed data.

After restoring, verify that `site_config.sh` points `DATA_DIR` at the restored
root and run the repository preflight/audit tools before training.
