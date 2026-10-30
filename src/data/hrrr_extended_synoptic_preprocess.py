#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
import time
import zipfile

import numpy as np
import s3fs
import zarr
from numcodecs import Blosc

HRRR_SOURCE_BUCKET = "hrrrzarr"
SOURCE_VARIABLE = "APCP_1hr_acc_fcst"
SOURCE_FORECAST_HOURS = np.arange(1, 49, dtype=np.int16)
FORECAST_HOURS_AXIS = np.arange(6, 49, dtype=np.int16)  # contiguous for HRRRArchive indexing
REQUESTED_FORECAST_HOURS = np.array([6, 12, 18, 24, 30, 36, 42, 48], dtype=np.int16)
SYNOPTIC_CYCLES = np.array([0, 6, 12, 18], dtype=np.int8)
NY, NX = 1059, 1799
SPATIAL_CHUNK = 256
QPF_CHUNKS = (1, 1, SPATIAL_CHUNK, SPATIAL_CHUNK)
WET_THRESHOLDS_MM = (0.1, 1.0, 5.0, 10.0, 25.0)
SCHEMA_VERSION = "1.0"


@dataclass
class SourceCycle:
    qpf6_by_fhr: dict[int, np.ndarray]
    x: np.ndarray
    y: np.ndarray
    source_dtype_bits: int
    source_path: str


def source_path(date: str, cycle: int) -> str:
    return f"{HRRR_SOURCE_BUCKET}/sfc/{date}/{date}_{cycle:02d}z_fcst.zarr"


def _derive_one_qpf6(arr, fhr: int) -> np.ndarray:
    # 6-h accumulation ending at fhr: F(fhr-5) ... F(fhr)
    s = int(fhr) - 6
    e = int(fhr)
    raw = np.asarray(arr[s:e, :, :], dtype=np.float32)
    if raw.shape != (6, NY, NX):
        raise RuntimeError(f"Unexpected source slice shape for F{fhr:02d}: {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise RuntimeError(f"Source APCP contains non-finite values for F{fhr:02d}")
    if float(np.min(raw)) < -1e-6:
        raise RuntimeError(f"Source APCP contains negative values for F{fhr:02d}: {float(np.min(raw))}")
    return np.sum(raw, axis=0, dtype=np.float32)


def _open_source_once(fs: s3fs.S3FileSystem, date: str, cycle: int) -> SourceCycle:
    base = source_path(date, cycle)
    if not fs.exists(f"{base}/.zmetadata"):
        raise FileNotFoundError(base)

    store = s3fs.S3Map(root=base, s3=fs, check=False)
    root = zarr.open_consolidated(store, mode="r")
    group = root["surface"][SOURCE_VARIABLE]
    arr = group["surface"][SOURCE_VARIABLE]
    fp = np.asarray(group["forecast_period"][:])

    if arr.ndim != 3 or arr.shape[0] < 48 or arr.shape[1:] != (NY, NX):
        raise RuntimeError(f"{base}: unexpected APCP shape {arr.shape}")
    if fp.size < 48 or not np.array_equal(fp[:48], SOURCE_FORECAST_HOURS):
        raise RuntimeError(f"{base}: unexpected F01-F48 periods {fp[:48]}")

    x = np.asarray(group["projection_x_coordinate"][:], dtype=np.float64)
    y = np.asarray(group["projection_y_coordinate"][:], dtype=np.float64)
    if x.shape != (NX,) or y.shape != (NY,):
        raise RuntimeError(f"{base}: unexpected coordinate shapes x={x.shape}, y={y.shape}")

    qpf6 = {}
    for fhr in REQUESTED_FORECAST_HOURS:
        qpf6[int(fhr)] = _derive_one_qpf6(arr, int(fhr))

    return SourceCycle(
        qpf6_by_fhr=qpf6,
        x=x,
        y=y,
        source_dtype_bits=np.dtype(arr.dtype).itemsize * 8,
        source_path=base,
    )


def read_source_cycle(fs, date: str, cycle: int, retries: int = 3) -> SourceCycle:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return _open_source_once(fs, date, cycle)
        except FileNotFoundError:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            wait = 5.0 * (2 ** (attempt - 1))
            print(f"  source read failed attempt {attempt}/{retries}: {exc}")
            print(f"  retrying in {wait:.0f} s")
            time.sleep(wait)
    raise RuntimeError(f"Failed source read: {date} {cycle:02d}Z") from last_exc


def _time_metadata(date: str):
    day = datetime.strptime(date, "%Y%m%d").replace(tzinfo=timezone.utc)
    init_time = np.empty(24, dtype=np.int64)
    valid_start = np.empty((24, len(FORECAST_HOURS_AXIS)), dtype=np.int64)
    valid_end = np.empty((24, len(FORECAST_HOURS_AXIS)), dtype=np.int64)
    for hour in range(24):
        init = day + timedelta(hours=hour)
        init_time[hour] = int(init.timestamp())
        for j, fhr in enumerate(FORECAST_HOURS_AXIS):
            end = init + timedelta(hours=int(fhr))
            valid_start[hour, j] = int((end - timedelta(hours=6)).timestamp())
            valid_end[hour, j] = int(end.timestamp())
    return init_time, valid_start, valid_end


def _create(root, name, *, shape, chunks, dtype, data=None, fill_value=None,
            compressor=None, attrs=None):
    kwargs = dict(name=name, shape=shape, chunks=chunks, dtype=dtype)
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    if compressor is not None:
        kwargs["compressor"] = compressor
    arr = root.create_dataset(**kwargs)
    if attrs:
        arr.attrs.update(attrs)
    if data is not None:
        arr[...] = data
    return arr


def create_daily_store(store, date: str):
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    root = zarr.group(store=store)
    root.attrs.update({
        "schema_version": SCHEMA_VERSION,
        "dataset": "HRRR rolling 6-h accumulated precipitation, extended synoptic subset",
        "model": "HRRR",
        "source_archive": "University of Utah/MesoWest public HRRR Zarr archive",
        "source_bucket": HRRR_SOURCE_BUCKET,
        "source_variable": SOURCE_VARIABLE,
        "derived_variable": "qpf6",
        "units": "mm",
        "accumulation_hours": 6,
        "forecast_hour_axis_start": 6,
        "forecast_hour_axis_end": 48,
        "stored_forecast_hours": REQUESTED_FORECAST_HOURS.astype(int).tolist(),
        "stored_init_hours": SYNOPTIC_CYCLES.astype(int).tolist(),
        "spatial_grid": "native HRRR CONUS",
        "spatial_shape": [NY, NX],
        "storage_dtype": "float32",
        "spatial_chunk": SPATIAL_CHUNK,
        "date_utc": date,
        "derivation": "For requested leads, sum the six preceding APCP_1hr_acc_fcst fields after float32 conversion.",
        "note": "Forecast-hour axis is contiguous F06-F48 so the existing HRRRArchive reader can index by fhr-6; only F06/F12/.../F48 chunks are physically written.",
    })

    init_time, valid_start, valid_end = _time_metadata(date)
    nf = len(FORECAST_HOURS_AXIS)
    _create(root, "init_hour", shape=(24,), chunks=(24,), dtype="i1",
            data=np.arange(24, dtype=np.int8))
    _create(root, "forecast_hour", shape=(nf,), chunks=(nf,), dtype="i2",
            data=FORECAST_HOURS_AXIS, attrs={"units": "h"})
    _create(root, "init_time_unix", shape=(24,), chunks=(24,), dtype="i8",
            data=init_time, attrs={"units": "seconds since 1970-01-01T00:00:00Z"})
    _create(root, "valid_start_unix", shape=(24, nf), chunks=(24, nf), dtype="i8",
            data=valid_start, attrs={"units": "seconds since 1970-01-01T00:00:00Z"})
    _create(root, "valid_end_unix", shape=(24, nf), chunks=(24, nf), dtype="i8",
            data=valid_end, attrs={"units": "seconds since 1970-01-01T00:00:00Z"})

    x = _create(root, "projection_x_coordinate", shape=(NX,), chunks=(NX,),
                dtype="f8", fill_value=np.nan, attrs={"units": "m"})
    y = _create(root, "projection_y_coordinate", shape=(NY,), chunks=(NY,),
                dtype="f8", fill_value=np.nan, attrs={"units": "m"})

    qpf6 = _create(root, "qpf6", shape=(24, nf, NY, NX), chunks=QPF_CHUNKS,
                   dtype="f4", fill_value=np.nan, compressor=compressor,
                   attrs={"units": "mm", "dimensions": ["init_hour", "forecast_hour", "y", "x"]})

    meta = {}
    for name in ["mean_mm", "max_mm", "finite_fraction"]:
        meta[name] = _create(root, name, shape=(24, nf), chunks=(1, nf),
                             dtype="f4", fill_value=np.nan, compressor=compressor)
    for threshold in WET_THRESHOLDS_MM:
        key = f"wet_fraction_ge_{str(threshold).replace('.', 'p')}mm"
        meta[key] = _create(root, key, shape=(24, nf), chunks=(1, nf),
                            dtype="f4", fill_value=np.nan, compressor=compressor)

    available = _create(root, "available", shape=(24, nf), chunks=(1, nf),
                        dtype="bool", fill_value=False, compressor=compressor)
    source_dtype_bits = _create(root, "source_dtype_bits", shape=(24,), chunks=(1,),
                                dtype="u1", fill_value=0, compressor=compressor)

    return dict(root=root, x=x, y=y, qpf6=qpf6, meta=meta,
                available=available, source_dtype_bits=source_dtype_bits)


def write_cycle(out, hour: int, source: SourceCycle, grid_written: bool):
    if not grid_written:
        out["x"][...] = source.x
        out["y"][...] = source.y

    for fhr, field in source.qpf6_by_fhr.items():
        j = int(fhr) - int(FORECAST_HOURS_AXIS[0])
        q = np.asarray(field, dtype=np.float32)
        out["qpf6"][hour, j, :, :] = q
        finite = np.isfinite(q)
        out["meta"]["finite_fraction"][hour, j] = np.mean(finite).astype(np.float32)
        out["meta"]["mean_mm"][hour, j] = np.nanmean(q, dtype=np.float64).astype(np.float32)
        out["meta"]["max_mm"][hour, j] = np.nanmax(q).astype(np.float32)
        for threshold in WET_THRESHOLDS_MM:
            key = f"wet_fraction_ge_{str(threshold).replace('.', 'p')}mm"
            out["meta"][key][hour, j] = np.mean(q >= threshold).astype(np.float32)
        out["available"][hour, j] = True

    out["source_dtype_bits"][hour] = source.source_dtype_bits


def process_day(date: str, output_path: Path, temp_path: Path,
                overwrite: bool = False, retries: int = 3):
    output_path, temp_path = Path(output_path), Path(temp_path)
    if output_path.exists() and not overwrite:
        print(f"Output already exists; skipping: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_path.exists():
        temp_path.unlink()

    fs = s3fs.S3FileSystem(anon=True)
    available_cycles, missing_cycles = [], []
    grid_written = False
    first_x = first_y = None

    try:
        with zarr.ZipStore(str(temp_path), mode="w", compression=zipfile.ZIP_STORED,
                           allowZip64=True) as store:
            out = create_daily_store(store, date)
            for hour in SYNOPTIC_CYCLES.astype(int):
                print(f"\n[{date} {hour:02d}Z]", flush=True)
                try:
                    source = read_source_cycle(fs, date, hour, retries=retries)
                except FileNotFoundError:
                    print("  source cycle not present; leaving unavailable", flush=True)
                    missing_cycles.append(hour)
                    continue

                if first_x is None:
                    first_x, first_y = source.x, source.y
                elif not np.array_equal(source.x, first_x) or not np.array_equal(source.y, first_y):
                    raise RuntimeError(f"Grid changed within {date} at {hour:02d}Z")

                print("  derived QPF6 at F06,F12,F18,F24,F30,F36,F42,F48", flush=True)
                write_cycle(out, hour, source, grid_written)
                grid_written = True
                available_cycles.append(hour)
                maxv = max(float(np.max(v)) for v in source.qpf6_by_fhr.values())
                meanv = float(np.mean([np.mean(v, dtype=np.float64) for v in source.qpf6_by_fhr.values()]))
                print(f"  max={maxv:.3f} mm mean-across-leads={meanv:.5f} mm", flush=True)
                del source

            if not available_cycles:
                raise RuntimeError(f"No requested source cycles available for {date}")
            zarr.consolidate_metadata(store)

        if output_path.exists():
            if overwrite:
                output_path.unlink()
            else:
                raise FileExistsError(output_path)
        os.replace(temp_path, output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    size_gib = output_path.stat().st_size / 1024**3
    print("\n" + "=" * 72)
    print(f"Completed {date}")
    print(f"Output: {output_path}")
    print(f"Size: {size_gib:.3f} GiB")
    print(f"Available requested cycles: {len(available_cycles)}/4")
    if missing_cycles:
        print("Missing cycles:", ",".join(f"{h:02d}" for h in missing_cycles))
    print("=" * 72)
