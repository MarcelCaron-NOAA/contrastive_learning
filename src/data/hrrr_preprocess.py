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
FORECAST_HOURS_IN = np.arange(1, 19, dtype=np.int16)
FORECAST_HOURS_OUT = np.arange(6, 19, dtype=np.int16)
NY, NX = 1059, 1799
SPATIAL_CHUNK = 256
QPF_CHUNKS = (1, 1, SPATIAL_CHUNK, SPATIAL_CHUNK)
WET_THRESHOLDS_MM = (0.1, 1.0, 5.0, 10.0, 25.0)
SCHEMA_VERSION = "1.0"


@dataclass
class SourceCycle:
    qpf1h: np.ndarray
    x: np.ndarray
    y: np.ndarray
    source_dtype_bits: int
    source_path: str


def source_path(date: str, cycle: int) -> str:
    return f"{HRRR_SOURCE_BUCKET}/sfc/{date}/{date}_{cycle:02d}z_fcst.zarr"


def _open_source_once(fs: s3fs.S3FileSystem, date: str, cycle: int) -> SourceCycle:
    base = source_path(date, cycle)
    if not fs.exists(f"{base}/.zmetadata"):
        raise FileNotFoundError(base)

    store = s3fs.S3Map(root=base, s3=fs, check=False)
    root = zarr.open_consolidated(store, mode="r")
    group = root["surface"][SOURCE_VARIABLE]
    arr = group["surface"][SOURCE_VARIABLE]
    fp = np.asarray(group["forecast_period"][:])

    if arr.ndim != 3 or arr.shape[0] < 18 or arr.shape[1:] != (NY, NX):
        raise RuntimeError(f"{base}: unexpected APCP shape {arr.shape}")
    if fp.size < 18 or not np.array_equal(fp[:18], FORECAST_HOURS_IN):
        raise RuntimeError(f"{base}: unexpected F01-F18 periods {fp[:18]}")

    qpf1h = np.asarray(arr[:18], dtype=np.float32)
    x = np.asarray(group["projection_x_coordinate"][:], dtype=np.float64)
    y = np.asarray(group["projection_y_coordinate"][:], dtype=np.float64)
    if x.shape != (NX,) or y.shape != (NY,):
        raise RuntimeError(f"{base}: unexpected coordinate shapes x={x.shape}, y={y.shape}")

    return SourceCycle(
        qpf1h=qpf1h,
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


def derive_qpf6(qpf1h: np.ndarray) -> np.ndarray:
    if qpf1h.shape != (18, NY, NX):
        raise ValueError(f"Unexpected qpf1h shape {qpf1h.shape}")
    if not np.all(np.isfinite(qpf1h)):
        raise RuntimeError("Source APCP contains non-finite values")
    if float(np.min(qpf1h)) < -1e-6:
        raise RuntimeError(f"Source APCP contains negative value {float(np.min(qpf1h))}")

    prefix = np.empty((19, NY, NX), dtype=np.float32)
    prefix[0] = 0.0
    np.cumsum(qpf1h, axis=0, dtype=np.float32, out=prefix[1:])
    return np.asarray(prefix[6:19] - prefix[0:13], dtype=np.float32)


def _time_metadata(date: str):
    day = datetime.strptime(date, "%Y%m%d").replace(tzinfo=timezone.utc)
    init_time = np.empty(24, dtype=np.int64)
    valid_start = np.empty((24, 13), dtype=np.int64)
    valid_end = np.empty((24, 13), dtype=np.int64)
    for hour in range(24):
        init = day + timedelta(hours=hour)
        init_time[hour] = int(init.timestamp())
        for j, fhr in enumerate(FORECAST_HOURS_OUT):
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
        "dataset": "HRRR rolling 6-h accumulated precipitation",
        "model": "HRRR",
        "source_archive": "University of Utah/MesoWest public HRRR Zarr archive",
        "source_bucket": HRRR_SOURCE_BUCKET,
        "source_variable": SOURCE_VARIABLE,
        "derived_variable": "qpf6",
        "units": "mm",
        "accumulation_hours": 6,
        "forecast_hour_start": 6,
        "forecast_hour_end": 18,
        "spatial_grid": "native HRRR CONUS",
        "spatial_shape": [NY, NX],
        "storage_dtype": "float32",
        "spatial_chunk": SPATIAL_CHUNK,
        "date_utc": date,
        "derivation": "Sum six consecutive APCP_1hr_acc_fcst fields after casting source to float32.",
    })

    init_time, valid_start, valid_end = _time_metadata(date)
    _create(root, "init_hour", shape=(24,), chunks=(24,), dtype="i1",
            data=np.arange(24, dtype=np.int8))
    _create(root, "forecast_hour", shape=(13,), chunks=(13,), dtype="i2",
            data=FORECAST_HOURS_OUT, attrs={"units": "h"})
    _create(root, "init_time_unix", shape=(24,), chunks=(24,), dtype="i8",
            data=init_time, attrs={"units": "seconds since 1970-01-01T00:00:00Z"})
    _create(root, "valid_start_unix", shape=(24, 13), chunks=(24, 13), dtype="i8",
            data=valid_start, attrs={"units": "seconds since 1970-01-01T00:00:00Z"})
    _create(root, "valid_end_unix", shape=(24, 13), chunks=(24, 13), dtype="i8",
            data=valid_end, attrs={"units": "seconds since 1970-01-01T00:00:00Z"})

    x = _create(root, "projection_x_coordinate", shape=(NX,), chunks=(NX,),
                dtype="f8", fill_value=np.nan, attrs={"units": "m"})
    y = _create(root, "projection_y_coordinate", shape=(NY,), chunks=(NY,),
                dtype="f8", fill_value=np.nan, attrs={"units": "m"})

    qpf6 = _create(root, "qpf6", shape=(24, 13, NY, NX), chunks=QPF_CHUNKS,
                   dtype="f4", fill_value=np.nan, compressor=compressor,
                   attrs={"units": "mm", "dimensions": ["init_hour", "forecast_hour", "y", "x"]})

    meta = {}
    for name in ["mean_mm", "max_mm", "finite_fraction"]:
        meta[name] = _create(root, name, shape=(24, 13), chunks=(1, 13),
                             dtype="f4", fill_value=np.nan, compressor=compressor)
    for threshold in WET_THRESHOLDS_MM:
        key = f"wet_fraction_ge_{str(threshold).replace('.', 'p')}mm"
        meta[key] = _create(root, key, shape=(24, 13), chunks=(1, 13),
                            dtype="f4", fill_value=np.nan, compressor=compressor)

    available = _create(root, "available", shape=(24, 13), chunks=(1, 13),
                        dtype="bool", fill_value=False, compressor=compressor)
    source_dtype_bits = _create(root, "source_dtype_bits", shape=(24,), chunks=(1,),
                                dtype="u1", fill_value=0, compressor=compressor)

    return dict(root=root, x=x, y=y, qpf6=qpf6, meta=meta,
                available=available, source_dtype_bits=source_dtype_bits)


def write_cycle(out, hour: int, source: SourceCycle, qpf6: np.ndarray,
                grid_written: bool):
    if not grid_written:
        out["x"][...] = source.x
        out["y"][...] = source.y

    out["qpf6"][hour, :, :, :] = qpf6
    finite = np.isfinite(qpf6)
    out["meta"]["finite_fraction"][hour, :] = np.mean(finite, axis=(1, 2)).astype(np.float32)
    out["meta"]["mean_mm"][hour, :] = np.nanmean(qpf6, axis=(1, 2), dtype=np.float64).astype(np.float32)
    out["meta"]["max_mm"][hour, :] = np.nanmax(qpf6, axis=(1, 2)).astype(np.float32)
    for threshold in WET_THRESHOLDS_MM:
        key = f"wet_fraction_ge_{str(threshold).replace('.', 'p')}mm"
        out["meta"][key][hour, :] = np.mean(qpf6 >= threshold, axis=(1, 2)).astype(np.float32)
    out["available"][hour, :] = True
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
            for hour in range(24):
                print(f"\n[{date} {hour:02d}Z]")
                try:
                    source = read_source_cycle(fs, date, hour, retries=retries)
                except FileNotFoundError:
                    print("  source cycle not present; leaving unavailable")
                    missing_cycles.append(hour)
                    continue

                if first_x is None:
                    first_x, first_y = source.x, source.y
                elif not np.array_equal(source.x, first_x) or not np.array_equal(source.y, first_y):
                    raise RuntimeError(f"Grid changed within {date} at {hour:02d}Z")

                print(f"  source float{source.source_dtype_bits}; deriving F06-F18 QPF6")
                qpf6 = derive_qpf6(source.qpf1h)
                write_cycle(out, hour, source, qpf6, grid_written)
                grid_written = True
                available_cycles.append(hour)
                print(f"  max={float(np.max(qpf6)):.3f} mm mean={float(np.mean(qpf6)):.5f} mm")
                del qpf6, source

            if not available_cycles:
                raise RuntimeError(f"No source cycles available for {date}")
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
    print(f"Available cycles: {len(available_cycles)}/24")
    if missing_cycles:
        print("Missing cycles:", ",".join(f"{h:02d}" for h in missing_cycles))
    print("=" * 72)

