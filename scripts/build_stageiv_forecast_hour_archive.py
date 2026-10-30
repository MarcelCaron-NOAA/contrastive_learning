#!/usr/bin/env python3
"""Build a balanced raw HRRR/Stage-IV archive for lead-time verification.

Stage IV is BUDGET-remapped once per valid time.  One deterministic 256x256
patch is then chosen and reused for every HRRR lead verifying at that valid
time.  If a candidate crop has incomplete Stage-IV or HRRR coverage, additional
deterministic crop locations are tried.  The run fails rather than silently
losing a valid time, preserving identical sample populations at every lead.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import HRRRArchive


GRID_SHAPE = (1059, 1799)


def read_met_plane(path: Path, preferred_name: str = "APCP_06") -> np.ndarray:
    try:
        import netCDF4
    except ImportError as e:
        raise RuntimeError("netCDF4 Python package is required") from e

    with netCDF4.Dataset(path) as ds:
        if preferred_name in ds.variables:
            var = ds.variables[preferred_name]
        else:
            candidates = []
            for name, v in ds.variables.items():
                if name.lower() in {"lat", "lon", "latitude", "longitude", "time", "projection"}:
                    continue
                if len(v.dimensions) >= 2:
                    candidates.append((name, v))
            if not candidates:
                raise RuntimeError(f"No 2-D data field found in {path}; variables={list(ds.variables)}")
            candidates.sort(key=lambda z: (0 if "apcp" in z[0].lower() else 1, z[0]))
            var = candidates[0][1]
        arr = var[:]
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        arr = np.asarray(arr, dtype=np.float32).squeeze()

    if arr.shape != GRID_SHAPE:
        raise RuntimeError(f"MET output grid {arr.shape} != expected HRRR grid {GRID_SHAPE}")
    return arr


def run_regrid(exe: str, src: Path, grid_template: Path, output: Path, field: str, width: int) -> None:
    cmd = [
        exe, str(src), str(grid_template), str(output),
        "-field", field,
        "-method", "BUDGET",
        "-width", str(width),
        "-name", "APCP_06",
        "-v", "1",
    ]
    print("REGRID:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def candidate_coordinates(seed: int, patch_size: int, attempts: int):
    ny, nx = GRID_SHAPE
    rng = np.random.default_rng(int(seed))
    seen = set()
    for _ in range(int(attempts)):
        y0 = int(rng.integers(0, ny - patch_size + 1))
        x0 = int(rng.integers(0, nx - patch_size + 1))
        if (y0, x0) not in seen:
            seen.add((y0, x0))
            yield y0, x0


def main() -> None:
    data = Path(os.environ["DATA_DIR"])
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pair-manifest",
        default=str(data / "derived/evaluation/stageiv_forecast_hour_v1/pair_manifest.csv.gz"),
    )
    p.add_argument(
        "--hrrr-manifest",
        default=str(data / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"),
    )
    p.add_argument(
        "--output-dir",
        default=str(data / "derived/evaluation/stageiv_forecast_hour_v1"),
    )
    p.add_argument("--stageiv-root", default=str(data / "observations/stage4/raw"))
    p.add_argument("--hrrr-grid-template", required=True)
    p.add_argument("--regrid-data-plane", default="regrid_data_plane")
    p.add_argument("--stageiv-field", default='name="APCP"; level="A06";')
    p.add_argument("--regrid-width", type=int, default=2)
    p.add_argument("--patch-attempts", type=int, default=64)
    p.add_argument("--flip-y", action="store_true")
    p.add_argument("--flip-x", action="store_true")
    args = p.parse_args()

    exe = shutil.which(args.regrid_data_plane)
    if exe is None:
        raise SystemExit(f"Could not find {args.regrid_data_plane!r} on PATH")

    grid_template = Path(args.hrrr_grid_template)
    if not grid_template.is_file():
        raise SystemExit(f"HRRR grid template not found: {grid_template}")

    manifest_path = Path(args.pair_manifest)
    if not manifest_path.is_file():
        raise SystemExit(
            f"Pair manifest not found: {manifest_path}\n"
            "Run ./scripts/prepare_stageiv_forecast_hour_manifest.sh first."
        )

    pairs = pd.read_csv(manifest_path, low_memory=False)
    required = [
        "sample_id", "valid_id", "valid_end", "init_hour", "forecast_hour",
        "patch_seed", "patch_size", "stageiv_path",
    ]
    missing = [c for c in required if c not in pairs.columns]
    if missing:
        raise SystemExit(f"Pair manifest missing columns: {missing}")

    pairs["valid_end"] = pd.to_datetime(pairs["valid_end"], utc=True, errors="raise")
    pairs["forecast_hour"] = pd.to_numeric(pairs["forecast_hour"], errors="raise").astype(int)
    pairs["init_hour"] = pd.to_numeric(pairs["init_hour"], errors="raise").astype(int)
    pairs = pairs.sort_values(["valid_end", "forecast_hour"]).reset_index(drop=True)

    fhrs = sorted(pairs["forecast_hour"].unique().astype(int).tolist())
    per_valid = pairs.groupby("valid_id").size()
    if per_valid.nunique() != 1 or int(per_valid.iloc[0]) != len(fhrs):
        raise SystemExit("Pair manifest is not balanced: each valid time must contain every forecast hour")

    ps_values = pairs["patch_size"].unique()
    if len(ps_values) != 1:
        raise SystemExit("Pair manifest contains multiple patch sizes")
    ps = int(ps_values[0])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_cases = len(pairs)
    valid_ids = list(dict.fromkeys(pairs["valid_id"].astype(str).tolist()))
    n_valid = len(valid_ids)

    hrrr_build = out / "hrrr_qpf6_mm.building.npy"
    stageiv_build = out / "stageiv_budget_qpf6_mm.building.npy"
    for pth in [hrrr_build, stageiv_build]:
        pth.unlink(missing_ok=True)

    hrrr_mm = np.lib.format.open_memmap(hrrr_build, mode="w+", dtype="float32", shape=(n_cases, ps, ps))
    stageiv_mm = np.lib.format.open_memmap(stageiv_build, mode="w+", dtype="float32", shape=(n_valid, ps, ps))

    archive = HRRRArchive(args.hrrr_manifest)
    stageiv_root = Path(args.stageiv_root)
    case_rows = []
    valid_rows = []
    case_counter = 0

    try:
        with tempfile.TemporaryDirectory(prefix="stageiv_fhr_regrid_", dir=os.environ.get("SCRATCH_DIR")) as td:
            td = Path(td)
            groups = list(pairs.groupby("valid_id", sort=False))
            for valid_index, (valid_id, g) in enumerate(groups):
                g = g.sort_values("forecast_hour")
                valid = pd.Timestamp(g.iloc[0]["valid_end"])
                filename = f"st4_conus.{valid.strftime('%Y%m%d%H')}.06h.grb2"
                src = stageiv_root / valid.strftime("%Y") / valid.strftime("%Y%m") / filename
                if not src.is_file():
                    raise RuntimeError(f"Missing Stage-IV file during build: {src}")

                nc = td / f"{valid_id}.budget.nc"
                run_regrid(exe, src, grid_template, nc, args.stageiv_field, args.regrid_width)
                obs_full = read_met_plane(nc)
                nc.unlink(missing_ok=True)
                if args.flip_y:
                    obs_full = obs_full[::-1, :]
                if args.flip_x:
                    obs_full = obs_full[:, ::-1]

                seed = int(g.iloc[0]["patch_seed"])
                chosen = None
                chosen_obs = None
                chosen_fcsts = None

                for attempt, (y0, x0) in enumerate(candidate_coordinates(seed, ps, args.patch_attempts), start=1):
                    obs_patch = np.asarray(obs_full[y0:y0+ps, x0:x0+ps], dtype=np.float32)
                    if obs_patch.shape != (ps, ps) or not np.isfinite(obs_patch).all():
                        continue

                    fcsts = []
                    good = True
                    for row in g.itertuples():
                        field = archive.read_patch(str(row.sample_id), y0, x0, ps)
                        if field.shape != (ps, ps) or not np.isfinite(field).all():
                            good = False
                            break
                        fcsts.append(np.maximum(field, 0.0).astype(np.float32, copy=False))
                    if good:
                        chosen = (y0, x0, attempt)
                        chosen_obs = np.maximum(obs_patch, 0.0).astype(np.float32, copy=False)
                        chosen_fcsts = fcsts
                        break

                if chosen is None:
                    raise RuntimeError(
                        f"Could not find a complete shared {ps}x{ps} patch for valid={valid} "
                        f"after {args.patch_attempts} deterministic attempts. Increase --patch-attempts "
                        "or diagnose Stage-IV coverage; do not silently drop the valid time."
                    )

                y0, x0, patch_attempt = chosen
                stageiv_mm[valid_index] = chosen_obs

                valid_rows.append(
                    {
                        "stageiv_index": valid_index,
                        "valid_id": str(valid_id),
                        "valid_end": valid.isoformat(),
                        "valid_start": (valid - pd.Timedelta(hours=6)).isoformat(),
                        "y0": y0,
                        "x0": x0,
                        "patch_size": ps,
                        "patch_attempt": patch_attempt,
                        "stageiv_source_path": str(src),
                        "stageiv_mean_mm": float(np.mean(chosen_obs)),
                        "stageiv_max_mm": float(np.max(chosen_obs)),
                        "stageiv_wet_fraction_ge0p1": float(np.mean(chosen_obs >= 0.1)),
                    }
                )

                for fcst, row in zip(chosen_fcsts, g.itertuples()):
                    hrrr_mm[case_counter] = fcst
                    d = row._asdict()
                    d.update(
                        {
                            "case_index": case_counter,
                            "stageiv_index": valid_index,
                            "y0": y0,
                            "x0": x0,
                            "patch_attempt": patch_attempt,
                            "hrrr_mean_mm": float(np.mean(fcst)),
                            "hrrr_max_mm": float(np.max(fcst)),
                            "hrrr_wet_fraction_ge0p1": float(np.mean(fcst >= 0.1)),
                        }
                    )
                    case_rows.append(d)
                    case_counter += 1

                print(
                    f"valid {valid_index+1:4d}/{n_valid} | {valid} | patch=({y0},{x0}) "
                    f"attempt={patch_attempt} | cases={case_counter}/{n_cases}",
                    flush=True,
                )

        if case_counter != n_cases or len(valid_rows) != n_valid:
            raise RuntimeError(
                f"Internal count mismatch: cases {case_counter}/{n_cases}, "
                f"valids {len(valid_rows)}/{n_valid}"
            )

        hrrr_mm.flush()
        stageiv_mm.flush()
        del hrrr_mm, stageiv_mm

        final_hrrr = out / "hrrr_qpf6_mm.npy"
        final_obs = out / "stageiv_budget_qpf6_mm.npy"
        final_hrrr.unlink(missing_ok=True)
        final_obs.unlink(missing_ok=True)
        hrrr_build.rename(final_hrrr)
        stageiv_build.rename(final_obs)

        cases = pd.DataFrame(case_rows).sort_values("case_index")
        valids = pd.DataFrame(valid_rows).sort_values("stageiv_index")
        cases.to_csv(out / "cases.csv", index=False)
        valids.to_csv(out / "valid_times.csv", index=False)

        metadata = {
            "pair_manifest": str(manifest_path),
            "hrrr_manifest": str(args.hrrr_manifest),
            "stageiv_root": str(stageiv_root),
            "hrrr_grid_template": str(grid_template),
            "regrid_method": "BUDGET",
            "regrid_width": int(args.regrid_width),
            "stageiv_field": args.stageiv_field,
            "n_valid_times": int(n_valid),
            "n_cases": int(n_cases),
            "forecast_hours": fhrs,
            "patch_size": ps,
            "patch_attempts_allowed": int(args.patch_attempts),
            "flip_y": bool(args.flip_y),
            "flip_x": bool(args.flip_x),
            "stageiv_storage": "one patch per unique valid time; cases.csv maps HRRR case -> stageiv_index",
        }
        (out / "archive_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        print("=" * 88)
        print("ARCHIVE BUILD COMPLETE")
        print("=" * 88)
        print(f"Valid times : {n_valid:,}")
        print(f"HRRR cases  : {n_cases:,}")
        print(f"HRRR array  : {final_hrrr}")
        print(f"Stage IV    : {final_obs}")
        print(f"Cases       : {out / 'cases.csv'}")
    finally:
        archive.close()


if __name__ == "__main__":
    main()
