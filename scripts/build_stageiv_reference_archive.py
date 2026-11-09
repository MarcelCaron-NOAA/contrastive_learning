#!/usr/bin/env python3
"""
Build a self-contained HRRR/Stage-IV validation reference archive.

For each unique 6-h Stage-IV valid time:
  1. use MET regrid_data_plane to remap Stage IV onto the native HRRR grid,
     once with BUDGET and once with BILIN;
  2. extract the same 256x256 patch from HRRR and both Stage-IV remaps;
  3. require complete finite Stage-IV coverage for the patch;
  4. store raw precipitation fields in streaming .npy arrays.

The output is deliberately raw mm / 6 h. Model preprocessing is applied later
and remains identical for forecast and observation.

Important:
  * --hrrr-grid-template must be a gridded file on the exact native HRRR grid
    represented by the project's Zarr archive.
  * A one-time visual orientation/alignment check is required before trusting
    the archive.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import HRRRArchive


GRID_SHAPE = (1059, 1799)


def read_met_plane(path: Path, preferred_name="APCP_06"):
    """
    Robustly find a 2-D MET output data variable and return [y,x] float32.
    """
    try:
        import netCDF4
    except ImportError as e:
        raise RuntimeError("netCDF4 Python package is required.") from e

    with netCDF4.Dataset(path) as ds:
        if preferred_name in ds.variables:
            var = ds.variables[preferred_name]
        else:
            candidates = []
            for name, v in ds.variables.items():
                if name.lower() in {
                    "lat","lon","latitude","longitude","time","projection",
                }:
                    continue
                if len(v.dimensions) >= 2:
                    candidates.append((name, v))
            if not candidates:
                raise RuntimeError(
                    f"No plausible data variable found in {path}. "
                    f"Variables={list(ds.variables)}"
                )
            # Prefer APCP-like names, otherwise first field.
            candidates.sort(
                key=lambda x: (0 if "apcp" in x[0].lower() else 1, x[0])
            )
            var = candidates[0][1]

        arr = var[:]
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        arr = np.asarray(arr, dtype=np.float32).squeeze()

    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2-D regridded plane; got {arr.shape}")
    if arr.shape != GRID_SHAPE:
        raise RuntimeError(
            f"Regridded grid shape {arr.shape} != expected HRRR {GRID_SHAPE}. "
            "Confirm target grid template and array orientation."
        )
    return arr


def run_regrid(
    exe: str,
    stageiv_path: Path,
    hrrr_grid_template: Path,
    output_path: Path,
    method: str,
    field: str,
    width: int,
):
    cmd = [
        exe,
        str(stageiv_path),
        str(hrrr_grid_template),
        str(output_path),
        "-field", field,
        "-method", method,
        "-width", str(width),
        "-name", "APCP_06",
        "-v", "1",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    data = Path(os.environ["DATA_DIR"])
    p = argparse.ArgumentParser()
    p.add_argument(
        "--candidate-manifest",
        default=str(
            data / "derived/evaluation/stageiv_reference_v1/candidate_pairs.csv"
        ),
    )
    p.add_argument(
        "--hrrr-manifest",
        default=str(
            data / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
        ),
    )
    p.add_argument(
        "--output-dir",
        default=str(data / "derived/evaluation/stageiv_reference_v1"),
    )
    p.add_argument(
        "--hrrr-grid-template",
        required=True,
        help="Any gridded file defining the exact native HRRR grid.",
    )
    p.add_argument(
        "--regrid-data-plane",
        default="regrid_data_plane",
    )
    p.add_argument(
        "--stageiv-root",
        default=None,
        help=(
            "Optional override directory containing Stage-IV files. "
            "Useful if candidate manifest was made before staging data."
        ),
    )
    p.add_argument(
        "--stageiv-field",
        default='name="APCP"; level="A06";',
    )
    p.add_argument("--regrid-width", type=int, default=2)
    p.add_argument("--target-count", type=int, default=1000)
    p.add_argument(
        "--flip-y", action="store_true",
        help="Apply only if an explicit orientation check demonstrates need.",
    )
    p.add_argument(
        "--flip-x", action="store_true",
        help="Apply only if an explicit orientation check demonstrates need.",
    )
    args = p.parse_args()

    exe = shutil.which(args.regrid_data_plane)
    if exe is None:
        raise RuntimeError(
            f"Could not find {args.regrid_data_plane!r} on PATH."
        )

    hrrr_grid_template = Path(args.hrrr_grid_template)
    if not hrrr_grid_template.is_file():
        raise RuntimeError(
            f"HRRR grid template does not exist: {hrrr_grid_template}"
        )

    candidates = pd.read_csv(args.candidate_manifest)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    target = int(args.target_count)
    ps_values = candidates["patch_size"].unique()
    if len(ps_values) != 1:
        raise RuntimeError("candidate manifest contains multiple patch sizes")
    ps = int(ps_values[0])

    hrrr_mm = np.lib.format.open_memmap(
        out / "hrrr_qpf6_mm.npy",
        mode="w+", dtype="float32", shape=(target, ps, ps),
    )
    budget_mm = np.lib.format.open_memmap(
        out / "stageiv_budget_qpf6_mm.npy",
        mode="w+", dtype="float32", shape=(target, ps, ps),
    )
    bilin_mm = np.lib.format.open_memmap(
        out / "stageiv_bilin_qpf6_mm.npy",
        mode="w+", dtype="float32", shape=(target, ps, ps),
    )

    archive = HRRRArchive(args.hrrr_manifest)
    accepted_rows = []
    accepted = 0
    missing_files = 0
    incomplete_patches = 0

    # Process by valid time / Stage-IV file so regridding happens only once.
    groups = candidates.groupby("stageiv_filename", sort=False)

    with tempfile.TemporaryDirectory(
        prefix="stageiv_regrid_", dir=os.environ.get("SCRATCH_DIR", None)
    ) as tmp:
        tmp = Path(tmp)

        for filename, g in groups:
            if accepted >= target:
                break
            if args.stageiv_root:
                ts = pd.to_datetime(g.iloc[0]["valid_end"], utc=True)
                src = (
                    Path(args.stageiv_root)
                    / ts.strftime("%Y")
                    / ts.strftime("%Y%m")
                    / filename
                )
            else:
                src = Path(g.iloc[0]["stageiv_path"])

            if not src.is_file():
                missing_files += 1
                print(f"MISSING {src}")
                continue

            budget_nc = tmp / f"{filename}.budget.nc"
            bilin_nc = tmp / f"{filename}.bilin.nc"

            run_regrid(
                exe, src, hrrr_grid_template, budget_nc,
                "BUDGET", args.stageiv_field, args.regrid_width,
            )
            run_regrid(
                exe, src, hrrr_grid_template, bilin_nc,
                "BILIN", args.stageiv_field, args.regrid_width,
            )

            budget_full = read_met_plane(budget_nc)
            bilin_full = read_met_plane(bilin_nc)

            if args.flip_y:
                budget_full = budget_full[::-1, :]
                bilin_full = bilin_full[::-1, :]
            if args.flip_x:
                budget_full = budget_full[:, ::-1]
                bilin_full = bilin_full[:, ::-1]

            for row in g.itertuples():
                if accepted >= target:
                    break
                y0, x0, size = int(row.y0), int(row.x0), int(row.patch_size)

                b = np.asarray(
                    budget_full[y0:y0+size, x0:x0+size], dtype=np.float32
                )
                l = np.asarray(
                    bilin_full[y0:y0+size, x0:x0+size], dtype=np.float32
                )

                if (
                    b.shape != (size,size)
                    or l.shape != (size,size)
                    or not np.isfinite(b).all()
                    or not np.isfinite(l).all()
                ):
                    incomplete_patches += 1
                    continue

                f = archive.read_patch(row.sample_id, y0, x0, size)
                if not np.isfinite(f).all():
                    raise RuntimeError(
                        f"Non-finite HRRR patch: {row.sample_id}"
                    )

                hrrr_mm[accepted] = np.maximum(f, 0.0)
                budget_mm[accepted] = np.maximum(b, 0.0)
                bilin_mm[accepted] = np.maximum(l, 0.0)

                d = row._asdict()
                d["case_index"] = accepted
                d["stageiv_source_path"] = str(src)
                accepted_rows.append(d)
                accepted += 1

            budget_nc.unlink(missing_ok=True)
            bilin_nc.unlink(missing_ok=True)
            print(
                f"accepted={accepted}/{target} | "
                f"missing_files={missing_files} | "
                f"incomplete={incomplete_patches}"
            )

    hrrr_mm.flush(); budget_mm.flush(); bilin_mm.flush()
    archive.close()

    if accepted < target:
        raise RuntimeError(
            f"Only accepted {accepted}/{target} requested cases. "
            "The .npy arrays are allocated to target size and should not be "
            "used. Increase candidate count/stage more Stage-IV files and rerun."
        )

    cases = pd.DataFrame(accepted_rows)
    cases.to_csv(out / "cases.csv", index=False)

    metadata = {
        "target_count": target,
        "patch_size": ps,
        "hrrr_grid_shape": list(GRID_SHAPE),
        "stageiv_field": args.stageiv_field,
        "budget_method": "BUDGET",
        "bilinear_method": "BILIN",
        "regrid_width": args.regrid_width,
        "hrrr_grid_template": str(hrrr_grid_template),
        "flip_y": args.flip_y,
        "flip_x": args.flip_x,
        "missing_stageiv_file_groups": missing_files,
        "rejected_incomplete_patches": incomplete_patches,
        "alignment_status": "REQUIRES_MANUAL_VISUAL_CHECK_BEFORE_ANALYSIS",
    }
    import json
    (out / "archive_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print("\nARCHIVE BUILT")
    print(f"Cases: {accepted}")
    print(f"Path : {out}")
    print(
        "REQUIRED NEXT STEP: visually inspect several HRRR/Stage-IV patches "
        "for orientation/alignment before running scientific analysis."
    )


if __name__ == "__main__":
    main()

