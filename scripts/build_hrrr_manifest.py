#!/usr/bin/env python3
from datetime import datetime, timezone
from pathlib import Path
import argparse, csv, gzip, os
import numpy as np
import zarr

STAT_VARS = ["mean_mm", "max_mm", "finite_fraction", "wet_fraction_ge_0p1mm", "wet_fraction_ge_1p0mm", "wet_fraction_ge_5p0mm", "wet_fraction_ge_10p0mm", "wet_fraction_ge_25p0mm"]
def iso(t):
    return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

p = argparse.ArgumentParser()
defroot = Path(os.environ.get("DATA_DIR", ".")) / "forecasts" / "hrrr" / "qpf6_native"
p.add_argument("--root", default=str(defroot)); p.add_argument("--output", default=None)
a = p.parse_args(); root = Path(a.root)
stores = sorted(root.glob("daily/*/*/hrrr_qpf6_*.zarr.zip"))
if not stores: raise SystemExit(f"No stores under {root}")
out = Path(a.output) if a.output else root / "metadata" / "hrrr_qpf6_manifest.csv.gz"
out.parent.mkdir(parents=True, exist_ok=True)
fields = ["sample_id","store","date","init_hour","forecast_hour","init_time","valid_start","valid_end","available","source_dtype_bits"] + STAT_VARS
rows = 0
with gzip.open(out, "wt", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for path in stores:
        with zarr.ZipStore(str(path), mode="r") as zs:
            z = zarr.open_consolidated(zs, mode="r")
            ih=np.asarray(z["init_hour"][:]); fh=np.asarray(z["forecast_hour"][:]); it=np.asarray(z["init_time_unix"][:]); vs=np.asarray(z["valid_start_unix"][:]); ve=np.asarray(z["valid_end_unix"][:]); av=np.asarray(z["available"][:]); bits=np.asarray(z["source_dtype_bits"][:]); stats={n:np.asarray(z[n][:]) for n in STAT_VARS}; date=str(z.attrs["date_utc"])
            for i,h in enumerate(ih):
                for j,l in enumerate(fh):
                    r={"sample_id":f"{date}{int(h):02d}_f{int(l):02d}","store":str(path),"date":date,"init_hour":int(h),"forecast_hour":int(l),"init_time":iso(it[i]),"valid_start":iso(vs[i,j]),"valid_end":iso(ve[i,j]),"available":bool(av[i,j]),"source_dtype_bits":int(bits[i])}
                    for n in STAT_VARS:
                        v=stats[n][i,j]; r[n]="" if not np.isfinite(v) else float(v)
                    w.writerow(r); rows += 1
        print(f"Indexed {path}")
print(f"Wrote {rows:,} rows to {out}")

