#!/usr/bin/env python3
import argparse, random, statistics, time
import numpy as np
import zarr

p = argparse.ArgumentParser()
p.add_argument("store")
p.add_argument("--reads", type=int, default=200)
p.add_argument("--patch", type=int, default=256)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
rng = random.Random(a.seed)

with zarr.ZipStore(a.store, mode="r") as zs:
    z = zarr.open_consolidated(zs, mode="r")
    qpf = z["qpf6"]
    avail = np.asarray(z["available"][:])
    pairs = list(zip(*np.where(avail)))
    if not pairs:
        raise RuntimeError("No available fields")
    _, _, ny, nx = qpf.shape
    times = []
    i, j = rng.choice(pairs)
    y0 = rng.randrange(0, ny-a.patch+1); x0 = rng.randrange(0, nx-a.patch+1)
    _ = qpf[i, j, y0:y0+a.patch, x0:x0+a.patch]
    for _ in range(a.reads):
        i, j = rng.choice(pairs)
        y0 = rng.randrange(0, ny-a.patch+1); x0 = rng.randrange(0, nx-a.patch+1)
        t0 = time.perf_counter()
        patch = qpf[i, j, y0:y0+a.patch, x0:x0+a.patch]
        _ = float(np.nanmean(patch))
        times.append(time.perf_counter()-t0)

logical_mib = a.patch*a.patch*4/1024**2
s = sorted(times)
p50 = statistics.median(s)
p95 = s[max(0, int(0.95*len(s))-1)]
mean = statistics.mean(s)
print(f"Reads              : {a.reads}")
print(f"Patch              : {a.patch}x{a.patch}")
print(f"Mean read time     : {mean*1000:.2f} ms")
print(f"Median read time   : {p50*1000:.2f} ms")
print(f"95th percentile    : {p95*1000:.2f} ms")
print(f"Logical throughput : {logical_mib/mean:.2f} MiB/s")

