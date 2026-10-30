#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import zarr

p=argparse.ArgumentParser()
p.add_argument('store')
a=p.parse_args()
path=Path(a.store)
if not path.is_file():
    raise SystemExit(f'Not found: {path}')
with zarr.ZipStore(str(path), mode='r') as zs:
    z=zarr.open_consolidated(zs, mode='r')
    ih=np.asarray(z['init_hour'][:]).astype(int)
    fh=np.asarray(z['forecast_hour'][:]).astype(int)
    av=np.asarray(z['available'][:], dtype=bool)
    q=z['qpf6']
    print('store:', path)
    print('qpf6 shape:', q.shape)
    print('forecast axis:', fh.tolist())
    rows=[]
    for i,h in enumerate(ih):
        for j,f in enumerate(fh):
            if av[i,j]: rows.append((int(h),int(f)))
    print('available pairs:', rows)
    expected={(c,f) for c in [0,6,12,18] for f in [6,12,18,24,30,36,42,48]}
    got=set(rows)
    print('n available:', len(got))
    print('missing expected:', sorted(expected-got))
    print('unexpected:', sorted(got-expected))
    if got:
        h,f=sorted(got)[0]
        field=np.asarray(q[h, f-6, :64, :64], dtype=np.float32)
        print(f'sample {h:02d}Z F{f:02d} 64x64 finite:', bool(np.isfinite(field).all()), 'min:', float(np.nanmin(field)), 'max:', float(np.nanmax(field)))
    if got != expected:
        raise SystemExit('STORE CHECK FAILED')
    print('STORE CHECK PASS')
