#!/usr/bin/env python3
"""Run one shard of the HRRR +/-1, +/-3, +/-6 h lag-selection experiment."""
from __future__ import annotations
import argparse, csv, gzip, json, os, random, sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import zarr

PROJECT_DIR=Path(os.environ.get("PROJECT_DIR",Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(PROJECT_DIR))
from src.metrics.timelag_metrics import fss_column,fss_curve,scalar_metrics,wet_fraction

LAGS=(1,3,6)
POS={(1,"earlier_init"):13,(1,"later_init"):11,
     (3,"earlier_init"):15,(3,"later_init"):9,
     (6,"earlier_init"):18,(6,"later_init"):6}
NY,NX=1059,1799

def parse_windows(s):
    vals=tuple(int(x) for x in s.split(",") if x.strip())
    if not vals or any(v<1 or v%2==0 for v in vals): raise ValueError("FSS windows must be odd")
    return vals

def as_bool(s):
    return s if s.dtype==bool else s.astype(str).str.lower().isin(["true","1","t","yes"])

class HRRRPatchReader:
    def __init__(self,max_open=32): self.max_open=max_open; self.cache=OrderedDict()
    def _open(self,path):
        path=str(path)
        if path in self.cache:
            store,root=self.cache.pop(path); self.cache[path]=(store,root); return root
        store=zarr.ZipStore(path,mode="r"); root=zarr.open_consolidated(store,mode="r")
        self.cache[path]=(store,root)
        while len(self.cache)>self.max_open:
            _,(s,_)=self.cache.popitem(last=False); s.close()
        return root
    def patch(self,row,y0,x0,size):
        root=self._open(row["store"])
        return np.asarray(root["qpf6"][int(row["init_hour"]),int(row["forecast_hour"])-6,
                     y0:y0+size,x0:x0+size],dtype=np.float32)
    def close(self):
        for s,_ in self.cache.values(): s.close()

class Stage4Reader:
    def __init__(self,mapping):
        m=np.load(mapping)
        self.idx=np.asarray(m["stage4_flat_index"],dtype=np.int64)
        self.valid=np.asarray(m["valid_mapping"],dtype=bool)
        self.shape=tuple(int(x) for x in m["stage4_shape"])
    def read(self,path):
        ds=xr.open_dataset(path,engine="cfgrib",backend_kwargs={"indexpath":""})
        try:
            name=next((n for n in ["tp","APCP","apcp"] if n in ds.data_vars and ds[n].ndim==2),None)
            if name is None: name=next(n for n,d in ds.data_vars.items() if d.ndim==2)
            arr=np.asarray(ds[name].values,dtype=np.float32)
        finally: ds.close()
        if tuple(arr.shape)!=self.shape: raise RuntimeError(f"Stage-IV shape changed: {arr.shape}")
        out=np.array(arr.ravel()[self.idx].reshape(NY,NX),copy=True)
        out[~self.valid.reshape(NY,NX)]=np.nan
        return out

def stage4_path(root,valid):
    d=valid.strftime("%Y%m%d"); h=valid.strftime("%H")
    return root/d[:4]/d[:6]/f"st4_conus.{d}{h}.06h.grb2"

def add_metrics(row,a,b,thr,windows):
    row.update(scalar_metrics(a,b))
    row.update({fss_column(w):v for w,v in fss_curve(a,b,thr,windows).items()})
    return row

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True); p.add_argument("--stage4-root",required=True)
    p.add_argument("--mapping",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--shard-index",type=int,required=True); p.add_argument("--n-shards",type=int,required=True)
    p.add_argument("--patch-size",type=int,default=256); p.add_argument("--patches-per-valid",type=int,default=1)
    p.add_argument("--negative-count",type=int,default=3); p.add_argument("--negative-min-days",type=float,default=90)
    p.add_argument("--wet-threshold-mm",type=float,default=1.0); p.add_argument("--min-wet-fraction",type=float,default=0.01)
    p.add_argument("--patch-attempts",type=int,default=60); p.add_argument("--fss-threshold-mm",type=float,default=5.0)
    p.add_argument("--fss-windows",default="1,3,7,15,31,61"); p.add_argument("--min-stage4-finite-fraction",type=float,default=0.99)
    p.add_argument("--seed",type=int,default=20260926); p.add_argument("--max-valid-times",type=int,default=None)
    a=p.parse_args(); windows=parse_windows(a.fss_windows)
    rng=random.Random(a.seed+100003*a.shard_index)

    df=pd.read_csv(a.manifest); df["available_bool"]=as_bool(df["available"])
    df["valid_end_dt"]=pd.to_datetime(df["valid_end"],utc=True)
    avail=df.loc[df["available_bool"]].copy()
    idx={(r.valid_end_dt,int(r.forecast_hour)):pd.Series(r._asdict()) for r in avail.itertuples(index=False)}
    f12=avail.loc[avail["forecast_hour"]==12].sort_values("valid_end_dt").reset_index(drop=True)
    needed=[12]+sorted(set(POS.values()))
    # ------------------------------------------------------------------
    # Complete-case valid times: C=F12 plus all +/-1/3/6-h positives.
    #
    # IMPORTANT FOR PILOTS:
    # If --max-valid-times is supplied, do NOT take the first chronological
    # cases. Select a deterministic, seasonally balanced sample from the
    # entire archive first, then distribute that sample across shards.
    #
    # --max-valid-times retains its original user-facing meaning: number of
    # pilot valid times PER SHARD.
    # ------------------------------------------------------------------
    eligible = sorted([
        v for v in f12["valid_end_dt"]
        if all((v, fhr) in idx for fhr in needed)
    ])

    complete_case_total = len(eligible)

    if a.max_valid_times is not None:
        total_target = int(a.max_valid_times) * int(a.n_shards)

        season_names = ("DJF", "MAM", "JJA", "SON")

        def _season(month):
            if month in (12, 1, 2):
                return "DJF"
            if month in (3, 4, 5):
                return "MAM"
            if month in (6, 7, 8):
                return "JJA"
            return "SON"

        pilot = pd.DataFrame({"valid_time": pd.DatetimeIndex(eligible)})
        pilot["season"] = pilot["valid_time"].dt.month.map(_season)

        base = total_target // 4
        remainder = total_target % 4
        season_targets = {
            season: base + (1 if i < remainder else 0)
            for i, season in enumerate(season_names)
        }

        selection_rng = np.random.default_rng(a.seed)
        selected = []

        for season in season_names:
            vals = pilot.loc[
                pilot["season"] == season, "valid_time"
            ].to_numpy()

            n_take = min(season_targets[season], len(vals))
            if n_take < season_targets[season]:
                print(
                    f"WARNING: requested {season_targets[season]} {season} "
                    f"cases but only {len(vals)} are available."
                )

            take_idx = selection_rng.choice(
                len(vals),
                size=n_take,
                replace=False,
            )
            selected.extend(vals[take_idx].tolist())

        eligible = sorted(pd.to_datetime(selected, utc=True).tolist())

        counts = pd.Series(
            [_season(v.month) for v in eligible]
        ).value_counts()

        print(
            "Seasonally balanced pilot sample: "
            + ", ".join(
                f"{s}={int(counts.get(s, 0))}"
                for s in season_names
            )
        )

    valids = eligible[a.shard_index :: a.n_shards]

    print(f"Complete-case valid times total: {complete_case_total:,}")
    print(
        f"Selected valid times total     : {len(eligible):,}"
        + (
            " (seasonally balanced pilot)"
            if a.max_valid_times is not None
            else " (full archive)"
        )
    )

    outdir=Path(a.output_dir); (outdir/"shards").mkdir(parents=True,exist_ok=True)
    outpath=outdir/"shards"/f"metrics_shard_{a.shard_index:03d}.csv.gz"
    reader=HRRRPatchReader(); s4=Stage4Reader(a.mapping); s4root=Path(a.stage4_root)
    rows=[]; accepted=0; skipped=0; s4samples=0

    try:
        for iv,valid in enumerate(valids,1):
            c=idx[(valid,12)]
            samehour=f12.loc[f12["valid_end_dt"].dt.hour==valid.hour]
            days=np.abs((samehour["valid_end_dt"]-valid).dt.total_seconds())/86400
            negcand=samehour.loc[days>a.negative_min_days]
            if len(negcand)<a.negative_count: continue
            opath=stage4_path(s4root,valid); obs=None
            if opath.is_file():
                try: obs=s4.read(opath)
                except Exception as e: print(f"WARNING Stage-IV {opath}: {e}")

            for pi in range(a.patches_per_valid):
                ok=False
                for _ in range(a.patch_attempts):
                    y0=rng.randrange(0,NY-a.patch_size+1); x0=rng.randrange(0,NX-a.patch_size+1)
                    cp=reader.patch(c,y0,x0,a.patch_size)
                    if wet_fraction(cp,a.wet_threshold_mm)>=a.min_wet_fraction:
                        ok=True; break
                if not ok: skipped+=1; continue
                accepted+=1
                sid=f"{valid.strftime('%Y%m%d%H')}_y{y0:04d}_x{x0:04d}_p{pi}"
                negrows=[negcand.iloc[i] for i in rng.sample(range(len(negcand)),a.negative_count)]

                for ni,n in enumerate(negrows):
                    npatch=reader.patch(n,y0,x0,a.patch_size)
                    row={"sample_id":sid,"valid_time":valid.isoformat(),"patch_y":y0,"patch_x":x0,
                         "relation":"CN","lag_hours":"","direction":"","negative_index":ni,
                         "a_id":c["sample_id"],"b_id":n["sample_id"],"stage4_available":obs is not None}
                    rows.append(add_metrics(row,cp,npatch,a.fss_threshold_mm,windows))

                positives=[]
                for lag in LAGS:
                    for direction in ("earlier_init","later_init"):
                        pr=idx[(valid,POS[(lag,direction)])]
                        pp=reader.patch(pr,y0,x0,a.patch_size); positives.append((lag,direction,pr,pp))
                        row={"sample_id":sid,"valid_time":valid.isoformat(),"patch_y":y0,"patch_x":x0,
                             "relation":"CP","lag_hours":lag,"direction":direction,"negative_index":"",
                             "a_id":c["sample_id"],"b_id":pr["sample_id"],"stage4_available":obs is not None}
                        rows.append(add_metrics(row,cp,pp,a.fss_threshold_mm,windows))

                if obs is not None:
                    op=obs[y0:y0+a.patch_size,x0:x0+a.patch_size]
                    if float(np.mean(np.isfinite(op)))>=a.min_stage4_finite_fraction:
                        s4samples+=1
                        row={"sample_id":sid,"valid_time":valid.isoformat(),"patch_y":y0,"patch_x":x0,
                             "relation":"CO","lag_hours":"","direction":"","negative_index":"",
                             "a_id":c["sample_id"],"b_id":str(opath),"stage4_available":True}
                        rows.append(add_metrics(row,cp,op,a.fss_threshold_mm,windows))
                        for lag,direction,pr,pp in positives:
                            row={"sample_id":sid,"valid_time":valid.isoformat(),"patch_y":y0,"patch_x":x0,
                                 "relation":"PO","lag_hours":lag,"direction":direction,"negative_index":"",
                                 "a_id":pr["sample_id"],"b_id":str(opath),"stage4_available":True}
                            rows.append(add_metrics(row,pp,op,a.fss_threshold_mm,windows))
                        for ni,n in enumerate(negrows):
                            npatch=reader.patch(n,y0,x0,a.patch_size)
                            row={"sample_id":sid,"valid_time":valid.isoformat(),"patch_y":y0,"patch_x":x0,
                                 "relation":"NO","lag_hours":"","direction":"","negative_index":ni,
                                 "a_id":n["sample_id"],"b_id":str(opath),"stage4_available":True}
                            rows.append(add_metrics(row,npatch,op,a.fss_threshold_mm,windows))
            if iv%100==0 or iv==len(valids):
                print(f"{iv:,}/{len(valids):,} valid times; accepted={accepted:,}; rows={len(rows):,}")
    finally: reader.close()

    scalar=["rmse_mm","bcrmse_mm","mean_abs_diff_mm","mean_bias_b_minus_a_mm","corr",
            "mean_a_mm","mean_b_mm","max_a_mm","max_b_mm"]
    fields=["sample_id","valid_time","patch_y","patch_x","relation","lag_hours","direction",
            "negative_index","a_id","b_id","stage4_available"]+scalar+[fss_column(w) for w in windows]
    with gzip.open(outpath,"wt",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    meta={"shard_index":a.shard_index,"n_shards":a.n_shards,"complete_case_valid_times_total":len(eligible),
          "shard_valid_times":len(valids),"accepted_samples":accepted,"stage4_samples":s4samples,
          "skipped_no_wet_patch":skipped,"rows":len(rows),"fss_threshold_mm":a.fss_threshold_mm,
          "fss_windows":list(windows),"patch_size":a.patch_size,"patches_per_valid":a.patches_per_valid,
          "negative_count":a.negative_count,"negative_min_days":a.negative_min_days,
          "wet_threshold_mm":a.wet_threshold_mm,"min_wet_fraction":a.min_wet_fraction,"seed":a.seed}
    outpath.with_suffix("").with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n")
    print(f"Wrote {outpath}"); print(json.dumps(meta,indent=2))

if __name__=="__main__": main()

