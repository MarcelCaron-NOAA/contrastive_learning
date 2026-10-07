#!/usr/bin/env python3
"""Evaluate signed morphology probes for V2, retaining V1 learned similarity for comparison."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data import PrecipTransform
from src.models.checkpoint_loader import load_model_checkpoint
from src.eval.verification_metrics import continuous_and_wang_metrics
from src.eval.morphology_signed_transforms import SIGNED_LEVELS, transform_signed


def defaults():
    data=Path(os.environ["DATA_DIR"])
    return {
        "archive": data/"derived/evaluation/wang_transform_v1",
        "v1": data/"derived/experiments/groupnorm_pilot_v1/checkpoint_best.pt",
        "v2": data/"derived/experiments/hard_ranking_finetune_v1/checkpoint_best.pt",
        "stats": data/"metadata/dataset_v1/precip_normalization_stats_v1.json",
        "out": data/"derived/experiments/hard_ranking_finetune_v1/evaluation/signed_morphology_v1_vs_v2",
    }


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

@torch.no_grad()
def encode(model,raw,pt,device,bs):
    out=[]
    for s in range(0,len(raw),bs):
        x=torch.stack([pt(v) for v in raw[s:s+bs]],0).to(device,non_blocking=True)
        _,z=model(x); out.append(F.normalize(z.float(),dim=-1).cpu().numpy())
    return np.concatenate(out)

def ci95(v):
    x=np.asarray(v,float); x=x[np.isfinite(x)]
    if not len(x): return np.nan,np.nan,np.nan,0
    m=float(x.mean())
    if len(x)==1: return m,m,m,1
    se=float(x.std(ddof=1)/np.sqrt(len(x))); return m,m-1.96*se,m+1.96*se,len(x)

def total_change(orig,pert):
    a=float(np.sum(orig,dtype=np.float64)); b=float(np.sum(pert,dtype=np.float64))
    return np.nan if a==0 else 100*(b-a)/a

def step(c): return int(c.get("fine_tune_step",c.get("step",-1)))


def main():
    d=defaults(); ap=argparse.ArgumentParser()
    ap.add_argument("--archive-dir",default=str(d["archive"])); ap.add_argument("--baseline-checkpoint",default=str(d["v1"])); ap.add_argument("--checkpoint",default=str(d["v2"]))
    ap.add_argument("--stats-json",default=str(d["stats"])); ap.add_argument("--output-dir",default=str(d["out"]))
    ap.add_argument("--case-count",type=int,default=500); ap.add_argument("--case-seed",type=int,default=20261007); ap.add_argument("--batch-size",type=int,default=32); ap.add_argument("--threshold-mm",type=float,default=2.54)
    args=ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
    arc=Path(args.archive_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    allraw=np.load(arc/"original_patches.npz")["qpf6_mm"]; cases=pd.read_csv(arc/"cases.csv")
    rng=np.random.default_rng(args.case_seed); n=min(args.case_count,len(allraw)); idx=np.sort(rng.choice(len(allraw),size=n,replace=False)); raw=np.asarray(allraw[idx],np.float32)
    cases.iloc[idx].to_csv(out/"selected_cases.csv",index=False)
    dev=torch.device("cuda"); v1,c1=load_model_checkpoint(args.baseline_checkpoint,device=dev); v2,c2=load_model_checkpoint(args.checkpoint,device=dev); v1.eval(); v2.eval()
    pt=PrecipTransform.from_stats_json("log1p_zscore",args.stats_json); z10=encode(v1,raw,pt,dev,args.batch_size); z20=encode(v2,raw,pt,dev,args.batch_size)
    rec=[]
    for fam,levels in SIGNED_LEVELS.items():
        print(f"\n{fam}: {len(levels)} signed biases x {n} cases")
        for j,bias in enumerate(levels):
            trans=np.stack([transform_signed(raw[i],fam,bias,case_seed=args.case_seed+int(idx[i])) for i in range(n)]).astype(np.float32)
            z1=encode(v1,trans,pt,dev,args.batch_size); z2=encode(v2,trans,pt,dev,args.batch_size)
            s1=np.sum(z10*z1,axis=1); s2=np.sum(z20*z2,axis=1)
            for i,(orig,pert) in enumerate(zip(raw,trans)):
                m=continuous_and_wang_metrics(pert,orig,args.threshold_mm)
                rec.append({"archive_case_index":int(idx[i]),"case_position":i,"family":fam,"bias":float(bias),
                            "learned_similarity_z_v1":float(s1[i]),"learned_similarity_z_v2":float(s2[i]),"learned_similarity_z_delta_v2_minus_v1":float(s2[i]-s1[i]),
                            "total_precip_change_pct":total_change(orig,pert),**m})
            print(f"  {j+1:2d}/{len(levels)} bias={float(bias):g}")
    per=pd.DataFrame(rec); per.to_csv(out/"signed_morphology_per_case.csv.gz",index=False)
    cols=["learned_similarity_z_v1","learned_similarity_z_v2","ts","pod","normalized_far","fss5","fss10","corr","rmse","mae","total_precip_change_pct"]
    rows=[]
    for (fam,bias),g in per.groupby(["family","bias"],sort=False):
        r={"family":fam,"bias":bias,"n_rows":len(g)}
        for col in cols:
            m,lo,hi,nn=ci95(g[col]); r[f"{col}_mean"]=m; r[f"{col}_ci95_lo"]=lo; r[f"{col}_ci95_hi"]=hi; r[f"{col}_n"]=nn
        rows.append(r)
    pd.DataFrame(rows).to_csv(out/"signed_morphology_summary.csv",index=False)
    meta={"baseline_checkpoint":str(args.baseline_checkpoint),"baseline_sha256":sha(args.baseline_checkpoint),"baseline_step":step(c1),"v2_checkpoint":str(args.checkpoint),"v2_sha256":sha(args.checkpoint),"v2_step":step(c2),"case_count":n,"case_seed":args.case_seed,"threshold_mm":args.threshold_mm,"families":SIGNED_LEVELS,"sealed_test_used":False,
          "sign_semantics":{"smoothing_sharpening":"negative=unsharp enhancement; positive=Gaussian smoothing","rotation_deg":"negative/positive rotation direction","principal_axis_log2_scale":"negative=compress original precip major axis; positive=elongate it; perpendicular reciprocal scale","tile_fragmentation_cells":"negative reverses deterministic jitter vectors; both signs are equal-magnitude fragmentation","object_dispersion_cells":"negative=inward concentration; positive=outward dispersion"}}
    (out/"run_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(f"\nOutput: {out}")

if __name__=="__main__": main()

