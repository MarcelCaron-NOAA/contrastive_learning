#!/usr/bin/env python3
"""Compare V1 and hard-ranking V2 on the existing HRRR-vs-Stage-IV validation archive."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.data import PrecipTransform
from src.models.checkpoint_loader import load_model_checkpoint
from src.eval.verification_metrics import mae, rmse, pearson_r, fss


def defaults():
    data=Path(os.environ["DATA_DIR"])
    return {"archive":data/"derived/evaluation/stageiv_reference_v1","v1":data/"derived/experiments/groupnorm_pilot_v1/checkpoint_best.pt","v2":data/"derived/experiments/hard_ranking_finetune_v1/checkpoint_best.pt","stats":data/"metadata/dataset_v1/precip_normalization_stats_v1.json","out":data/"derived/experiments/hard_ranking_finetune_v1/evaluation/stageiv_v1_vs_v2"}

def sha(p):
    h=hashlib.sha256();
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()
@torch.no_grad()
def encode(model,raw,pt,dev,bs):
    out=[]
    for s in range(0,len(raw),bs):
        x=torch.stack([pt(v) for v in raw[s:s+bs]],0).to(dev,non_blocking=True); _,z=model(x); out.append(F.normalize(z.float(),dim=-1).cpu().numpy())
    return np.concatenate(out)
def spearman(x,y): return float(pd.Series(x).rank(method="average").corr(pd.Series(y).rank(method="average")))
def step(c): return int(c.get("fine_tune_step",c.get("step",-1)))
def quantiles(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; qs=[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]; vals=np.quantile(x,qs); return {f"q{int(q*100):02d}":float(v) for q,v in zip(qs,vals)}
def savefig(path): plt.tight_layout(); plt.savefig(path,dpi=200,bbox_inches="tight"); plt.close()


def main():
    d=defaults(); ap=argparse.ArgumentParser()
    ap.add_argument("--archive-dir",default=str(d["archive"])); ap.add_argument("--baseline-checkpoint",default=str(d["v1"])); ap.add_argument("--checkpoint",default=str(d["v2"])); ap.add_argument("--stats-json",default=str(d["stats"])); ap.add_argument("--output-dir",default=str(d["out"])); ap.add_argument("--batch-size",type=int,default=32); ap.add_argument("--fss-threshold",type=float,default=1.0); ap.add_argument("--fss-width",type=int,default=15)
    args=ap.parse_args();
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
    arc=Path(args.archive_dir); out=Path(args.output_dir); plots=out/"plots"; out.mkdir(parents=True,exist_ok=True); plots.mkdir(parents=True,exist_ok=True)
    cases=pd.read_csv(arc/"cases.csv"); fcst=np.load(arc/"hrrr_qpf6_mm.npy",mmap_mode="r"); obs=np.load(arc/"stageiv_budget_qpf6_mm.npy",mmap_mode="r")
    if not (len(cases)==len(fcst)==len(obs)): raise RuntimeError("Stage-IV archive length mismatch")
    pt=PrecipTransform.from_stats_json("log1p_zscore",args.stats_json); dev=torch.device("cuda")
    v1,c1=load_model_checkpoint(args.baseline_checkpoint,device=dev); v2,c2=load_model_checkpoint(args.checkpoint,device=dev); v1.eval(); v2.eval()
    zf1=encode(v1,fcst,pt,dev,args.batch_size); zo1=encode(v1,obs,pt,dev,args.batch_size); zf2=encode(v2,fcst,pt,dev,args.batch_size); zo2=encode(v2,obs,pt,dev,args.batch_size)
    s1=np.sum(zf1*zo1,axis=1); s2=np.sum(zf2*zo2,axis=1)
    rec=[]
    for i in range(len(cases)):
        f=np.asarray(fcst[i],np.float32); o=np.asarray(obs[i],np.float32); r=cases.iloc[i].to_dict(); wet_f=float(np.mean(f>=.1)); wet_o=float(np.mean(o>=.1))
        rec.append({**r,"learned_v1":float(s1[i]),"learned_v2":float(s2[i]),"learned_delta_v2_minus_v1":float(s2[i]-s1[i]),"corr":pearson_r(f,o),"rmse":rmse(f,o),"mae":mae(f,o),"fss":fss(f,o,args.fss_threshold,args.fss_width),"fcst_wet_fraction_ge0p1":wet_f,"obs_wet_fraction_ge0p1":wet_o,"both_wet":bool((wet_f>0) and (wet_o>0)),"fcst_mean_mm":float(np.mean(f)),"obs_mean_mm":float(np.mean(o)),"fcst_max_mm":float(np.max(f)),"obs_max_mm":float(np.max(o))})
    df=pd.DataFrame(rec); df.to_csv(out/"per_case_metrics.csv.gz",index=False)
    summary=[]
    for name in ["learned_v1","learned_v2"]:
        summary.append({"metric":name,"mean":float(np.nanmean(df[name])),"std":float(np.nanstd(df[name])),"spearman_vs_corr":spearman(df[name],df["corr"]),"spearman_vs_fss":spearman(df[name],df["fss"]),**quantiles(df[name])})
    pd.DataFrame(summary).to_csv(out/"score_summary.csv",index=False)
    meta={"v1_checkpoint":str(args.baseline_checkpoint),"v1_sha256":sha(args.baseline_checkpoint),"v1_step":step(c1),"v2_checkpoint":str(args.checkpoint),"v2_sha256":sha(args.checkpoint),"v2_step":step(c2),"n_cases":len(df),"stageiv_regrid":"BUDGET","representative_fss_threshold_mm":args.fss_threshold,"representative_fss_width":args.fss_width,"sealed_test_used":False}
    (out/"run_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    df.sort_values("learned_delta_v2_minus_v1").head(50).to_csv(out/"largest_v2_score_decreases.csv",index=False); df.sort_values("learned_delta_v2_minus_v1",ascending=False).head(50).to_csv(out/"largest_v2_score_increases.csv",index=False)

    bins=np.linspace(min(df.learned_v1.min(),df.learned_v2.min()),max(df.learned_v1.max(),df.learned_v2.max()),45)
    plt.figure(figsize=(8,5)); plt.hist(df.learned_v1,bins=bins,alpha=.45,label="Baseline V1"); plt.hist(df.learned_v2,bins=bins,alpha=.45,label="Hard-ranking V2"); plt.xlabel("S(F,O): learned cosine similarity"); plt.ylabel("case count"); plt.title("HRRR–Stage IV learned similarity distribution"); plt.legend(); savefig(plots/"learned_similarity_v1_vs_v2_histogram.png")
    both=df[df.both_wet]
    plt.figure(figsize=(8,5)); plt.hist(both.learned_v1,bins=bins,alpha=.45,label="Baseline V1"); plt.hist(both.learned_v2,bins=bins,alpha=.45,label="Hard-ranking V2"); plt.xlabel("S(F,O): learned cosine similarity"); plt.ylabel("case count"); plt.title("HRRR–Stage IV learned similarity: both-wet cases"); plt.legend(); savefig(plots/"learned_similarity_v1_vs_v2_bothwet.png")
    plt.figure(figsize=(6,6)); plt.scatter(df.learned_v1,df.learned_v2,s=10,alpha=.5); lo=min(df.learned_v1.min(),df.learned_v2.min()); hi=max(df.learned_v1.max(),df.learned_v2.max()); plt.plot([lo,hi],[lo,hi],"--",lw=1); plt.xlabel("Baseline V1 similarity"); plt.ylabel("Hard-ranking V2 similarity"); plt.title("Stage-IV F/O score: V1 vs V2"); savefig(plots/"stageiv_v1_vs_v2_scatter.png")
    plt.figure(figsize=(7,5.5)); plt.scatter(df.fss,df.learned_v2,s=10,alpha=.5); plt.xlabel(f"FSS threshold={args.fss_threshold:g} mm, width={args.fss_width}"); plt.ylabel("Hard-ranking V2 similarity"); plt.title("V2 learned similarity vs FSS"); savefig(plots/"v2_vs_fss.png")
    plt.figure(figsize=(7,5.5)); plt.scatter(df["corr"],df.learned_v2,s=10,alpha=.5); plt.xlabel("Pearson correlation"); plt.ylabel("Hard-ranking V2 similarity"); plt.title("V2 learned similarity vs correlation"); savefig(plots/"v2_vs_correlation.png")
    plt.figure(figsize=(7,5.5)); plt.scatter(df.learned_v1,df.learned_delta_v2_minus_v1,s=10,alpha=.5); plt.axhline(0,color="0.4",lw=1); plt.xlabel("Baseline V1 similarity"); plt.ylabel("V2 − V1 similarity"); plt.title("Where hard-ranking fine-tuning moved Stage-IV scores"); savefig(plots/"v2_minus_v1_vs_v1_score.png")

    print("="*80); print("STAGE-IV V1/V2 COMPARISON COMPLETE"); print("="*80)
    for name in ["learned_v1","learned_v2"]:
        print(f"{name:12s} mean={df[name].mean():+.4f} median={df[name].median():+.4f} Spearman corr={spearman(df[name],df['corr']):+.3f} FSS={spearman(df[name],df['fss']):+.3f}")
    print(f"delta V2-V1 median={df.learned_delta_v2_minus_v1.median():+.4f} p05={df.learned_delta_v2_minus_v1.quantile(.05):+.4f} p95={df.learned_delta_v2_minus_v1.quantile(.95):+.4f}")
    print(f"Output: {out}")

if __name__=="__main__": main()

