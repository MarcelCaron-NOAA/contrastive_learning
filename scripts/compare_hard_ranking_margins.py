#!/usr/bin/env python3
"""
Compare per-case hard-ranking margins for the source V1 checkpoint and V2.

This is the diagnostic discussed after the first fine-tuning run.  A ranking
percentage only records the sign of a margin.  These distributions show
whether fine-tuning shifted the *whole* margin distribution, widened already-
correct cases, or changed the negative/ambiguous tail.
"""
from __future__ import annotations
import argparse, json, os
from dataclasses import fields
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.data import HRRRArchive, PrecipTransform
from src.data.hard_ranking_sampler import HardRankingRecipe, HardRankingDataset, hard_ranking_collate
from src.models.checkpoint_loader import load_model_checkpoint


def defaults():
    data=Path(os.environ["DATA_DIR"])
    exp=data/"derived/experiments/hard_ranking_finetune_v1"
    return {"recipes":exp/"fixed_validation_recipes.jsonl","manifest":data/"forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz","stats":data/"metadata/dataset_v1/precip_normalization_stats_v1.json","v1":data/"derived/experiments/groupnorm_pilot_v1/checkpoint_best.pt","v2":exp/"checkpoint_best.pt","out":exp/"evaluation/ranking_margin_v1_vs_v2"}

def load_recipes(path):
    names={f.name for f in fields(HardRankingRecipe)}; out=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        d=json.loads(line); d={k:v for k,v in d.items() if k in names}; d["negative_ids"]=tuple(d["negative_ids"]); out.append(HardRankingRecipe(**d))
    return out

@torch.no_grad()
def encode_views(model,batch,device):
    names=["anchor","positive","tl3","tl6","e1","e2","e3","e6"]
    B=batch["anchor"].shape[0]; x=torch.cat([batch[k] for k in names],dim=0).to(device,non_blocking=True)
    with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True): _,z=model(x)
    z=F.normalize(z.float(),dim=-1)
    return {name:z[i*B:(i+1)*B] for i,name in enumerate(names)}

def margins(z):
    sim={k:torch.sum(z["anchor"]*z[k],dim=1).cpu().numpy() for k in ["positive","tl3","tl6","e1","e2","e3","e6"]}
    return {"p_minus_tl3":sim["positive"]-sim["tl3"],"p_minus_tl6":sim["positive"]-sim["tl6"],"tl3_minus_tl6":sim["tl3"]-sim["tl6"],"e1_minus_e2":sim["e1"]-sim["e2"],"e1_minus_e3":sim["e1"]-sim["e3"],"e1_minus_e6":sim["e1"]-sim["e6"],"e2_minus_e3":sim["e2"]-sim["e3"],"e2_minus_e6":sim["e2"]-sim["e6"],"e3_minus_e6":sim["e3"]-sim["e6"]}

def savefig(p): plt.tight_layout(); plt.savefig(p,dpi=210,bbox_inches="tight"); plt.close()


def main():
    d=defaults(); ap=argparse.ArgumentParser(); ap.add_argument("--recipes",default=str(d["recipes"])); ap.add_argument("--manifest",default=str(d["manifest"])); ap.add_argument("--stats-json",default=str(d["stats"])); ap.add_argument("--baseline-checkpoint",default=str(d["v1"])); ap.add_argument("--checkpoint",default=str(d["v2"])); ap.add_argument("--output-dir",default=str(d["out"])); ap.add_argument("--batch-size",type=int,default=8); args=ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
    recipes=load_recipes(args.recipes); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    pt=PrecipTransform.from_stats_json("log1p_zscore",args.stats_json); arc=HRRRArchive(args.manifest); ds=HardRankingDataset(arc,negative_count=len(recipes[0].negative_ids),transform=pt)
    dev=torch.device("cuda"); v1,_=load_model_checkpoint(args.baseline_checkpoint,device=dev); v2,_=load_model_checkpoint(args.checkpoint,device=dev); v1.eval(); v2.eval()
    rows=[]
    try:
        for s in range(0,len(recipes),args.batch_size):
            rr=recipes[s:s+args.batch_size]; batch=hard_ranking_collate([ds[r] for r in rr]); z1=encode_views(v1,batch,dev); z2=encode_views(v2,batch,dev); m1=margins(z1); m2=margins(z2)
            tlmask=batch["tl_pair_mask"].numpy(); emask=batch["evolution_pair_mask"].numpy()
            names_tl=["p_minus_tl3","p_minus_tl6","tl3_minus_tl6"]; names_e=["e1_minus_e2","e1_minus_e3","e1_minus_e6","e2_minus_e3","e2_minus_e6","e3_minus_e6"]
            for i,r in enumerate(rr):
                base={"case_index":s+i,"anchor_id":r.anchor_id,"valid_time":r.anchor_valid_time,"forecast_hour":r.anchor_forecast_hour,"branch_sign":r.branch_sign}
                for q,name in enumerate(names_tl):
                    if tlmask[i,q]: rows.append({**base,"pair":name,"family":"TL","v1_margin":float(m1[name][i]),"v2_margin":float(m2[name][i])})
                for q,name in enumerate(names_e):
                    if emask[i,q]: rows.append({**base,"pair":name,"family":"evolution","v1_margin":float(m1[name][i]),"v2_margin":float(m2[name][i])})
            print(f"processed {min(s+len(rr),len(recipes))}/{len(recipes)} recipes")
    finally: arc.close()
    df=pd.DataFrame(rows); df["delta_margin_v2_minus_v1"]=df.v2_margin-df.v1_margin; df.to_csv(out/"per_case_pair_margins.csv.gz",index=False)
    summary=df.groupby(["family","pair"]).agg(n=("v1_margin","size"),v1_median=("v1_margin","median"),v2_median=("v2_margin","median"),v1_mean=("v1_margin","mean"),v2_mean=("v2_margin","mean"),v1_correct_pct=("v1_margin",lambda x:100*np.mean(x>0)),v2_correct_pct=("v2_margin",lambda x:100*np.mean(x>0)),median_delta=("delta_margin_v2_minus_v1","median")).reset_index(); summary.to_csv(out/"margin_summary.csv",index=False)
    # The adjacent relations are the easiest to interpret visually.
    pairs=["p_minus_tl3","tl3_minus_tl6","e1_minus_e2","e2_minus_e3","e3_minus_e6"]
    fig,axes=plt.subplots(len(pairs),1,figsize=(8,15))
    for ax,pair in zip(axes,pairs):
        g=df[df.pair==pair]; lo=float(min(g.v1_margin.min(),g.v2_margin.min())); hi=float(max(g.v1_margin.max(),g.v2_margin.max())); bins=np.linspace(lo,hi,45)
        ax.hist(g.v1_margin,bins=bins,alpha=.45,label="Baseline V1"); ax.hist(g.v2_margin,bins=bins,alpha=.45,label="Hard-ranking V2"); ax.axvline(0,color="0.35",lw=1); ax.set_title(pair.replace("_"," ")); ax.set_xlabel("ordered similarity margin (positive = desired ordering)"); ax.set_ylabel("case count"); ax.grid(alpha=.15)
    h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,loc="upper right"); fig.tight_layout(); fig.savefig(out/"adjacent_margin_distributions_v1_vs_v2.png",dpi=210,bbox_inches="tight"); plt.close(fig)
    print("\n",summary.to_string(index=False)); print(f"\nOutput: {out}")

if __name__=="__main__": main()

