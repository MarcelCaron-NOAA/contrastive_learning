#!/usr/bin/env python3
"""Plot the signed morphology suite using the same visual grammar as the Wang plots."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

METRICS=[("ts","TS","forestgreen"),("pod","POD","limegreen"),("normalized_far","normalized FAR","seagreen"),("fss10","FSS-10","blue"),("fss5","FSS-5","lightskyblue"),("learned_similarity_z_v2","Learned similarity","red")]
TITLES={
 "smoothing_sharpening":"Sharpening / smoothing biases experiments",
 "rotation_deg":"Rotation biases experiments",
 "principal_axis_log2_scale":"Principal-axis compression / elongation biases experiments",
 "tile_fragmentation_cells":"Tile-fragmentation biases experiments",
 "object_dispersion_cells":"Object concentration / dispersion biases experiments",
}
XLABEL={
 "smoothing_sharpening":"bias (− sharpening, + smoothing sigma)",
 "rotation_deg":"rotation bias (degrees)",
 "principal_axis_log2_scale":"log2 major-axis scale (− compression, + elongation)",
 "tile_fragmentation_cells":"signed tile-jitter bias (grid cells)",
 "object_dispersion_cells":"bias (grid cells; − inward, + outward)",
}

def plot_ci(ax,g,stem,label,color,ls="-",fill=True,lw=1.5):
    x=g.bias.to_numpy(); y=g[f"{stem}_mean"].to_numpy(); ax.plot(x,y,label=label,color=color,lw=lw,linestyle=ls)
    lo=f"{stem}_ci95_lo"; hi=f"{stem}_ci95_hi"
    if fill and lo in g and hi in g: ax.fill_between(x,g[lo],g[hi],color=color,alpha=.16,linewidth=0)

def decorate(ax,fam):
    ax.axvline(0,color="0.5",lw=.8,alpha=.55); ax.set_title(TITLES[fam]); ax.set_ylabel("verification scores"); ax.set_xlabel(XLABEL[fam]); ax.set_ylim(-.02,1.05); ax.grid(alpha=.25)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("summary_csv"); ap.add_argument("--output-dir",default=None); args=ap.parse_args()
    df=pd.read_csv(args.summary_csv); out=Path(args.output_dir) if args.output_dir else Path(args.summary_csv).parent; out.mkdir(parents=True,exist_ok=True)
    fams=list(TITLES)
    # One figure per family: same metric ordering/colors as Wang-style plot.
    for fam in fams:
        g=df[df.family==fam].sort_values("bias"); fig,ax=plt.subplots(figsize=(7.5,5.3))
        for stem,label,color in METRICS: plot_ci(ax,g,stem,label,color)
        decorate(ax,fam); ax.legend(loc="best"); fig.tight_layout(); fig.savefig(out/f"{fam}_wang_style.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    # Compact all-family panel.
    fig,axes=plt.subplots(len(fams),1,figsize=(7.5,16.5))
    for ax,fam in zip(axes,fams):
        g=df[df.family==fam].sort_values("bias")
        for stem,label,color in METRICS: plot_ci(ax,g,stem,label,color)
        decorate(ax,fam)
    h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,loc="center right",bbox_to_anchor=(1.22,.5)); fig.tight_layout(rect=[0,0,.83,1]); fig.savefig(out/"signed_morphology_wang_style_all.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    # Learned-only direct V1/V2 comparison.
    fig,axes=plt.subplots(len(fams),1,figsize=(7.5,16.5))
    for ax,fam in zip(axes,fams):
        g=df[df.family==fam].sort_values("bias")
        plot_ci(ax,g,"learned_similarity_z_v1","Baseline V1","darkred",ls="--",fill=False,lw=1.7)
        plot_ci(ax,g,"learned_similarity_z_v2","Hard-ranking V2","red",ls="-",fill=True,lw=1.7)
        ax.axvline(0,color="0.5",lw=.8,alpha=.55); ax.set_title(TITLES[fam]); ax.set_ylabel("learned cosine similarity"); ax.set_xlabel(XLABEL[fam]); ax.set_ylim(-.02,1.05); ax.grid(alpha=.25)
    h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,loc="center right",bbox_to_anchor=(1.20,.5)); fig.tight_layout(rect=[0,0,.84,1]); fig.savefig(out/"signed_morphology_learned_v1_vs_v2.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    print(out/"signed_morphology_wang_style_all.png"); print(out/"signed_morphology_learned_v1_vs_v2.png")

if __name__=="__main__": main()

