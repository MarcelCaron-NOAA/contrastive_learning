#!/usr/bin/env python3
"""
Figure-8-style raw displacement results:
TS, FSS-10, and learned similarity in three side-by-side panels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("per_case_csv")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    df = pd.read_csv(args.per_case_csv)
    g = df[df["family"] == "displacement"].copy()
    out = (
        Path(args.output)
        if args.output
        else Path(args.per_case_csv).with_name("wang_raw_displacement.png")
    )

    specs = [
        ("ts","TS"),
        ("fss10","FSS-10"),
        ("learned_similarity_z","Learned similarity"),
    ]

    fig, axes = plt.subplots(1,3,figsize=(11,3.8),sharey=False)
    for ax,(col,title) in zip(axes,specs):
        ax.scatter(
            g["bias"], g[col], s=4, alpha=.10
        )
        ax.set_title(title)
        ax.set_xlabel("biases")
        ax.set_ylabel("verification score")
        ax.grid(alpha=.25)
        ax.set_xlim(-10.5,10.5)

    fig.tight_layout()
    fig.savefig(out,dpi=220,bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()

