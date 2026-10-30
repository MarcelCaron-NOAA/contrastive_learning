#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from feature_inversion_common import MM_PER_IN, PRECIP_LEVELS_IN, PRECIP_COLORS_IN, PRECIP_OVER_IN


def load_target(model_dir: Path, target: float):
    tag = ('p' if target >= 0 else 'm') + f'{abs(target):.2f}'.replace('.', 'p')
    td = model_dir / f'target_{tag}'
    field = np.load(td / 'optimized_precip_mm.npy') / MM_PER_IN
    meta = json.loads((td / 'summary.json').read_text())
    return field, meta


def make_cmap():
    cmap = ListedColormap(PRECIP_COLORS_IN)
    cmap.set_under('white')
    cmap.set_over(PRECIP_OVER_IN)
    norm = BoundaryNorm(PRECIP_LEVELS_IN, cmap.N)
    return cmap, norm


def plot_model(outroot: Path, model: str, targets: list[float], original_in: np.ndarray):
    cmap, norm = make_cmap()
    fields = [original_in]
    metas = [{'target_similarity':1.0,'achieved_similarity':1.0,'max_mm':float(original_in.max()*MM_PER_IN)}]
    for t in targets:
        f, m = load_target(outroot / model, t)
        fields.append(f); metas.append(m)

    fig, axes = plt.subplots(1, len(fields), figsize=(3.2*len(fields), 3.65))
    mesh = None
    for j, (ax, f, m) in enumerate(zip(np.atleast_1d(axes), fields, metas)):
        mesh = ax.pcolormesh(f, shading='auto', cmap=cmap, norm=norm)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_title('Original\nS = 1.000', fontsize=10, fontweight='bold')
        else:
            ax.set_title(
                f"Target {m['target_similarity']:+.2f}\nAchieved {m['achieved_similarity']:+.3f}",
                fontsize=10, fontweight='bold'
            )
    cbar = fig.colorbar(mesh, ax=np.atleast_1d(axes).tolist(), orientation='horizontal', fraction=0.055, pad=0.08, extend='max')
    cbar.set_label('6-h precipitation (in)')
    fig.suptitle(f'{model.upper()} feature-space inversion', fontsize=14, fontweight='bold')
    fig.subplots_adjust(left=0.025, right=0.995, top=0.84, bottom=0.20, wspace=0.05)
    fig.savefig(outroot / f'gallery_{model}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_combined(outroot: Path, models: list[str], targets: list[float], original_in: np.ndarray):
    cmap, norm = make_cmap()
    ncol = 1 + len(targets)
    fig, axes = plt.subplots(len(models), ncol, figsize=(3.05*ncol, 3.05*len(models)), squeeze=False)
    mesh = None
    for i, model in enumerate(models):
        fields = [original_in]; metas = [{'target_similarity':1.0,'achieved_similarity':1.0}]
        for t in targets:
            f, m = load_target(outroot / model, t); fields.append(f); metas.append(m)
        for j, (f, m) in enumerate(zip(fields, metas)):
            ax = axes[i,j]
            mesh = ax.pcolormesh(f, shading='auto', cmap=cmap, norm=norm)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                title = 'Original\nS = 1.000'
            else:
                title = f"Target {m['target_similarity']:+.2f}\nAchieved {m['achieved_similarity']:+.3f}"
            if i == 0:
                ax.set_title(title, fontsize=9, fontweight='bold')
            if j == 0:
                ax.set_ylabel(model.upper(), fontsize=11, fontweight='bold')
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.030, pad=0.045, extend='max')
    cbar.set_label('6-h precipitation (in)')
    fig.suptitle('Feature-space inversion pilot', fontsize=15, fontweight='bold')
    fig.subplots_adjust(left=0.055, right=0.995, top=0.91, bottom=0.10, wspace=0.035, hspace=0.09)
    fig.savefig(outroot / 'gallery_all_models.png', dpi=210, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--models', nargs='+', default=['nfv','syn3','syn8'])
    ap.add_argument('--targets', nargs='+', default=['0.5','0.0','-0.5','-1.0'])
    args = ap.parse_args()
    outroot = Path(args.input_dir)
    targets = [float(x) for x in args.targets]
    original_in = np.load(outroot / 'original_stageiv_precip_mm.npy') / MM_PER_IN
    for model in args.models:
        plot_model(outroot, model, targets, original_in)
    if len(args.models) > 1:
        plot_combined(outroot, args.models, targets, original_in)
    print(f'Wrote inversion galleries to {outroot}')


if __name__ == '__main__':
    main()
