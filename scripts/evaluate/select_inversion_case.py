#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from src.eval.feature_inversion import (
    MM_PER_IN, PRECIP_LEVELS_IN, PRECIP_COLORS_IN, PRECIP_OVER_IN,
    default_archive_dir, load_stageiv_archive, rank_candidates,
)


def main():
    ap = argparse.ArgumentParser(description='Rank and visualize candidate Stage-IV fields for feature inversion.')
    ap.add_argument('--archive-dir', default=str(default_archive_dir()))
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--min-wet-pixels', type=int, default=64)
    ap.add_argument('--top-k', type=int, default=12)
    args = ap.parse_args()

    archive = Path(args.archive_dir)
    out = Path(args.output_dir) if args.output_dir else Path(os.environ['DATA_DIR']) / 'derived/evaluation/feature_inversion_v1/case_selection'
    out.mkdir(parents=True, exist_ok=True)

    cases, stageiv = load_stageiv_archive(archive)
    ranked = rank_candidates(cases, stageiv, args.min_wet_pixels)
    ranked.to_csv(out / 'ranked_stageiv_candidates.csv', index=False)

    top = ranked.head(args.top_k).copy()
    cmap = ListedColormap(PRECIP_COLORS_IN)
    cmap.set_under('white')
    cmap.set_over(PRECIP_OVER_IN)
    norm = BoundaryNorm(PRECIP_LEVELS_IN, cmap.N)

    ncol = 4
    nrow = int(np.ceil(len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.2, 3.05 * nrow), squeeze=False)
    mesh = None
    for ax in axes.ravel():
        ax.axis('off')
    for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
        idx = int(row.case_index)
        f_in = np.asarray(stageiv[idx], float) / MM_PER_IN
        mesh = ax.pcolormesh(f_in, shading='auto', cmap=cmap, norm=norm)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f"case {idx} | {row.get('label','')}\n"
            f"max={row.max_in:.2f} in | wet={int(row.wet_pixels_ge_0p1in)} | axis={row.axis_ratio:.1f}",
            fontsize=9,
        )
        ax.axis('on')
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.035, pad=0.035, extend='max')
        cbar.set_label('6-h precipitation (in)')
    fig.suptitle('Stage IV candidates for feature-space inversion pilot', fontsize=14, fontweight='bold')
    fig.subplots_adjust(left=0.03, right=0.99, top=0.91, bottom=0.09, wspace=0.08, hspace=0.22)
    fig.savefig(out / 'candidate_gallery.png', dpi=190, bbox_inches='tight')
    plt.close(fig)

    selected = top.iloc[0].to_dict()
    (out / 'auto_selected_case.json').write_text(json.dumps(selected, indent=2, default=str) + '\n')

    print('=' * 88)
    print('FEATURE-INVERSION CASE SELECTION')
    print('=' * 88)
    print(f'Archive          : {archive}')
    print(f'Eligible cases   : {len(ranked)}')
    print(f'Auto-selected    : case {int(top.iloc[0].case_index)}')
    print(f'Candidate gallery: {out / "candidate_gallery.png"}')
    print(f'Candidate CSV    : {out / "ranked_stageiv_candidates.csv"}')
    print()
    print(top[['case_index','label','max_in','wet_pixels_ge_0p1in','center_distance_normalized','edge_mass_fraction','axis_ratio','selection_score']].to_string(index=False))


if __name__ == '__main__':
    main()
