#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

LAGS=(1,3,6)

def default_exp():
    return Path(os.environ['DATA_DIR'])/'derived/experiments/first_real_train_validation_v1'

def newest(exp, pattern):
    files=sorted(exp.glob(pattern))
    if not files:
        raise SystemExit(f'No files match {pattern} in {exp}')
    return files[-1]

def make_boxplot(avg, col, path, title):
    data=[avg.loc[avg['lag_hours']==lag, col].dropna().to_numpy() for lag in LAGS]
    plt.figure(figsize=(8,5.5))
    plt.boxplot(data, showfliers=False, whis=(5,95))
    plt.xticks(range(1,len(LAGS)+1), [f'{lag} h' for lag in LAGS])
    plt.xlabel('Initialization time-lag magnitude')
    plt.ylabel('A / same-valid-time field cosine similarity')
    plt.title(title)
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment-dir', default=str(default_exp()))
    args=p.parse_args()
    exp=Path(args.experiment_dir)
    long_path=newest(exp,'timelag_similarity_long_step*.csv')
    anchor_path=newest(exp,'timelag_similarity_anchor_step*.csv')
    summary_path=newest(exp,'timelag_similarity_summary_step*.csv')
    long_df=pd.read_csv(long_path)
    anchor_df=pd.read_csv(anchor_path)
    summary_df=pd.read_csv(summary_path)
    step=int(anchor_df['checkpoint_step'].iloc[0]) if 'checkpoint_step' in anchor_df else -1
    avg=(long_df.groupby(['checkpoint_step','anchor_id','valid_time','valid_hour_utc','season','anchor_fhr','lag_hours'],as_index=False)[['z_similarity','h_similarity']].mean().rename(columns={'z_similarity':'z_similarity_direction_mean','h_similarity':'h_similarity_direction_mean'}))
    plot_dir=exp/'diagnostic_plots'/'timelag_gradation'
    plot_dir.mkdir(parents=True, exist_ok=True)
    make_boxplot(avg,'z_similarity_direction_mean',plot_dir/f'timelag_similarity_z_boxplot_step{step}.png','Projection z: same-valid-time similarity versus time lag')
    make_boxplot(avg,'h_similarity_direction_mean',plot_dir/f'timelag_similarity_h_boxplot_step{step}.png','Backbone h: same-valid-time similarity versus time lag')
    print('='*80)
    print('TIME-LAG GRADATION RECOVERY / SUMMARY')
    print('='*80)
    print(summary_df.to_string(index=False))
    print()
    print(f"z ordering fractions: S1>S3={100*anchor_df['z_1_gt_3'].mean():.1f}% | S3>S6={100*anchor_df['z_3_gt_6'].mean():.1f}% | S1>S3>S6={100*anchor_df['z_full_monotonic'].mean():.1f}%")
    print(f"h ordering fractions: S1>S3={100*anchor_df['h_1_gt_3'].mean():.1f}% | S3>S6={100*anchor_df['h_3_gt_6'].mean():.1f}% | S1>S3>S6={100*anchor_df['h_full_monotonic'].mean():.1f}%")
    print()
    print(f'Long pairs : {long_path}')
    print(f'Per-anchor : {anchor_path}')
    print(f'Summary    : {summary_path}')
    print(f'Plots      : {plot_dir}')

if __name__=='__main__':
    main()

