#!/usr/bin/env python3

"""
Build dataset-v1 chronological splits and the eligible HRRR anchor index.

Locked v1 design
================

Anchor A
--------
* HRRR 6-h accumulated precipitation
* forecast leads F07-F17
* split membership determined by the VALID END TIME of the 6-h field

Positive P
----------
* same valid time / same valid 6-h accumulation period as A
* initialization time offset by -1 h or +1 h from A
* equivalently:
      init(A)-1 h -> lead(A)+1 h
      init(A)+1 h -> lead(A)-1 h
* BOTH positive directions must exist for A to be eligible

Negative N
----------
N is NOT permanently paired with A here.

The negative pool contains available HRRR F07-F17 fields. During training,
N will be drawn dynamically subject to:

* same split as A
* same forecast lead as A
* same UTC valid hour as A
* absolute valid-time separation STRICTLY > 90 days

This keeps forecast lead and diurnal timing from becoming easy shortcuts.

Splits
======

TRAIN:
    2020-12-02 through 2024-09-20

GUARD 1:
    2024-09-21 through 2024-09-30
    excluded from model development

VALIDATION:
    2024-10-01 through 2025-09-20

GUARD 2:
    2025-09-21 through 2025-09-30
    excluded from model development

TEST:
    2025-10-01 through 2026-09-22
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------

SPLITS = [
    ("train",      "2020-12-02", "2024-09-20"),
    ("guard_1",    "2024-09-21", "2024-09-30"),
    ("validation", "2024-10-01", "2025-09-20"),
    ("guard_2",    "2025-09-21", "2025-09-30"),
    ("test",       "2025-10-01", "2026-09-22"),
]

MODEL_SPLITS = ("train", "validation", "test")

ANCHOR_LEADS = tuple(range(7, 18))

NEGATIVE_MIN_DAYS = 90.0


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_col(df, candidates, required=True):
    """Return the first matching column name."""
    for name in candidates:
        if name in df.columns:
            return name

    if required:
        raise RuntimeError(
            "Could not find any of these columns:\n"
            f"  {candidates}\n\n"
            "Manifest columns are:\n"
            f"  {list(df.columns)}"
        )

    return None


def as_bool(series):
    """Interpret common text/integer representations of boolean values."""
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "t", "yes"])
    )


def parse_time_series(series):
    """
    Parse timestamps robustly.

    Handles ordinary datetime strings as well as integer Unix timestamps
    stored in seconds, milliseconds, microseconds, or nanoseconds.
    """

    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")

        finite = values[np.isfinite(values)]

        if len(finite) == 0:
            return pd.to_datetime(
                series,
                errors="coerce",
                utc=True,
            )

        median_abs = float(
            np.nanmedian(np.abs(finite))
        )

        if median_abs < 1e11:
            unit = "s"
        elif median_abs < 1e14:
            unit = "ms"
        elif median_abs < 1e17:
            unit = "us"
        else:
            unit = "ns"

        return pd.to_datetime(
            values,
            unit=unit,
            errors="coerce",
            utc=True,
        )

    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )


def assign_split(timestamp):
    """Assign a field to a chronological split from its valid time."""

    if pd.isna(timestamp):
        return "outside"

    date = timestamp.date()

    for name, start, end in SPLITS:

        start_date = pd.Timestamp(start).date()
        end_date = pd.Timestamp(end).date()

        if start_date <= date <= end_date:
            return name

    return "outside"


def season_from_month(month):
    if month in (12, 1, 2):
        return "DJF"
    elif month in (3, 4, 5):
        return "MAM"
    elif month in (6, 7, 8):
        return "JJA"
    else:
        return "SON"


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------

def load_manifest(path):

    print(f"Reading manifest:\n  {path}")

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    sample_col = find_col(
        df,
        [
            "sample_id",
            "field_id",
            "id",
        ],
    )

    store_col = find_col(
        df,
        [
            "store",
            "store_path",
            "zarr_path",
            "path",
        ],
    )

    init_col = find_col(
        df,
        [
            "init_hour",
            "cycle_hour",
            "cycle",
        ],
    )

    fhr_col = find_col(
        df,
        [
            "forecast_hour",
            "lead_hour",
            "fhr",
            "lead",
        ],
    )

    valid_col = find_col(
        df,
        [
            "valid_end",
            "valid_end_time",
            "valid_time",
            "valid_end_unix",
        ],
    )

    available_col = find_col(
        df,
        ["available"],
        required=False,
    )

    print()
    print("Using manifest columns:")
    print(f"  field ID      : {sample_col}")
    print(f"  store         : {store_col}")
    print(f"  init hour     : {init_col}")
    print(f"  forecast hour : {fhr_col}")
    print(f"  valid time    : {valid_col}")
    print(f"  available     : {available_col}")
    print()

    if available_col is not None:

        available = as_bool(
            df[available_col]
        )

        print(
            f"Available fields: "
            f"{available.sum():,} / {len(df):,}"
        )

        df = df.loc[available].copy()

    df["field_id"] = (
        df[sample_col]
        .astype(str)
    )

    df["store_path"] = (
        df[store_col]
        .astype(str)
    )

    df["init_hour_std"] = pd.to_numeric(
        df[init_col],
        errors="coerce",
    )

    df["forecast_hour_std"] = pd.to_numeric(
        df[fhr_col],
        errors="coerce",
    )

    df["valid_time"] = parse_time_series(
        df[valid_col]
    )

    df = df.dropna(
        subset=[
            "field_id",
            "store_path",
            "init_hour_std",
            "forecast_hour_std",
            "valid_time",
        ]
    ).copy()

    df["init_hour_std"] = (
        df["init_hour_std"]
        .astype(int)
    )

    df["forecast_hour_std"] = (
        df["forecast_hour_std"]
        .astype(int)
    )

    # Nanoseconds since epoch.
    # This gives us a simple integer time coordinate for fast comparisons.
    # Integer Unix seconds.
    #
    # Do this through an explicit timedelta calculation instead of relying on
    # datetime64's internal storage resolution, which may be seconds,
    # milliseconds, microseconds, or nanoseconds depending on pandas/version.
    epoch = pd.Timestamp(
        "1970-01-01",
        tz="UTC",
    )

    df["valid_unix_s"] = (
        (df["valid_time"] - epoch)
        .dt.total_seconds()
        .round()
        .astype(np.int64)
    )

    df["valid_date"] = (
        df["valid_time"]
        .dt.strftime("%Y-%m-%d")
    )

    df["valid_hour_utc"] = (
        df["valid_time"]
        .dt.hour
        .astype(int)
    )

    df["season"] = (
        df["valid_time"]
        .dt.month
        .map(season_from_month)
    )

    df["split"] = (
        df["valid_time"]
        .map(assign_split)
    )

    # -------------------------------------------------------------
    # Integrity check:
    #
    # Same valid time + forecast lead should uniquely determine the
    # initialization for this archive.
    # -------------------------------------------------------------

    duplicated = df.duplicated(
        subset=[
            "valid_unix_s",
            "forecast_hour_std",
        ],
        keep=False,
    )

    if duplicated.any():

        example = df.loc[
            duplicated,
            [
                "field_id",
                "valid_time",
                "forecast_hour_std",
                "init_hour_std",
            ],
        ].head(20)

        raise RuntimeError(
            "Duplicate (valid_time, forecast_hour) keys "
            "were found in the manifest.\n\n"
            + example.to_string(index=False)
        )

    return df


# ---------------------------------------------------------------------
# Negative candidate counts
# ---------------------------------------------------------------------

def count_negative_candidates(
    negative_pool,
    anchors,
):
    """
    Count eligible negative fields for every anchor.

    Requirements:

      same split
      same forecast lead
      same UTC valid hour
      abs(valid_N - valid_A) > 90 days

    Important:
      strict > 90 days, not >= 90 days
    """

    cutoff_s = int(
        NEGATIVE_MIN_DAYS
        * 24
        * 60
        * 60
    )

    # -------------------------------------------------------------
    # Build sorted time arrays for each:
    #
    #   split / forecast lead / UTC valid hour
    #
    # -------------------------------------------------------------

    time_groups = {}

    grouped = negative_pool.groupby(
        [
            "split",
            "forecast_hour",
            "valid_hour_utc",
        ],
        sort=False,
    )

    for key, group in grouped:
        
        times = np.sort(
            group["valid_unix_s"].to_numpy(
                dtype=np.int64
            )
        )

        time_groups[key] = times

    print()
    print(
        "Negative-pool groups: "
        f"{len(time_groups):,}"
    )

    # -------------------------------------------------------------
    # Process anchors groupwise.
    # Much faster than searching one anchor at a time.
    # -------------------------------------------------------------

    counts = pd.Series(
        0,
        index=anchors.index,
        dtype=np.int32,
    )

    anchor_groups = anchors.groupby(
        [
            "split",
            "anchor_fhr",
            "valid_hour_utc",
        ],
        sort=False,
    )

    missing_groups = []

    for key, anchor_index in anchor_groups.groups.items():

        times = time_groups.get(key)

        if times is None:

            missing_groups.append(key)
            continue

        anchor_index = np.asarray(
            list(anchor_index)
        )

        anchor_times = (
            anchors.loc[
                anchor_index,
                "valid_unix_s",
            ]
            .to_numpy(dtype=np.int64)
        )

        # Strictly EARLIER than A - 90 days.
        n_earlier = np.searchsorted(
            times,
            anchor_times - cutoff_s,
            side="left",
        )

        # Strictly LATER than A + 90 days.
        first_later = np.searchsorted(
            times,
            anchor_times + cutoff_s,
            side="right",
        )

        n_later = (
            len(times)
            - first_later
        )

        counts.loc[anchor_index] = (
            n_earlier + n_later
        ).astype(np.int32)

    if missing_groups:

        print()
        print(
            "WARNING: some anchor grouping keys "
            "were absent from the negative pool:"
        )

        for key in missing_groups[:20]:
            print(f"  {key}")

        if len(missing_groups) > 20:
            print(
                f"  ... plus "
                f"{len(missing_groups) - 20} more"
            )

    return counts


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    manifest_path = Path(
        args.manifest
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Load canonical HRRR field manifest.
    # -------------------------------------------------------------

    fields = load_manifest(
        manifest_path
    )

    print()
    print(
        "Available fields after manifest cleanup: "
        f"{len(fields):,}"
    )

    print()
    print("Field counts by split:")

    print(
        fields["split"]
        .value_counts()
        .reindex(
            [
                "train",
                "guard_1",
                "validation",
                "guard_2",
                "test",
                "outside",
            ],
            fill_value=0,
        )
        .to_string()
    )

    # -------------------------------------------------------------
    # Build lookup:
    #
    #   (valid time, forecast lead) -> field ID
    #
    # This is how +/-1-h same-valid-time positives are identified.
    # -------------------------------------------------------------

    field_lookup = {
        (
            int(row.valid_unix_s),
            int(row.forecast_hour_std),
        ): row.field_id
        for row in fields.itertuples(
            index=False
        )
    }

    # -------------------------------------------------------------
    # Candidate anchors.
    # -------------------------------------------------------------

    candidate_anchors = fields.loc[
        fields["split"].isin(
            MODEL_SPLITS
        )
        &
        fields["forecast_hour_std"].isin(
            ANCHOR_LEADS
        )
    ].copy()

    candidate_anchors = (
        candidate_anchors
        .reset_index(drop=True)
    )

    print()
    print(
        "Candidate A fields "
        "(before positive availability check): "
        f"{len(candidate_anchors):,}"
    )

    # -------------------------------------------------------------
    # Find both +/-1-h positives.
    #
    # If A has lead L:
    #
    #   P initialized 1 h earlier:
    #       same valid time, lead L+1
    #
    #   P initialized 1 h later:
    #       same valid time, lead L-1
    # -------------------------------------------------------------

    positive_minus1 = []
    positive_plus1 = []

    for row in candidate_anchors.itertuples(
        index=False
    ):

        valid_unix_s = int(
            row.valid_unix_s
        )

        lead = int(
            row.forecast_hour_std
        )

        positive_minus1.append(
            field_lookup.get(
                (
                    valid_unix_s,
                    lead + 1,
                )
            )
        )

        positive_plus1.append(
            field_lookup.get(
                (
                    valid_unix_s,
                    lead - 1,
                )
            )
        )

    candidate_anchors[
        "positive_init_minus1h_id"
    ] = positive_minus1

    candidate_anchors[
        "positive_init_plus1h_id"
    ] = positive_plus1

    has_both_positives = (
        candidate_anchors[
            "positive_init_minus1h_id"
        ].notna()
        &
        candidate_anchors[
            "positive_init_plus1h_id"
        ].notna()
    )

    print(
        "Candidate A fields with both "
        "+/-1-h positives: "
        f"{has_both_positives.sum():,}"
    )

    anchor_source = (
        candidate_anchors.loc[
            has_both_positives
        ]
        .copy()
    )

    # -------------------------------------------------------------
    # Build the compact eligible-anchor table.
    # -------------------------------------------------------------

    anchors = pd.DataFrame(
        {
            "anchor_id":
                anchor_source[
                    "field_id"
                ].astype(str),

            "split":
                anchor_source[
                    "split"
                ],

            "valid_time":
                anchor_source[
                    "valid_time"
                ],

            "valid_unix_s":
                anchor_source[
                    "valid_unix_s"
                ].astype(np.int64),

            "valid_date":
                anchor_source[
                    "valid_date"
                ],

            "valid_hour_utc":
                anchor_source[
                    "valid_hour_utc"
                ].astype(int),

            "season":
                anchor_source[
                    "season"
                ],

            "anchor_init_hour":
                anchor_source[
                    "init_hour_std"
                ].astype(int),

            "anchor_fhr":
                anchor_source[
                    "forecast_hour_std"
                ].astype(int),

            "positive_init_minus1h_id":
                anchor_source[
                    "positive_init_minus1h_id"
                ].astype(str),

            "positive_init_plus1h_id":
                anchor_source[
                    "positive_init_plus1h_id"
                ].astype(str),
        }
    )

    anchors[
        "positive_init_minus1h_fhr"
    ] = (
        anchors["anchor_fhr"]
        + 1
    )

    anchors[
        "positive_init_plus1h_fhr"
    ] = (
        anchors["anchor_fhr"]
        - 1
    )

    anchors = (
        anchors
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------
    # Negative pool.
    #
    # Do NOT rename forecast_hour here. Keeping the canonical name
    # prevents the key mismatch that broke the previous version.
    # -------------------------------------------------------------

    negative_pool = fields.loc[
        fields["split"].isin(
            MODEL_SPLITS
        )
        &
        fields[
            "forecast_hour_std"
        ].isin(
            ANCHOR_LEADS
        ),
        [
            "field_id",
            "split",
            "valid_time",
            "valid_unix_s",
            "valid_date",
            "valid_hour_utc",
            "season",
            "init_hour_std",
            "forecast_hour_std",
            "store_path",
        ],
    ].copy()

    negative_pool = (
        negative_pool.rename(
            columns={
                "init_hour_std":
                    "init_hour",

                "forecast_hour_std":
                    "forecast_hour",
            }
        )
        .reset_index(drop=True)
    )

    print()
    print(
        "Negative-pool fields: "
        f"{len(negative_pool):,}"
    )

    print()
    print(
        "Negative pool by split:"
    )

    print(
        negative_pool[
            "split"
        ]
        .value_counts()
        .reindex(
            MODEL_SPLITS,
            fill_value=0,
        )
        .to_string()
    )

    # -------------------------------------------------------------
    # Count eligible N candidates for each A.
    # -------------------------------------------------------------

    print()
    print("Valid-time integer sanity check:")
    print(
        "  anchor range:",
        int(anchors["valid_unix_s"].min()),
        "to",
        int(anchors["valid_unix_s"].max()),
    )
    print(
        "  negative range:",
        int(negative_pool["valid_unix_s"].min()),
        "to",
        int(negative_pool["valid_unix_s"].max()),
    )
    print(
        "  90-day cutoff:",
        int(NEGATIVE_MIN_DAYS * 86400),
        "seconds",
    )
    anchors[
        "negative_candidate_count"
    ] = count_negative_candidates(
        negative_pool,
        anchors,
    )

    # -------------------------------------------------------------
    # Sanity-check distribution BEFORE deleting anything.
    # -------------------------------------------------------------

    counts = anchors[
        "negative_candidate_count"
    ]

    print()
    print(
        "Negative candidate-count distribution "
        "BEFORE eligibility filtering:"
    )

    print(
        counts.describe(
            percentiles=[
                0.01,
                0.05,
                0.50,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    n_zero = int(
        (counts == 0).sum()
    )

    print()
    print(
        "Anchors with zero valid negative candidates: "
        f"{n_zero:,}"
    )

    # This should be very small or zero.
    #
    # If it is huge, STOP instead of silently producing an empty index.
    if n_zero > 0.01 * len(anchors):

        raise RuntimeError(
            "\nMore than 1% of otherwise-eligible anchors have "
            "zero negative candidates.\n"
            "That is not expected for these chronological splits "
            "and indicates a bug in the negative matching logic.\n"
            "No dataset index has been written."
        )

    # Remove only genuinely unusable edge cases, if any.
    anchors = (
        anchors.loc[
            anchors[
                "negative_candidate_count"
            ] > 0
        ]
        .copy()
    )

    # -------------------------------------------------------------
    # Sort.
    # -------------------------------------------------------------

    anchors = (
        anchors.sort_values(
            [
                "split",
                "valid_time",
                "anchor_fhr",
                "anchor_init_hour",
            ]
        )
        .reset_index(drop=True)
    )

    negative_pool = (
        negative_pool.sort_values(
            [
                "split",
                "forecast_hour",
                "valid_hour_utc",
                "valid_time",
            ]
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------
    # Integrity checks.
    # -------------------------------------------------------------

    assert len(anchors) > 0

    assert anchors[
        "split"
    ].isin(
        MODEL_SPLITS
    ).all()

    assert anchors[
        "anchor_fhr"
    ].between(
        7,
        17,
    ).all()

    assert (
        anchors[
            "positive_init_minus1h_fhr"
        ]
        ==
        anchors[
            "anchor_fhr"
        ]
        + 1
    ).all()

    assert (
        anchors[
            "positive_init_plus1h_fhr"
        ]
        ==
        anchors[
            "anchor_fhr"
        ]
        - 1
    ).all()

    assert (
        anchors[
            "negative_candidate_count"
        ]
        > 0
    ).all()

    # -------------------------------------------------------------
    # Split definition JSON.
    # -------------------------------------------------------------

    split_definition = {
        "version": "v1",

        "split_basis":
            "valid end time of the HRRR 6-h precipitation field",

        "splits_inclusive": {
            name: {
                "start": start,
                "end": end,
            }
            for name, start, end in SPLITS
        },

        "guard_band_days": 10,

        "anchor_forecast_hours":
            list(
                ANCHOR_LEADS
            ),

        "positive_definition": {
            "same_valid_time_as_anchor":
                True,

            "initialization_offsets_hours":
                [-1, 1],

            "require_both_directions":
                True,
        },

        "negative_definition": {
            "same_split_as_anchor":
                True,

            "same_forecast_hour_as_anchor":
                True,

            "same_valid_utc_hour_as_anchor":
                True,

            "minimum_absolute_valid_time_separation_days":
                NEGATIVE_MIN_DAYS,

            "comparison":
                "strictly greater than",

            "same_season_required":
                False,

            "sampled_dynamically":
                True,
        },
    }

    split_json = (
        output_dir
        / "split_definition_v1.json"
    )

    split_json.write_text(
        json.dumps(
            split_definition,
            indent=2,
        )
        + "\n"
    )

    # -------------------------------------------------------------
    # Write primary tables.
    # -------------------------------------------------------------

    anchor_path = (
        output_dir
        / "eligible_anchors_v1.csv.gz"
    )

    negative_path = (
        output_dir
        / "negative_pool_v1.csv.gz"
    )

    anchors.to_csv(
        anchor_path,
        index=False,
        compression="gzip",
    )

    negative_pool.to_csv(
        negative_path,
        index=False,
        compression="gzip",
    )

    # -------------------------------------------------------------
    # Human-readable summaries.
    # -------------------------------------------------------------

    anchor_summary = (
        anchors.groupby(
            [
                "split",
                "anchor_fhr",
            ]
        )
        .agg(
            anchors=(
                "anchor_id",
                "size",
            ),

            median_negative_candidates=(
                "negative_candidate_count",
                "median",
            ),

            min_negative_candidates=(
                "negative_candidate_count",
                "min",
            ),

            max_negative_candidates=(
                "negative_candidate_count",
                "max",
            ),
        )
        .reset_index()
    )

    anchor_summary.to_csv(
        output_dir
        / "eligible_anchor_summary_v1.csv",
        index=False,
    )

    season_summary = (
        anchors.groupby(
            [
                "split",
                "season",
            ]
        )
        .size()
        .rename("anchors")
        .reset_index()
    )

    season_summary.to_csv(
        output_dir
        / "eligible_anchor_counts_by_season_v1.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Final stdout summary.
    # -------------------------------------------------------------

    print()
    print("=" * 78)
    print("DATASET INDEX V1 COMPLETE")
    print("=" * 78)

    print(
        f"Eligible anchors : "
        f"{anchor_path}"
    )

    print(
        f"Negative pool    : "
        f"{negative_path}"
    )

    print(
        f"Split spec       : "
        f"{split_json}"
    )

    print()
    print(
        "Eligible anchors by split:"
    )

    print(
        anchors[
            "split"
        ]
        .value_counts()
        .reindex(
            MODEL_SPLITS,
            fill_value=0,
        )
        .to_string()
    )

    print()
    print(
        "Eligible anchors by split / season:"
    )

    print(
        anchors.groupby(
            [
                "split",
                "season",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            index=MODEL_SPLITS,
            columns=[
                "DJF",
                "MAM",
                "JJA",
                "SON",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print()
    print(
        "Eligible anchors by split / forecast hour:"
    )

    print(
        anchors.groupby(
            [
                "split",
                "anchor_fhr",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            index=MODEL_SPLITS,
            columns=ANCHOR_LEADS,
            fill_value=0,
        )
        .to_string()
    )

    print()
    print(
        "Negative candidate count:"
    )

    print(
        f"  min    = "
        f"{int(anchors['negative_candidate_count'].min()):,}"
    )

    print(
        f"  median = "
        f"{anchors['negative_candidate_count'].median():,.0f}"
    )

    print(
        f"  max    = "
        f"{int(anchors['negative_candidate_count'].max()):,}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
