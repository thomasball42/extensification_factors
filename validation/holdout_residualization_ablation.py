"""
Ablation: does two-way (country + year) fixed-effects residualization
actually improve held-out predictive accuracy, compared to plain per-series
demeaning (the USE_RESIDUALIZATION=False fallback in _compute_beta_crops.py)?
And if two-way FE doesn't help, is that because the YEAR effect (absorbing
global shocks) isn't pulling its weight, or because the COUNTRY (entity)
effect is adding noise on top of a year effect that's fine on its own?

For each qualifying series, this runs the exact same held-out comparison as
holdout_comparison.py (see that module's docstring for the static-vs-TVP
methodology) once per preprocessing ARM below, all on the same train/test
split, all reusing holdout_comparison.compare_on_series unchanged -- so the
only thing that differs between arms is the preprocessing step under test:

  fe       two-way (country + year) FE residualization -- what the
           production pipeline actually uses (_holdout_common.residualized_target_series)
  time_fe  year-only FE residualization, no country effect
           (_holdout_common.time_fe_target_series)
  raw      plain per-series demeaning, no fixed effects at all -- what
           USE_RESIDUALIZATION=False falls back to
           (_holdout_common.demeaned_target_series)

Two comparisons are reported per pair of arms:
  1. Primary: TVP model MAE -- does the production model's own prediction
     accuracy improve going raw -> time_fe -> fe?
  2. Secondary: static-OLS-baseline MAE -- is any effect specific to the TVP
     fit, or does it show up for a naive model too (i.e. is it about the
     underlying data-generating process)?

For tractability this runs on a random sample of series by default
(SAMPLE_N). Set SAMPLE_N = None for a full run.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _utils import load_aggregate_matcher
from _holdout_common import (
    load_wide, load_groups, build_item_area_data,
    residualized_target_series, time_fe_target_series, demeaned_target_series,
)
from holdout_comparison import compare_on_series, reindex_levels

MIN_TRAIN_OBS = 50
HOLDOUT_YEARS = 5
SAMPLE_N = None       # None = run on every qualifying series
RANDOM_SEED = 0

ARMS = {
    "fe": residualized_target_series,
    "time_fe": time_fe_target_series,
    "raw": demeaned_target_series,
}
PER_ARM_KEYS = ["Q_hat", "R_hat", "beta_static_train", "mae_static", "bias_static",
                "mae_tvp", "bias_tvp", "tvp_beats_static"]
# pairs to report a head-to-head win share for (winner listed first)
PAIRS = [("fe", "raw"), ("time_fe", "raw"), ("fe", "time_fe")]

is_aggregate = load_aggregate_matcher()

OUT_PATH: Path = Path("outputs") / "validation" / "holdout_residualization_ablation.csv"


def score_series_all(area, item, item_code, area_entry, panel):
    """Run compare_on_series once per arm in ARMS on the same train/test
    split, returning a single row with <arm>_* prefixed metrics plus
    pairwise win flags, or None if the series doesn't qualify under any arm."""
    g = area_entry["g"]
    diff_years = area_entry["diff_years"]
    if len(g) < MIN_TRAIN_OBS + HOLDOUT_YEARS + 1:
        return None
    T = len(diff_years)
    train_end = T - HOLDOUT_YEARS
    if train_end < MIN_TRAIN_OBS:
        return None

    prod_level = reindex_levels(g["Year"].values.astype(int), g["Production"].values)

    arm_outs = {}
    for arm_name, series_fn in ARMS.items():
        a, p = series_fn(area, area_entry, panel, train_end)
        out = compare_on_series(a, p, prod_level, train_end)
        if out is None:
            return None
        arm_outs[arm_name] = out

    row = {"Area": area, "Item": item, "Item Code": item_code, "is_aggregate": is_aggregate(item),
           "n_test": arm_outs["fe"]["n_test"], "test_avg_production": arm_outs["fe"]["test_avg_production"]}
    for arm_name, out in arm_outs.items():
        row.update({f"{arm_name}_{k}": out[k] for k in PER_ARM_KEYS})
    for winner, loser in PAIRS:
        row[f"{winner}_beats_{loser}_tvp"] = arm_outs[winner]["mae_tvp"] < arm_outs[loser]["mae_tvp"]
        row[f"{winner}_beats_{loser}_static"] = arm_outs[winner]["mae_static"] < arm_outs[loser]["mae_static"]
    return row


def summarize_arm_comparison(res_df, label, fe_col, raw_col, win_col):
    print(f"\n--- {label} ---")
    print(f"Mean MAE -- {fe_col}: {res_df[fe_col].mean():.4f}")
    print(f"Mean MAE -- {raw_col}: {res_df[raw_col].mean():.4f}")
    print(f"Share of series where {fe_col} beats {raw_col}: {res_df[win_col].mean():.1%}")

    win = res_df[res_df[win_col]]
    lose = res_df[~res_df[win_col]]
    if len(win) > 0:
        abs_imp = win[raw_col] - win[fe_col]
        rel_imp = abs_imp / win[raw_col]
        print(f"Among winners ({len(win)} series): mean/median MAE reduction "
              f"{abs_imp.mean():.4f}/{abs_imp.median():.4f} ({rel_imp.mean():.1%}/{rel_imp.median():.1%} relative)")
    if len(lose) > 0:
        abs_worse = lose[fe_col] - lose[raw_col]
        rel_worse = abs_worse / lose[raw_col]
        print(f"Among losers ({len(lose)} series):  mean/median MAE increase  "
              f"{abs_worse.mean():.4f}/{abs_worse.median():.4f} ({rel_worse.mean():.1%}/{rel_worse.median():.1%} relative)")


if __name__ == "__main__":
    wide = load_wide()
    groups = load_groups(wide, SAMPLE_N, RANDOM_SEED)

    by_item = {}
    for (area, item, item_code), _ in groups:
        by_item.setdefault((item, item_code), []).append(area)
    total = sum(len(areas) for areas in by_item.values())

    results = []
    done = 0
    for (item, item_code), target_areas in by_item.items():
        g_item = wide[(wide["Item"] == item) & (wide["Item Code"] == item_code)]
        area_data, panel = build_item_area_data(g_item)
        for area in target_areas:
            done += 1
            print(done, "/", total, area, item, item_code)
            area_entry = area_data.get(area)
            if area_entry is None:
                continue
            out = score_series_all(area, item, item_code, area_entry, panel)
            if out is None:
                continue
            results.append(out)

    all_res_df = pd.DataFrame(results)

    # FAOSTAT aggregate items double-count their component crops -- excluded
    # from summary stats, same convention as holdout_comparison.py.
    res_df = all_res_df[~all_res_df["is_aggregate"]].reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUT_PATH, index=False)

    print(f"\nCompared {len(res_df)} series (held-out prediction, {HOLDOUT_YEARS} test years each) "
          f"across arms: {', '.join(ARMS)}.")

    for winner, loser in PAIRS:
        summarize_arm_comparison(res_df, f"Primary (TVP model): {winner} vs. {loser}",
                                  f"{winner}_mae_tvp", f"{loser}_mae_tvp", f"{winner}_beats_{loser}_tvp")
    for winner, loser in PAIRS:
        summarize_arm_comparison(res_df, f"Secondary (static-OLS baseline): {winner} vs. {loser}",
                                  f"{winner}_mae_static", f"{loser}_mae_static", f"{winner}_beats_{loser}_static")

    print(f"\nWritten to {OUT_PATH}")
