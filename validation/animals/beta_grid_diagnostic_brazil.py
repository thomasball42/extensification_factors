"""
Brazil-regional analogue of beta_grid_diagnostic.py: the same combined
preprocessing-arm x Q-prior grid, applied to Brazil's own regional
(immediate-region, cod_rgi) pasture/production panel
(data/inputs/brazil/NL_local_domain.csv) instead of the national FAOSTAT
series -- to see whether the same settings exploration changes anything at
this much finer spatial resolution and much shorter time series (2011-2021,
so at most 10 annual diffs per region, vs 60+ nationally).

The production Brazil pipeline (_compute_beta_animals_brazil.py) currently
runs with USE_RESIDUALIZATION=False (arm "raw" here) and the same Q-prior as
the national pipeline (scale=0.02, strength=0.9, from config.json) -- that
combination is this script's anchor row, NOT "fe"+prod_default like
beta_grid_diagnostic.py's anchor (which matches national's actual default,
USE_RESIDUALIZATION=True).

A prior holdout-validation setup for Brazil (_holdout_common_brazil.py,
holdout_comparison_animals_brazil.py) was deleted from the codebase in a
recent commit alongside a pivot toward validation/brazil/compare_beta_
animals.py's cross-resolution distributional comparison (regional beta
distribution vs. the single national value, no train/test split at all).
There's no commit message explaining why, but the short regional series is
the likely reason: holding out even 2 years after fitting Q/R by MLE on the
remaining ~8 points is much noisier than holding out 5 of 60+ years
nationally. This script rebuilds an equivalent holdout comparison anyway
(per explicit request), sized down for the short series (HOLDOUT_YEARS=2,
MIN_TRAIN_OBS=6) -- read its win-rate/MAE numbers with that noise floor in
mind; they are not as statistically solid as the national grid's.

All model-fitting/residualization logic is reused unchanged from _stats.py
and validation/_holdout_common.py / holdout_comparison.py (both fully
generic over the panel's entity/time columns, so they apply to a
region-level panel exactly as they do to a country-level one) -- this
script only adds Brazil-specific data loading and the grid loop/
aggregation, mirroring beta_grid_diagnostic.py's structure.

Diagnostic only -- does not touch _compute_beta_animals_brazil.py,
_compute_beta_animals_brazil_linear.py, or config.json.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # extensification_factors/ root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # validation/

from _stats import fit_tvp_beta, build_raw_diffs
from _holdout_common import residualized_target_series, time_fe_target_series, demeaned_target_series
import holdout_comparison
from holdout_comparison import compare_on_series, reindex_levels

DATA_PATH: Path = Path("data") / "inputs" / "brazil" / "NL_local_domain.csv"
AREA_COL = "area_past_ha"
PRODUCTION_COL = "P_T"

MIN_OBS = 8         # matches _compute_beta_animals_brazil.py's full-sample fit floor
HOLDOUT_YEARS = 2    # sized down from the national 5 -- series here run ~10 diff-years, not 60+
MIN_TRAIN_OBS = 6    # sized down from the national 30, for the same reason

# compare_on_series (imported above) reads MIN_TRAIN_OBS from holdout_comparison's
# own module globals at call time, not from this script's local constant of the
# same name -- its internal `train_valid.sum() < MIN_TRAIN_OBS` check would
# otherwise still enforce crops' threshold of 30, which no Brazil region (at
# most ~8 training points) could ever clear. Patch the shared module global
# in place so the already-imported compare_on_series function picks it up.
holdout_comparison.MIN_TRAIN_OBS = MIN_TRAIN_OBS

ARMS = {
    "fe": residualized_target_series,        # two-way (region+year) FE
    "time_fe": time_fe_target_series,        # year-only FE
    "raw": demeaned_target_series,           # plain per-series demeaning -- Brazil's current production default
}

Q_PRIOR_CANDIDATES = [
    {"label": "prod_default", "q_prior_scale": 0.02, "q_prior_strength": 0.9},  # current production setting (same config.json as national)
    {"label": "no_prior",     "q_prior_scale": None, "q_prior_strength": 1.0},
    {"label": "scale_0.05",   "q_prior_scale": 0.05, "q_prior_strength": 0.9},
    {"label": "strength_1.5", "q_prior_scale": 0.02, "q_prior_strength": 1.5},
    {"label": "loose_combo",  "q_prior_scale": 0.05, "q_prior_strength": 1.5},
]

ANCHOR_ARM, ANCHOR_LABEL = "raw", "prod_default"  # matches _compute_beta_animals_brazil.py's actual current settings

OUT_DIR: Path = Path("outputs") / "validation" / "animals"
DETAIL_DIR: Path = OUT_DIR / "detail_brazil"
SUMMARY_PATH: Path = OUT_DIR / "beta_grid_summary_brazil.csv"


def load_region_data():
    """Brazil-regional analogue of _holdout_common_animals.load_area_data:
    returns (region_data, panel) with the same shape (region_data[cod_rgi] =
    {"g", "diff_years"}; panel has entity/time/a/p columns), so the generic
    arm functions in _holdout_common.py apply unchanged. Mirrors
    _compute_beta_animals_brazil.py's own data-prep exactly (same columns,
    same validity filter, same MIN_OBS floor)."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df["year"] = df["year"].astype(int)

    region_data = {}
    panel_rows = []
    for cod_rgi, g in df.groupby("cod_rgi"):
        g = g.sort_values("year")
        g = g[np.isfinite(g[AREA_COL]) & np.isfinite(g[PRODUCTION_COL])]
        g = g[(g[AREA_COL] > 0) & (g[PRODUCTION_COL] > 0)]
        if len(g) < MIN_OBS + 1:
            continue
        years = g["year"].values
        with np.errstate(divide="ignore"):
            log_area = np.log(g[AREA_COL].values)
            log_prod = np.log(g[PRODUCTION_COL].values)
        diff_years, a_t, p_t = build_raw_diffs(years, log_area, log_prod)
        region_data[cod_rgi] = {"g": g, "diff_years": diff_years}
        panel_rows.append(pd.DataFrame({"entity": cod_rgi, "time": diff_years, "a": a_t, "p": p_t}))

    panel = pd.concat(panel_rows, ignore_index=True) if panel_rows else pd.DataFrame(columns=["entity", "time", "a", "p"])
    return region_data, panel


def fit_full_sample_betas(region_data, panel, series_fn, q_scale, q_strength):
    """Full-sample (no holdout truncation) TVP beta fit per region -- see
    beta_grid_diagnostic.py's function of the same name for the rationale.
    Every region in region_data already cleared MIN_OBS+1 in load_region_data,
    so no re-filtering is needed here."""
    betas = []
    n_fit = 0
    for region, entry in region_data.items():
        T = len(entry["diff_years"])
        a, p = series_fn(region, entry, panel, T)
        fit = fit_tvp_beta(a, p, min_obs=MIN_OBS, q_prior_scale=q_scale, q_prior_strength=q_strength)
        if fit is None:
            continue
        n_fit += 1
        betas.append(fit["beta"])
    flat = np.concatenate(betas) if betas else np.array([])
    flat = flat[np.isfinite(flat)]
    return flat, n_fit


def score_series_arm(region, entry, panel, series_fn, q_scale, q_strength):
    """Leak-safe holdout comparison for one region under one (arm, prior)
    setting -- same logic as beta_grid_diagnostic.py's score_series_arm, with
    Brazil's own column names and the shortened HOLDOUT_YEARS/MIN_TRAIN_OBS."""
    g, diff_years = entry["g"], entry["diff_years"]
    if len(g) < MIN_TRAIN_OBS + HOLDOUT_YEARS + 1:
        return None
    T = len(diff_years)
    train_end = T - HOLDOUT_YEARS
    if train_end < MIN_TRAIN_OBS:
        return None
    a, p = series_fn(region, entry, panel, train_end)
    prod_level = reindex_levels(g["year"].values.astype(int), g[PRODUCTION_COL].values)
    out = compare_on_series(a, p, prod_level, train_end, q_scale, q_strength)
    if out is not None:
        out["region"] = region
    return out


def run_combination(region_data, panel, arm_name, series_fn, prior_cfg):
    q_scale, q_strength = prior_cfg["q_prior_scale"], prior_cfg["q_prior_strength"]

    flat_betas, n_fit = fit_full_sample_betas(region_data, panel, series_fn, q_scale, q_strength)

    holdout_rows = []
    for region, entry in region_data.items():
        out = score_series_arm(region, entry, panel, series_fn, q_scale, q_strength)
        if out is not None:
            holdout_rows.append(out)
    holdout_df = pd.DataFrame(holdout_rows)

    summary = {
        "arm": arm_name,
        "prior_label": prior_cfg["label"],
        "q_prior_scale": q_scale,
        "q_prior_strength": q_strength,
        "n_regions_total": len(region_data),
        "n_regions_fit_full_sample": n_fit,
        "n_beta_obs": len(flat_betas),
        "median_beta": float(np.median(flat_betas)) if len(flat_betas) else np.nan,
        "mean_beta": float(np.mean(flat_betas)) if len(flat_betas) else np.nan,
        "n_holdout_regions": len(holdout_df),
        "holdout_win_rate": float(holdout_df["tvp_beats_static"].mean()) if len(holdout_df) else np.nan,
        "mean_mae_static": float(holdout_df["mae_static"].mean()) if len(holdout_df) else np.nan,
        "mean_mae_tvp": float(holdout_df["mae_tvp"].mean()) if len(holdout_df) else np.nan,
        "median_beta_static_train": float(holdout_df["beta_static_train"].median()) if len(holdout_df) else np.nan,
    }
    return summary, holdout_df


if __name__ == "__main__":
    region_data, panel = load_region_data()
    print(f"Loaded {len(region_data)} regions (MIN_OBS={MIN_OBS}, HOLDOUT_YEARS={HOLDOUT_YEARS}, MIN_TRAIN_OBS={MIN_TRAIN_OBS}).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for arm_name, series_fn in ARMS.items():
        for prior_cfg in Q_PRIOR_CANDIDATES:
            print(f"\n=== arm={arm_name}, prior={prior_cfg['label']} "
                  f"(q_prior_scale={prior_cfg['q_prior_scale']}, q_prior_strength={prior_cfg['q_prior_strength']}) ===")
            summary, holdout_df = run_combination(region_data, panel, arm_name, series_fn, prior_cfg)
            summaries.append(summary)

            print(f"  full-sample: {summary['n_regions_fit_full_sample']}/{summary['n_regions_total']} regions fit, "
                  f"median beta={summary['median_beta']:.5f}, mean beta={summary['mean_beta']:.5f}")
            print(f"  holdout:     {summary['n_holdout_regions']} regions, "
                  f"win rate={summary['holdout_win_rate']:.1%}, "
                  f"mae_static={summary['mean_mae_static']:.4f}, mae_tvp={summary['mean_mae_tvp']:.4f}")

            detail_path = DETAIL_DIR / f"holdout_{arm_name}__{prior_cfg['label']}.csv"
            holdout_df.to_csv(detail_path, index=False)

    summary_df = pd.DataFrame(summaries)

    anchor_rows = summary_df[(summary_df["arm"] == ANCHOR_ARM) & (summary_df["prior_label"] == ANCHOR_LABEL)]
    if len(anchor_rows) == 1:
        anchor = anchor_rows.iloc[0]
        summary_df["delta_median_beta_vs_prod_default"] = summary_df["median_beta"] - anchor["median_beta"]
        summary_df["delta_win_rate_vs_prod_default"] = summary_df["holdout_win_rate"] - anchor["holdout_win_rate"]
    else:
        summary_df["delta_median_beta_vs_prod_default"] = np.nan
        summary_df["delta_win_rate_vs_prod_default"] = np.nan

    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\n=== Combined grid summary (Brazil regional) ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    if len(anchor_rows) == 1:
        candidates = summary_df[
            (summary_df["delta_median_beta_vs_prod_default"] > 0)
            & (summary_df["delta_win_rate_vs_prod_default"] >= -0.02)
        ]
        print("\n=== Candidates worth a look (higher beta, holdout win rate within 2pp of "
              "Brazil's current production default [raw, prod_default]) ===")
        if len(candidates) > 0:
            print(candidates.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        else:
            print("None found.")

    print(f"\nWritten summary to {SUMMARY_PATH}")
    print(f"Written per-combination holdout detail CSVs to {DETAIL_DIR}")
