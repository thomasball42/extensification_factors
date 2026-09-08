"""
Combined diagnostic grid for the animal beta pipeline: every preprocessing
arm (two-way FE / year-only FE / plain demeaning) x every Q-shrinkage prior
candidate, reporting both (a) the resulting full-sample beta magnitude and
(b) leak-safe holdout performance vs. a static OLS baseline, for each
combination -- to see whether any setting lets beta come out higher than the
current production value while the TVP model still beats the static
baseline in holdout validation.

Diagnostic only -- does not touch _compute_beta_animals.py,
_compute_beta_animals_linear.py, or config.json. All model-fitting and
residualization logic is reused unchanged from _stats.py and the existing
validation/_holdout_common*.py / holdout_comparison*.py modules; this script
only adds the grid loop and aggregation/reporting.

fe + prod_default reproduces exactly what outputs/beta_animals.csv and
outputs/validation/holdout_static_comparison_animals_pweighted.csv already
compute -- used below as the anchor row every other combination is compared
against.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # extensification_factors/ root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # validation/

from _stats import fit_tvp_beta
from _holdout_common import residualized_target_series, time_fe_target_series, demeaned_target_series
from _holdout_common_animals import load_area_data
from holdout_comparison import compare_on_series, reindex_levels, MIN_TRAIN_OBS, HOLDOUT_YEARS

MIN_OBS = 15  # matches _compute_beta_animals.py's full-sample fit floor

ARMS = {
    "fe": residualized_target_series,        # two-way (country+year) FE -- production default
    "time_fe": time_fe_target_series,        # year-only FE, no country effect
    "raw": demeaned_target_series,           # plain per-series demeaning, no FE at all
}

Q_PRIOR_CANDIDATES = [
    {"label": "prod_default", "q_prior_scale": 0.02, "q_prior_strength": 0.9},  # current production setting
    {"label": "no_prior",     "q_prior_scale": None, "q_prior_strength": 1.0},  # unregularized MLE ceiling
    {"label": "scale_0.05",   "q_prior_scale": 0.05, "q_prior_strength": 0.9},  # raise the prior mean on Q
    {"label": "strength_1.5", "q_prior_scale": 0.02, "q_prior_strength": 1.5},  # loosen the pull toward the prior
    {"label": "loose_combo",  "q_prior_scale": 0.05, "q_prior_strength": 1.5},  # both loosened together
]

ANCHOR_ARM, ANCHOR_LABEL = "fe", "prod_default"

OUT_DIR: Path = Path("outputs") / "validation" / "animals"
DETAIL_DIR: Path = OUT_DIR / "detail"
SUMMARY_PATH: Path = OUT_DIR / "beta_grid_summary.csv"


def fit_full_sample_betas(area_data, panel, series_fn, q_scale, q_strength):
    """Full-sample (no holdout truncation) TVP beta fit per area, under one
    (arm, prior) setting -- series_fn is called with train_end = full series
    length, so no years are excluded from the preprocessing step, matching
    what the production pipeline would produce under that setting. Returns
    (flat_betas, n_qualified, n_fit): flat_betas is every fitted beta_t
    across every qualifying area concatenated -- its median/mean is directly
    comparable to beta_animals.csv's current value, computed the same way."""
    betas = []
    n_qualified = 0
    n_fit = 0
    for area, entry in area_data.items():
        if len(entry["g"]) < MIN_OBS + 1:
            continue
        n_qualified += 1
        T = len(entry["diff_years"])
        a, p = series_fn(area, entry, panel, T)
        fit = fit_tvp_beta(a, p, min_obs=MIN_OBS, q_prior_scale=q_scale, q_prior_strength=q_strength)
        if fit is None:
            continue
        n_fit += 1
        betas.append(fit["beta"])
    flat = np.concatenate(betas) if betas else np.array([])
    flat = flat[np.isfinite(flat)]
    return flat, n_qualified, n_fit


def score_series_arm(area, entry, panel, series_fn, q_scale, q_strength):
    """Leak-safe holdout comparison for one area under one (arm, prior)
    setting -- mirrors holdout_comparison_animals.score_series, parameterized
    over the preprocessing arm, with the prior passed through explicitly
    since it's the thing under test here (unlike holdout_residualization_
    ablation.py, which only ablates preprocessing at a fixed prior)."""
    g, diff_years = entry["g"], entry["diff_years"]
    if len(g) < MIN_TRAIN_OBS + HOLDOUT_YEARS + 1:
        return None
    T = len(diff_years)
    train_end = T - HOLDOUT_YEARS
    if train_end < MIN_TRAIN_OBS:
        return None
    a, p = series_fn(area, entry, panel, train_end)
    prod_level = reindex_levels(g["Year"].values.astype(int), g["Production"].values)
    out = compare_on_series(a, p, prod_level, train_end, q_scale, q_strength)
    if out is not None:
        out["Area"] = area
    return out


def run_combination(area_data, panel, arm_name, series_fn, prior_cfg):
    q_scale, q_strength = prior_cfg["q_prior_scale"], prior_cfg["q_prior_strength"]

    flat_betas, n_qual, n_fit = fit_full_sample_betas(area_data, panel, series_fn, q_scale, q_strength)

    holdout_rows = []
    for area, entry in area_data.items():
        out = score_series_arm(area, entry, panel, series_fn, q_scale, q_strength)
        if out is not None:
            holdout_rows.append(out)
    holdout_df = pd.DataFrame(holdout_rows)

    summary = {
        "arm": arm_name,
        "prior_label": prior_cfg["label"],
        "q_prior_scale": q_scale,
        "q_prior_strength": q_strength,
        "n_areas_qualified_full_sample": n_qual,
        "n_areas_fit_full_sample": n_fit,
        "n_beta_obs": len(flat_betas),
        "median_beta": float(np.median(flat_betas)) if len(flat_betas) else np.nan,
        "mean_beta": float(np.mean(flat_betas)) if len(flat_betas) else np.nan,
        "n_holdout_series": len(holdout_df),
        "holdout_win_rate": float(holdout_df["tvp_beats_static"].mean()) if len(holdout_df) else np.nan,
        "mean_mae_static": float(holdout_df["mae_static"].mean()) if len(holdout_df) else np.nan,
        "mean_mae_tvp": float(holdout_df["mae_tvp"].mean()) if len(holdout_df) else np.nan,
        "median_beta_static_train": float(holdout_df["beta_static_train"].median()) if len(holdout_df) else np.nan,
    }
    return summary, holdout_df


if __name__ == "__main__":
    area_data, panel = load_area_data()
    print(f"Loaded {len(area_data)} areas.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for arm_name, series_fn in ARMS.items():
        for prior_cfg in Q_PRIOR_CANDIDATES:
            print(f"\n=== arm={arm_name}, prior={prior_cfg['label']} "
                  f"(q_prior_scale={prior_cfg['q_prior_scale']}, q_prior_strength={prior_cfg['q_prior_strength']}) ===")
            summary, holdout_df = run_combination(area_data, panel, arm_name, series_fn, prior_cfg)
            summaries.append(summary)

            print(f"  full-sample: {summary['n_areas_fit_full_sample']}/{summary['n_areas_qualified_full_sample']} areas fit, "
                  f"median beta={summary['median_beta']:.5f}, mean beta={summary['mean_beta']:.5f}")
            print(f"  holdout:     {summary['n_holdout_series']} series, "
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

    print("\n=== Combined grid summary ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    if len(anchor_rows) == 1:
        candidates = summary_df[
            (summary_df["delta_median_beta_vs_prod_default"] > 0)
            & (summary_df["delta_win_rate_vs_prod_default"] >= -0.02)
        ]
        print("\n=== Candidates worth a look (higher beta, holdout win rate within 2pp of production default) ===")
        if len(candidates) > 0:
            print(candidates.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        else:
            print("None found.")

    print(f"\nWritten summary to {SUMMARY_PATH}")
    print(f"Written per-combination holdout detail CSVs to {DETAIL_DIR}")
