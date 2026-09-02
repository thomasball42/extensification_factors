"""
Comparison of TVP vs. static OLS
baseline, via held-out cross-validation

For each qualifying series: hold out the final HOLDOUT_YEARS years, fit (a)
the static OLS beta and (b) the TVP model's (Q, R) using ONLY the training
years, then compare one-step-ahead prediction error on the held-out years
for both.

Both models are fit on the same series the main pipeline (_compute_beta_crops.py)
actually fits: log(Area harvested)/log(Production) diffs residualized against
two-way (country + year) fixed effects -- not raw diffs, which would validate
a different data-generating process than the one that actually produces
beta_crops.csv. The residualization for a given target country excludes that
country's own held-out years from the fixed-effects fit, so its own held-out
values can't leak into its own entity effect; other countries' contemporaneous
data is used as-is. See _holdout_common.residualized_target_series and
_stats.residualize_two_way_fe_oos.

For tractability this runs on a random sample of series by default (SAMPLE_N),
matching the convention used in comparable gridded-dataset validation papers
(e.g. cross-validating on a random subset of stations rather than all of
them). Set SAMPLE_N = None for a full run.

Production weighting weights the results using avg production over the timeseries

"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _stats import kalman_filter_diag, q_shrinkage_penalty
from _utils import load_aggregate_matcher
from _holdout_common import load_wide, load_groups, build_item_area_data, residualized_target_series

MIN_TRAIN_OBS = 30
HOLDOUT_YEARS = 5
SAMPLE_N = None       # None
RANDOM_SEED = 0
PRODUCTION_WEIGHTING = True
Q_PRIOR_SCALE = 0.02      # None = no shrinkage (unchanged default behaviour). Set e.g. 0.001 to enable.
Q_PRIOR_STRENGTH = 0.9     # std of log(Q) around the prior mean -- smaller = more shrinkage.

is_aggregate = load_aggregate_matcher()

OUT_PATH: Path = Path("outputs") / "validation" / f"holdout_static_comparison{"_pweighted" if PRODUCTION_WEIGHTING else ""}.csv"
OUT_BY_ITEM_PATH: Path = Path("outputs") / "validation" / f"holdout_static_comparison_by_item{"_pweighted" if PRODUCTION_WEIGHTING else ""}.csv"
OUT_AGGREGATES_PATH: Path = Path("outputs") / "validation" / f"holdout_static_comparison_aggregates{"_pweighted" if PRODUCTION_WEIGHTING else ""}.csv"


def reindex_levels(years, values):
    """Reindex a raw (non-differenced) series onto the full calendar-year
    range, aligned with build_raw_diffs' diff_years (= full_years[1:]).
    Used only to compute production weights for reporting -- not part of
    any model fit, so this has no bearing on leakage/estimation validity."""
    full_years = np.arange(int(years.min()), int(years.max()) + 1)
    s = pd.Series(values, index=years).reindex(full_years)
    return s.values[1:]


def compare_on_series(a, p, prod_level, train_end, q_prior_scale=None, q_prior_strength=1.0):
    """a, p: two-way (country+year) fixed-effects-residualized diffs for one
    series (see residualized_target_series) -- already centred by the
    residualization step, so no further demeaning happens here."""
    T = len(a)
    valid = np.isfinite(a) & np.isfinite(p)
    train_valid = valid.copy()
    train_valid[train_end:] = False
    if train_valid.sum() < MIN_TRAIN_OBS or not valid[train_end:].any():
        return None

    a_tr, p_tr = a[train_valid], p[train_valid]
    var_p_tr = np.var(p_tr)
    if not (np.isfinite(var_p_tr) and var_p_tr > 0):
        return None

    # --- static OLS baseline, fit on train only, applied to test ---
    beta_static = np.sum(p_tr * a_tr) / np.sum(p_tr * p_tr)  # demeaned -> Cov/Var form

    test_mask = np.zeros(T, dtype=bool)
    test_mask[train_end:] = True
    test_mask &= valid
    a_test, p_test = a[test_mask], p[test_mask]

    resid_static = a_test - beta_static * p_test
    mae_static = float(np.mean(np.abs(resid_static)))
    bias_static = float(np.mean(resid_static))

    # --- TVP model, (Q,R) fit on train only, evaluated on test ---
    var_a_tr = np.var(a_tr)
    if not (np.isfinite(var_a_tr) and var_a_tr > 0):
        return None

    def neg_ll(log_qr):
        Q, R = np.exp(log_qr)
        *_, ll = kalman_filter_diag(a, p, Q, R, fit_upto=train_end)
        if not np.isfinite(ll):
            return 1e10
        return -ll + q_shrinkage_penalty(Q, var_a_tr, q_prior_scale, q_prior_strength)

    res = minimize(neg_ll, np.log([0.05 * var_a_tr, var_a_tr]),
                    method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 500})
    if not res.success:
        return None
    Q_hat, R_hat = np.exp(res.x)
    _, v, F, _ = kalman_filter_diag(a, p, Q_hat, R_hat, fit_upto=T)
    v_test = v[test_mask]
    mae_tvp = float(np.mean(np.abs(v_test)))
    bias_tvp = float(np.mean(v_test))

    # --- production weight: mean production level over the held-out test years ---
    test_avg_production = float(np.nanmean(prod_level[test_mask]))
    if not np.isfinite(test_avg_production):
        test_avg_production = 0.0

    return {
        "n_test": int(test_mask.sum()),
        "Q_hat": Q_hat, "R_hat": R_hat,
        "beta_static_train": beta_static,
        "mae_static": mae_static, "bias_static": bias_static,
        "mae_tvp": mae_tvp, "bias_tvp": bias_tvp,
        "tvp_beats_static": mae_tvp < mae_static,
        "test_avg_production": test_avg_production,
    }


def score_series(area, item, item_code, area_entry, panel, q_prior_scale=None, q_prior_strength=1.0):
    """Residualize `area`'s series (against the item's cross-country panel,
    holding out its own test years from the fixed-effects fit) and run the
    held-out comparison on it. Returns the tagged result dict, or None if
    the series doesn't qualify (too short, degenerate variance, etc.)."""
    g = area_entry["g"]
    diff_years = area_entry["diff_years"]
    if len(g) < MIN_TRAIN_OBS + HOLDOUT_YEARS + 1:
        return None
    T = len(diff_years)
    train_end = T - HOLDOUT_YEARS
    if train_end < MIN_TRAIN_OBS:
        return None

    a, p = residualized_target_series(area, area_entry, panel, train_end)
    prod_level = reindex_levels(g["Year"].values.astype(int), g["Production"].values)

    out = compare_on_series(a, p, prod_level, train_end, q_prior_scale, q_prior_strength)
    if out is None:
        return None
    out.update({"Area": area, "Item": item, "Item Code": item_code, "is_aggregate": is_aggregate(item)})
    return out


def run_holdout_comparison(groups, wide, q_prior_scale=None, q_prior_strength=1.0,
                            out_path=None, out_by_item_path=None, out_aggregates_path=None,
                            production_weighting=False, verbose=True):
    """Score every group under the given (q_prior_scale, q_prior_strength),
    print the usual summary (if verbose), optionally write CSVs, and return
    (res_df, agg_df, by_item_df) for programmatic use (by_item_df is None
    unless production_weighting is True and weights are usable).

    Groups are processed one Item at a time so each item's cross-country
    panel (needed for fixed-effects residualization) is built once and
    reused across every target country scored within that item."""
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
            if verbose:
                print(done, "/", total, area, item, item_code)
            area_entry = area_data.get(area)
            if area_entry is None:
                continue
            out = score_series(area, item, item_code, area_entry, panel, q_prior_scale, q_prior_strength)
            if out is None:
                continue
            results.append(out)

    all_res_df = pd.DataFrame(results)

    # FAOSTAT aggregate items (e.g. "Cereals, primary") double-count their
    # component crops (e.g. "Wheat", "Maize"), so they are excluded from the
    # summary stats and production weighting below -- including them would
    # double-count both the series in unweighted averages and the tonnage in
    # production-weighted ones. They're written to their own file for reference.
    agg_df = all_res_df[all_res_df["is_aggregate"]]
    res_df = all_res_df[~all_res_df["is_aggregate"]].reset_index(drop=True)

    if len(agg_df) > 0:
        if out_aggregates_path is not None:
            out_aggregates_path.parent.mkdir(parents=True, exist_ok=True)
            agg_df.to_csv(out_aggregates_path, index=False)
        if verbose:
            suffix = f" (written to {out_aggregates_path})" if out_aggregates_path is not None else ""
            print(f"Excluded {len(agg_df)} aggregate-item series from calculations{suffix}.")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(out_path, index=False)

    if verbose:
        print(f"\nCompared {len(res_df)} series (held-out prediction, {HOLDOUT_YEARS} test years each).")
        print(f"--- Unweighted (every series counts equally, regardless of size) ---")
        print(f"Aggregate MAE -- static baseline: {res_df['mae_static'].mean():.4f}")
        print(f"Aggregate MAE -- TVP:             {res_df['mae_tvp'].mean():.4f}")
        print(f"Share of series where TVP beats static baseline: {res_df['tvp_beats_static'].mean():.1%}")

        win = res_df[res_df["tvp_beats_static"]]
        lose = res_df[~res_df["tvp_beats_static"]]
        if len(win) > 0:
            abs_imp = win["mae_static"] - win["mae_tvp"]
            rel_imp = abs_imp / win["mae_static"]
            print(f"Among TVP winners ({len(win)} series): mean/median MAE reduction "
                  f"{abs_imp.mean():.4f}/{abs_imp.median():.4f} ({rel_imp.mean():.1%}/{rel_imp.median():.1%} relative)")
        if len(lose) > 0:
            abs_worse = lose["mae_tvp"] - lose["mae_static"]
            rel_worse = abs_worse / lose["mae_static"]
            print(f"Among TVP losers ({len(lose)} series):  mean/median MAE increase  "
                  f"{abs_worse.mean():.4f}/{abs_worse.median():.4f} ({rel_worse.mean():.1%}/{rel_worse.median():.1%} relative)")

    by_item = None
    if production_weighting:
        w = res_df["test_avg_production"].values
        usable = w > 0
        if usable.sum() == 0:
            if verbose:
                print("\nProduction weighting requested but no series had usable production weights -- skipped.")
        else:
            wsub = res_df.loc[usable]
            weights = wsub["test_avg_production"].values
            if verbose:
                print(f"\n--- Production-weighted, POOLED across all commodities ---")
                print(f"({usable.sum()} of {len(res_df)} series had a usable weight; caveat: pooling raw tonnage")
                print(f" across different commodities is not fully fair -- see module docstring)")
                print(f"Weighted MAE -- static baseline: {np.average(wsub['mae_static'], weights=weights):.4f}")
                print(f"Weighted MAE -- TVP:             {np.average(wsub['mae_tvp'], weights=weights):.4f}")
                print(f"Weighted bias -- static baseline: {np.average(wsub['bias_static'], weights=weights):.4f}")
                print(f"Weighted bias -- TVP:             {np.average(wsub['bias_tvp'], weights=weights):.4f}")
                print(f"Production-weighted share where TVP beats static: "
                      f"{np.average(wsub['tvp_beats_static'], weights=weights):.1%}")

                wwin = wsub[wsub["tvp_beats_static"]]
                if len(wwin) > 0:
                    wwin_weights = wwin["test_avg_production"].values
                    wabs_imp = wwin["mae_static"] - wwin["mae_tvp"]
                    wrel_imp = wabs_imp / wwin["mae_static"]
                    print(f"Among TVP winners, production-weighted ({len(wwin)} series): "
                          f"mean MAE reduction {np.average(wabs_imp, weights=wwin_weights):.4f} "
                          f"({np.average(wrel_imp, weights=wwin_weights):.1%} relative)")

            # --- per-Item weighted breakdown: avoids pooling raw tonnage across
            # heterogeneous commodities; weight is only ever compared within the
            # same Item, so a bulky commodity can't distort another's comparison ---
            def weighted_item_stats(g):
                wt = g["test_avg_production"].values
                if wt.sum() <= 0:
                    return pd.Series({
                        "n_series": len(g), "total_production_weight": 0.0,
                        "weighted_mae_static": np.nan, "weighted_mae_tvp": np.nan,
                        "weighted_tvp_win_share": np.nan,
                    })
                return pd.Series({
                    "n_series": len(g),
                    "total_production_weight": wt.sum(),
                    "weighted_mae_static": np.average(g["mae_static"], weights=wt),
                    "weighted_mae_tvp": np.average(g["mae_tvp"], weights=wt),
                    "weighted_tvp_win_share": np.average(g["tvp_beats_static"], weights=wt),
                })

            by_item = wsub.groupby("Item").apply(weighted_item_stats, include_groups=False)
            by_item = by_item.sort_values("total_production_weight", ascending=False)
            if out_by_item_path is not None:
                out_by_item_path.parent.mkdir(parents=True, exist_ok=True)
                by_item.to_csv(out_by_item_path)
            if verbose:
                print(f"\nPer-item weighted breakdown (top 10 by total production weight):")
                print(by_item.head(10).to_string())
                if out_by_item_path is not None:
                    print(f"\nFull per-item breakdown written to {out_by_item_path}")

    if verbose and out_path is not None:
        print(f"\nWritten to {out_path}")

    return res_df, agg_df, by_item


if __name__ == "__main__":
    wide = load_wide()
    groups = load_groups(wide, SAMPLE_N, RANDOM_SEED)
    run_holdout_comparison(
        groups, wide, Q_PRIOR_SCALE, Q_PRIOR_STRENGTH,
        out_path=OUT_PATH, out_by_item_path=OUT_BY_ITEM_PATH, out_aggregates_path=OUT_AGGREGATES_PATH,
        production_weighting=PRODUCTION_WEIGHTING,
    )
