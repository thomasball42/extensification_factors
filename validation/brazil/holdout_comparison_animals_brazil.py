"""
TVP vs. static-OLS holdout comparison for Brazil's local immediate-region
pasture panel -- the Brazil analog of ../holdout_comparison.py. For each
qualifying region: hold out the final HOLDOUT_YEARS years, fit (a) the
static OLS beta and (b) the TVP model's (Q, R) using ONLY the training
years, then compare one-step-ahead prediction error on the held-out years
for both.

Reuses ../holdout_comparison.py's compare_on_series/reindex_levels and
../_holdout_common.py's residualized_target_series/demeaned_target_series
verbatim -- none of that logic is FAOSTAT-specific, it only needs two
aligned diff arrays, a production/weight array and a train/test split index
(see _holdout_common_brazil.py, which builds the Brazil-shaped panel these
expect). Brazil has a single series type per region (no FAOSTAT item
dimension), so there's no per-item outer loop and no aggregate-item
filtering, unlike the crops version.

MIN_TRAIN_OBS=6, HOLDOUT_YEARS=2 (vs. crops' 30/5): the local file only
supports <=10 annual diffs per region (2011-2021), and 6+2=8 exactly matches
_compute_beta_animals_brazil.py's own MIN_OBS=8 -- so every region that
qualifies for the production fit also qualifies for this holdout evaluation,
with no additional exclusions. USE_RESIDUALIZATION defaults to False to
match that same script's default, so this validates the model configuration
actually used for the production beta estimates.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent                # .../extensification_factors/validation/brazil
VALIDATION_DIR = FILE_DIR.parent                            # .../extensification_factors/validation
EXT_FACTORS_DIR = VALIDATION_DIR.parent                      # .../extensification_factors

sys.path.insert(0, str(EXT_FACTORS_DIR))
sys.path.insert(0, str(VALIDATION_DIR))
sys.path.insert(0, str(FILE_DIR))

from _utils import load_q_prior_config
from _holdout_common import residualized_target_series, demeaned_target_series
import holdout_comparison as _hc
from _holdout_common_brazil import load_data, load_groups, build_region_data

MIN_TRAIN_OBS = 6
HOLDOUT_YEARS = 2
SAMPLE_N = None            # None = run on every qualifying region (only ~440, cheap)
RANDOM_SEED = 0
PRODUCTION_WEIGHTING = True
USE_RESIDUALIZATION = False  # matches _compute_beta_animals_brazil.py's default

Q_PRIOR_SCALE, Q_PRIOR_STRENGTH = load_q_prior_config()

# compare_on_series() reads MIN_TRAIN_OBS as a free variable resolved against
# the module it was DEFINED in (holdout_comparison.py), not this module's
# namespace -- importing it alone would silently keep enforcing crops'
# MIN_TRAIN_OBS=30, which no Brazil region could ever satisfy. Overriding the
# attribute on the imported module object patches the same globals dict the
# function actually reads from.
_hc.MIN_TRAIN_OBS = MIN_TRAIN_OBS
compare_on_series = _hc.compare_on_series
reindex_levels = _hc.reindex_levels

OUT_PATH = Path("outputs") / "validation" / "brazil" / "holdout_comparison_animals_brazil.csv"
OUT_BY_UF_PATH = Path("outputs") / "validation" / "brazil" / "holdout_comparison_animals_brazil_by_uf.csv"


def score_region(cod_rgi, region_entry, panel, q_prior_scale, q_prior_strength):
    g = region_entry["g"]
    diff_years = region_entry["diff_years"]
    if len(g) < MIN_TRAIN_OBS + HOLDOUT_YEARS + 1:
        return None
    T = len(diff_years)
    train_end = T - HOLDOUT_YEARS
    if train_end < MIN_TRAIN_OBS:
        return None

    if USE_RESIDUALIZATION:
        a, p = residualized_target_series(cod_rgi, region_entry, panel, train_end)
    else:
        a, p = demeaned_target_series(cod_rgi, region_entry, panel, train_end)

    prod_level = reindex_levels(g["year"].values.astype(int), g["P_T"].values)

    out = compare_on_series(a, p, prod_level, train_end, q_prior_scale, q_prior_strength)
    if out is None:
        return None
    current = g.iloc[-1]
    out.update({"uf": current["uf"], "cod_rgi": cod_rgi, "nome_rgi": current["nome_rgi"]})
    return out


def main():
    df = load_data()
    region_ids = load_groups(df, SAMPLE_N, RANDOM_SEED)
    region_data, panel = build_region_data(df)

    results = []
    for i, cod_rgi in enumerate(region_ids):
        region_entry = region_data.get(cod_rgi)
        if region_entry is None:
            continue
        print(i + 1, "/", len(region_ids), cod_rgi)
        out = score_region(cod_rgi, region_entry, panel, Q_PRIOR_SCALE, Q_PRIOR_STRENGTH)
        if out is None:
            continue
        results.append(out)

    res_df = pd.DataFrame(results)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUT_PATH, index=False)

    print(f"\nCompared {len(res_df)} regions (held-out prediction, {HOLDOUT_YEARS} test years each).")
    print("--- Unweighted (every region counts equally) ---")
    print(f"Aggregate MAE -- static baseline: {res_df['mae_static'].mean():.4f}")
    print(f"Aggregate MAE -- TVP:             {res_df['mae_tvp'].mean():.4f}")
    print(f"Share of regions where TVP beats static baseline: {res_df['tvp_beats_static'].mean():.1%}")

    win = res_df[res_df["tvp_beats_static"]]
    lose = res_df[~res_df["tvp_beats_static"]]
    if len(win) > 0:
        abs_imp = win["mae_static"] - win["mae_tvp"]
        rel_imp = abs_imp / win["mae_static"]
        print(f"Among TVP winners ({len(win)} regions): mean/median MAE reduction "
              f"{abs_imp.mean():.4f}/{abs_imp.median():.4f} ({rel_imp.mean():.1%}/{rel_imp.median():.1%} relative)")
    if len(lose) > 0:
        abs_worse = lose["mae_tvp"] - lose["mae_static"]
        rel_worse = abs_worse / lose["mae_static"]
        print(f"Among TVP losers ({len(lose)} regions):  mean/median MAE increase  "
              f"{abs_worse.mean():.4f}/{abs_worse.median():.4f} ({rel_worse.mean():.1%}/{rel_worse.median():.1%} relative)")

    by_uf = None
    if PRODUCTION_WEIGHTING:
        w = res_df["test_avg_production"].values
        usable = w > 0
        if usable.sum() == 0:
            print("\nProduction weighting requested but no region had a usable production weight -- skipped.")
        else:
            wsub = res_df.loc[usable]
            weights = wsub["test_avg_production"].values
            print(f"\n--- Production-weighted, pooled across all regions ---")
            print(f"({usable.sum()} of {len(res_df)} regions had a usable weight)")
            print(f"Weighted MAE -- static baseline: {np.average(wsub['mae_static'], weights=weights):.4f}")
            print(f"Weighted MAE -- TVP:             {np.average(wsub['mae_tvp'], weights=weights):.4f}")
            print(f"Production-weighted share where TVP beats static: "
                  f"{np.average(wsub['tvp_beats_static'], weights=weights):.1%}")

            def weighted_uf_stats(g):
                wt = g["test_avg_production"].values
                if wt.sum() <= 0:
                    return pd.Series({
                        "n_regions": len(g), "total_production_weight": 0.0,
                        "weighted_mae_static": np.nan, "weighted_mae_tvp": np.nan,
                        "weighted_tvp_win_share": np.nan,
                    })
                return pd.Series({
                    "n_regions": len(g),
                    "total_production_weight": wt.sum(),
                    "weighted_mae_static": np.average(g["mae_static"], weights=wt),
                    "weighted_mae_tvp": np.average(g["mae_tvp"], weights=wt),
                    "weighted_tvp_win_share": np.average(g["tvp_beats_static"], weights=wt),
                })

            by_uf = wsub.groupby("uf").apply(weighted_uf_stats, include_groups=False)
            by_uf = by_uf.sort_values("total_production_weight", ascending=False)
            OUT_BY_UF_PATH.parent.mkdir(parents=True, exist_ok=True)
            by_uf.to_csv(OUT_BY_UF_PATH)
            print(f"\nPer-state (uf) weighted breakdown:")
            print(by_uf.to_string())
            print(f"\nFull per-state breakdown written to {OUT_BY_UF_PATH}")

    print(f"\nWritten to {OUT_PATH}")
    return res_df, by_uf


if __name__ == "__main__":
    main()
