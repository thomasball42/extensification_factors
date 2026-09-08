"""
Exploring per-product beta for animal products - wanted to test if individual
products might be 'driving' changes in pasture area. Turns out this isn't the 
case, values stay pretty much the same as before. 

Uses the same preprocessing/prior setting as production (two-way country+
year FE residualization, Q_PRIOR_SCALE/Q_PRIOR_STRENGTH from
holdout_comparison.py, currently 0.02/0.9) since that combination was
already found to be the best-performing arm in beta_grid_diagnostic.py.

Diagnostic only -- does not touch _compute_beta_animals.py or config.json.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # extensification_factors/ root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # validation/

from _stats import build_raw_diffs, fit_tvp_beta
from _holdout_common import residualized_target_series
from holdout_comparison import (
    compare_on_series, reindex_levels, MIN_TRAIN_OBS, HOLDOUT_YEARS,
    Q_PRIOR_SCALE, Q_PRIOR_STRENGTH,
)

DATA_PATH: Path = Path("data") / "inputs"
CONFIG_PATH: Path = Path(__file__).resolve().parent.parent.parent / "config.json"

MIN_OBS = 15         # matches _compute_beta_animals.py's full-sample fit floor
MIN_RAW_OBS = 2       # matches _holdout_common_animals.load_area_data's floor

OUT_DIR: Path = Path("outputs") / "validation" / "animals"
DETAIL_DIR: Path = OUT_DIR / "detail_by_product"
SUMMARY_PATH: Path = OUT_DIR / "beta_by_product_summary.csv"


def load_item_area_data():
    """Per-product analogue of _holdout_common_animals.load_area_data:
    returns {(item, item_code): (area_data, panel)}, one entry per grazing
    animal product, each built exactly like the pooled loader except the
    production side is kept split by Item instead of summed across all
    grazing commodities. The pasture-area side (a_t) is identical to the
    pooled version -- total grazing land per Area/Year -- since that's the
    only pasture-area series FAOSTAT provides."""
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    raw_animal_products = {int(k): v for k, v in config["raw_animal_products"].items()}
    land_use_codes = {int(k): v for k, v in config["land_use_codes"].items()}
    grazing_animal_products = {k: v for k, v in raw_animal_products.items() if v["grazing"]}
    grazing_land_types = {k: v for k, v in land_use_codes.items() if v["grazing"]}

    df = pd.read_csv(
        DATA_PATH / "Production_Crops_Livestock_E_All_Data_(Normalized).csv",
        encoding="latin-1", low_memory=False,
    )
    year_value_col = re.compile(r"^Y\d{4}$")
    land_use_df = pd.read_csv(
        DATA_PATH / "Inputs_LandUse_E_All_Data.csv",
        usecols=lambda c: c in ["Area", "Area Code", "Item", "Item Code", "Element", "Unit"] or year_value_col.match(c),
    )

    df = df[df["Item Code"].isin(grazing_animal_products.keys())]
    land_use_df = land_use_df[(land_use_df["Element"] == "Area") & (land_use_df["Item Code"].isin(grazing_land_types.keys()))]

    wide = df.pivot_table(
        index=["Area", "Area Code", "Item", "Item Code", "Year"], columns="Element", values="Value"
    ).reset_index()
    wide = wide.dropna(subset=["Production"])[["Area", "Area Code", "Item", "Item Code", "Year", "Production"]]

    pasture_year_cols = [c for c in land_use_df.columns
                          if c not in ["Area", "Area Code", "Item", "Item Code", "Element", "Unit"]]
    pasture_years = np.array([int(c[1:]) for c in pasture_year_cols])

    def area_pasture_for(area_code, years):
        area_pasture_wide = land_use_df.loc[land_use_df["Area Code"] == area_code, pasture_year_cols]
        pasture_values = area_pasture_wide.values
        all_nan = np.all(np.isnan(pasture_values), axis=0)
        pasture_sum = np.nansum(pasture_values, axis=0)
        pasture_sum[all_nan] = np.nan
        return pd.Series(pasture_sum, index=pasture_years).reindex(years).values

    items = {}
    for (item, item_code), g_item in wide.groupby(["Item", "Item Code"]):
        area_data = {}
        panel_rows = []
        for (area, area_code), g in g_item.groupby(["Area", "Area Code"]):
            g = g.sort_values("Year")
            if len(g) < MIN_RAW_OBS:
                continue
            years = g["Year"].values.astype(int)
            area_pasture = area_pasture_for(area_code, years)
            with np.errstate(divide="ignore"):
                log_pasture = np.log(area_pasture)
                log_P = np.log(g["Production"].values)
            diff_years, a_t, p_t = build_raw_diffs(years, log_pasture, log_P)
            area_data[area] = {"g": g, "diff_years": diff_years}
            panel_rows.append(pd.DataFrame({"entity": area, "time": diff_years, "a": a_t, "p": p_t}))

        panel = pd.concat(panel_rows, ignore_index=True) if panel_rows else pd.DataFrame(columns=["entity", "time", "a", "p"])
        items[(item, item_code)] = (area_data, panel)
    return items


def score_item(area_data, panel):
    """Full-sample TVP fit + full-sample static-OLS significance + leak-safe
    holdout comparison, per Area, for one product's (area_data, panel)."""
    full_betas = []
    static_rows = []
    holdout_rows = []
    n_qualified = 0
    n_fit = 0

    for area, entry in area_data.items():
        if len(entry["g"]) < MIN_OBS + 1:
            continue
        n_qualified += 1
        T = len(entry["diff_years"])

        a_full, p_full = residualized_target_series(area, entry, panel, T)

        fit = fit_tvp_beta(a_full, p_full, min_obs=MIN_OBS, q_prior_scale=Q_PRIOR_SCALE, q_prior_strength=Q_PRIOR_STRENGTH)
        if fit is not None:
            n_fit += 1
            full_betas.append(fit["beta"])

        valid = np.isfinite(a_full) & np.isfinite(p_full)
        if valid.sum() >= MIN_OBS:
            beta_s, intercept, r, p_value, se = linregress(p_full[valid], a_full[valid])
            static_rows.append({"Area": area, "beta_static": beta_s, "p_value": p_value, "n_obs": int(valid.sum())})

        g, diff_years = entry["g"], entry["diff_years"]
        if len(g) >= MIN_TRAIN_OBS + HOLDOUT_YEARS + 1:
            train_end = T - HOLDOUT_YEARS
            if train_end >= MIN_TRAIN_OBS:
                a_h, p_h = residualized_target_series(area, entry, panel, train_end)
                prod_level = reindex_levels(g["Year"].values.astype(int), g["Production"].values)
                out = compare_on_series(a_h, p_h, prod_level, train_end, Q_PRIOR_SCALE, Q_PRIOR_STRENGTH)
                if out is not None:
                    out["Area"] = area
                    holdout_rows.append(out)

    flat_betas = np.concatenate(full_betas) if full_betas else np.array([])
    flat_betas = flat_betas[np.isfinite(flat_betas)]
    static_df = pd.DataFrame(static_rows)
    holdout_df = pd.DataFrame(holdout_rows)

    summary = {
        "n_areas_qualified": n_qualified,
        "n_areas_fit_tvp": n_fit,
        "n_beta_obs": len(flat_betas),
        "median_beta_tvp": float(np.median(flat_betas)) if len(flat_betas) else np.nan,
        "mean_beta_tvp": float(np.mean(flat_betas)) if len(flat_betas) else np.nan,
        "n_areas_static": len(static_df),
        "median_beta_static": float(static_df["beta_static"].median()) if len(static_df) else np.nan,
        "mean_abs_beta_static": float(static_df["beta_static"].abs().mean()) if len(static_df) else np.nan,
        "frac_significant_p05": float((static_df["p_value"] < 0.05).mean()) if len(static_df) else np.nan,
        "n_holdout_series": len(holdout_df),
        "holdout_win_rate": float(holdout_df["tvp_beats_static"].mean()) if len(holdout_df) else np.nan,
        "mean_mae_static": float(holdout_df["mae_static"].mean()) if len(holdout_df) else np.nan,
        "mean_mae_tvp": float(holdout_df["mae_tvp"].mean()) if len(holdout_df) else np.nan,
    }
    return summary, static_df, holdout_df


if __name__ == "__main__":
    items = load_item_area_data()
    print(f"Loaded {len(items)} grazing animal products.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for (item, item_code), (area_data, panel) in items.items():
        print(f"\n=== {item} (Item Code {item_code}) ===")
        summary, static_df, holdout_df = score_item(area_data, panel)
        summary = {"item": item, "item_code": item_code, **summary}
        rows.append(summary)

        print(f"  full-sample: {summary['n_areas_fit_tvp']}/{summary['n_areas_qualified']} areas fit, "
              f"median beta_tvp={summary['median_beta_tvp']:.5f}")
        print(f"  static OLS:  {summary['n_areas_static']} areas, "
              f"median beta_static={summary['median_beta_static']:.5f}, "
              f"frac significant (p<0.05)={summary['frac_significant_p05']:.1%}"
              if summary["n_areas_static"] else "  static OLS:  no qualifying areas")
        if summary["n_holdout_series"]:
            print(f"  holdout:     {summary['n_holdout_series']} series, "
                  f"win rate={summary['holdout_win_rate']:.1%}, "
                  f"mae_static={summary['mean_mae_static']:.4f}, mae_tvp={summary['mean_mae_tvp']:.4f}")

        safe_code = str(item_code)
        static_df.to_csv(DETAIL_DIR / f"static_{safe_code}.csv", index=False)
        holdout_df.to_csv(DETAIL_DIR / f"holdout_{safe_code}.csv", index=False)

    summary_df = pd.DataFrame(rows).sort_values("frac_significant_p05")
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\n=== Per-product summary (sorted by fraction of areas with a significant slope, ascending) ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print(f"\nWritten summary to {SUMMARY_PATH}")
    print(f"Written per-product detail CSVs to {DETAIL_DIR}")
