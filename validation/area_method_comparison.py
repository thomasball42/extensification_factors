"""
How the physical-area crop betas differ from the harvested-area ones (both
from _compute_beta_crops.py, run once per config.json area_method), for every
(country, crop) series in the physical-area output.

1. The published products: the physical-area beta in its current (last) year
   against the harvested-area beta in that same year, and in the harvested
   series' own current year.

2. Why they differ. Over the common window the physical area exists for
   (2002-2019 diffs), with a_h = Dln(Area harvested), a_phys = Dln(Physical
   area) and kappa = Dln(CI) = a_h - a_phys, the OLS slopes on
   p = Dln(Production) split exactly as

       beta_h = beta_phys + beta_ci

   so beta_ci is the part of a harvested-area beta that is cropping
   intensity rather than land. Comparing beta_h on this window with the
   full-sample harvested beta separates the effect of the shorter window from
   the effect of the area definition. OLS rather than TVP because only OLS
   slopes are additive -- validation/holdout.py shows the TVP and OLS betas
   barely differ over this window.

FAOSTAT aggregate items are excluded from the summaries (they double count
their component crops) but kept in the per-series table. Means are
production-weighted, by each series' current production.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _stats import build_annual_diffs, build_raw_diffs
from _utils import load_crop_wide

PHYSICAL_PATH: Path = Path("outputs") / "physical_area" / "time_varying" / "beta_crops.csv"
HARVESTED_PATH: Path = Path("outputs") / "harvested_area" / "time_varying" / "beta_crops.csv"
OUT_DIR: Path = Path("outputs") / "physical_area" / "validation"

MIN_OBS: int = 15  # matches _compute_beta_crops.py
TOP_N_ITEMS: int = 20  # by physical area, for the per-item table
KEYS = ["Area Code", "Item Code"]

BETA_COLS = {
    "beta_phys_tvp": "physical, TVP, current year",
    "beta_harv_tvp_same_year": "harvested, TVP, same year",
    "beta_harv_tvp_current": "harvested, TVP, its current year",
    "beta_h_full": "harvested, OLS, full sample",
    "beta_h_window": "harvested, OLS, common window",
    "beta_phys_window": "physical, OLS, common window",
    "beta_ci_window": "cropping intensity, OLS, common window",
}


def beta_at(out, years):
    """beta_<year> (and se_<year>) of each row of a beta_crops.csv frame, at
    that row's entry of `years`."""
    beta = pd.Series(np.nan, index=out.index)
    se = pd.Series(np.nan, index=out.index)
    for yr in pd.unique(years.dropna()):
        rows = years == yr
        if f"beta_{int(yr)}" in out.columns:
            beta[rows] = out.loc[rows, f"beta_{int(yr)}"]
            se[rows] = out.loc[rows, f"se_{int(yr)}"]
    return beta, se


def ols_slope(a, p):
    """Slope of a on p for already-demeaned diffs."""
    return (p @ a) / (p @ p)


def ols_betas(g):
    """Full-sample harvested-area OLS beta, and the beta_h = beta_phys +
    beta_ci split over the years with a physical area, for one series."""
    g = g.sort_values("Year").dropna(subset=["Area harvested", "Production", "Yield"])
    rec = {}
    if len(g) < MIN_OBS + 1:
        return rec
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ah = np.log(g["Area harvested"].values)
        log_phys = np.log(g["Physical area"].values)
        log_p = np.log(g["Production"].values)
    years = g["Year"].values.astype(int)

    _, a, p = build_annual_diffs(years, log_ah, log_p)
    valid = np.isfinite(a) & np.isfinite(p)
    if valid.sum() >= MIN_OBS and p[valid] @ p[valid] > 0:
        rec["beta_h_full"] = ols_slope(a[valid], p[valid])

    w = np.isfinite(log_phys)
    if w.sum() < MIN_OBS + 1:
        return rec
    years, log_ah, log_phys, log_p = years[w], log_ah[w], log_phys[w], log_p[w]
    diffs = {k: build_raw_diffs(years, v, log_p)[1:] for k, v in
             [("h", log_ah), ("phys", log_phys), ("ci", log_ah - log_phys)]}
    p = diffs["h"][1]
    valid = np.isfinite(p) & np.all([np.isfinite(d[0]) for d in diffs.values()], axis=0)
    pv = p[valid] - p[valid].mean()
    if valid.sum() < MIN_OBS or not pv @ pv > 0:
        return rec
    rec["n_window"] = int(valid.sum())
    for k, (a_k, _) in diffs.items():
        rec[f"beta_{k}_window"] = ols_slope(a_k[valid] - a_k[valid].mean(), pv)
    return rec


def weighted_stats(x, w):
    m = x.notna()
    return pd.Series({
        "n_series": int(m.sum()),
        "weighted_mean": np.average(x[m], weights=w[m]),
        "median": x[m].median(),
        "p25": x[m].quantile(0.25),
        "p75": x[m].quantile(0.75),
    })


if __name__ == "__main__":
    if not HARVESTED_PATH.exists():
        sys.exit(f"{HARVESTED_PATH} not found -- run 1_run_compute.py with area_method = \"harvested_area\" first")

    phys = pd.read_csv(PHYSICAL_PATH)
    harv = pd.read_csv(HARVESTED_PATH).set_index(KEYS)
    prod_col = next(c for c in phys.columns if c.startswith("current_production_"))
    phys_area_col = next(c for c in phys.columns if c.startswith("current_physical_area_"))

    out = phys[["Area", "Area Code", "Item", "Item Code", "is_aggregate", "current_year", prod_col, phys_area_col]].copy()
    out["beta_phys_tvp"], out["se_phys_tvp"] = beta_at(phys, phys["current_year"])

    harv = harv.reindex(pd.MultiIndex.from_frame(out[KEYS])).reset_index(drop=True)
    out["current_year_harv"] = harv["current_year"]
    out["beta_harv_tvp_same_year"], out["se_harv_tvp_same_year"] = beta_at(harv, out["current_year"])
    out["beta_harv_tvp_current"], out["se_harv_tvp_current"] = beta_at(harv, harv["current_year"])

    wide = load_crop_wide(with_physical=True)
    wanted = wide.set_index(KEYS).index.isin(pd.MultiIndex.from_frame(out[KEYS]))
    ols = pd.DataFrame.from_records(
        [{"Area Code": ac, "Item Code": ic, **ols_betas(g)} for (ac, ic), g in wide[wanted].groupby(KEYS)]
    )
    out = out.merge(ols, on=KEYS, how="left", validate="one_to_one")
    resid = (out["beta_h_window"] - out["beta_phys_window"] - out["beta_ci_window"]).abs().max()

    x = out[~out["is_aggregate"] & out[prod_col].gt(0)]
    w = x[prod_col]
    summary = pd.DataFrame({col: weighted_stats(x[col], w) for col in BETA_COLS}).T
    summary.insert(0, "description", pd.Series(BETA_COLS))

    pairs = [
        ("beta_phys_tvp", "beta_harv_tvp_same_year", "published products, same year"),
        ("beta_phys_tvp", "beta_harv_tvp_current", "published products, each in its current year"),
        ("beta_h_window", "beta_phys_window", "area definition only (same window, OLS)"),
        ("beta_h_full", "beta_h_window", "window only (harvested area, OLS)"),
    ]
    correlations = pd.DataFrame([
        {"x": a, "y": b, "comparison": label,
         "n_series": int(x[[a, b]].dropna().shape[0]),
         "correlation": x[a].corr(x[b]),
         "weighted_mean_diff_y_minus_x": np.average((x[b] - x[a]).dropna(), weights=w[(x[b] - x[a]).notna()]),
         "median_abs_diff": (x[b] - x[a]).abs().median()}
        for a, b, label in pairs
    ])

    by_item = x.groupby("Item").apply(
        lambda d: pd.Series({
            "n_series": len(d),
            "physical_area_ha": d[phys_area_col].sum(),
            **{col: np.average(d[col].dropna(), weights=d.loc[d[col].notna(), prod_col]) if d[col].notna().any() else np.nan
               for col in BETA_COLS},
        }),
        include_groups=False,
    ).sort_values("physical_area_ha", ascending=False).head(TOP_N_ITEMS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "area_method_comparison_series.csv", index=False)
    summary.to_csv(OUT_DIR / "area_method_comparison_summary.csv")
    correlations.to_csv(OUT_DIR / "area_method_comparison_correlations.csv", index=False)
    by_item.to_csv(OUT_DIR / "area_method_comparison_by_item.csv")

    print(f"{len(x)} non-aggregate series in the physical-area output "
          f"({x['beta_harv_tvp_same_year'].notna().sum()} matched in the harvested-area output); "
          f"max |beta_h - beta_phys - beta_ci| over the window = {resid:.1e}\n")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    print(correlations.drop(columns=["x", "y"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nTop {TOP_N_ITEMS} items by physical area (production-weighted betas):")
    print(by_item.drop(columns=["beta_harv_tvp_current"]).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nWritten to {OUT_DIR}")
