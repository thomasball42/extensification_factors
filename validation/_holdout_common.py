"""
Shared cross-country panel + leak-safe residualization plumbing for the
held-out validation scripts (hold_out_static_comparison.py,
hold_out_validation.py, and q_prior_shrinkage_validation.py via the former).

The main pipeline (_compute_beta_crops.py) fits its TVP model on log-diffs
residualized against two-way (country + year) fixed effects, not on raw
log-diffs -- see residualize_two_way_fe in _stats.py. A holdout test that
skips this step is validating a different data-generating process than the
one that actually produces beta_crops.csv. Reproducing it correctly requires
building the cross-country panel per Item (fixed effects are identified
jointly across every country growing that item) and excluding only the
target country's held-out rows from the fixed-effects fit itself -- see
residualized_target_series / _stats.residualize_two_way_fe_oos -- so the
target's own held-out values can't leak into its own entity effect. Other
countries' contemporaneous data is used as-is, mirroring what would
genuinely be observable in real time.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from _stats import build_raw_diffs, residualize_two_way_fe_oos, residualize_time_fe_oos

DATA_PATH: Path = Path("data") / "inputs"
elements = ["Area harvested", "Production", "Yield"]
columns = ["Area", "Item", "Item Code", "Element", "Year", "Value", "Unit"]


def load_wide():
    """Load FAOSTAT crop series (Area harvested / Production / Yield) into
    wide (one column per Element) form, restricted to items that report
    Area harvested."""
    df = pd.read_csv(
        DATA_PATH / "Production_Crops_Livestock_E_All_Data_(Normalized).csv",
        encoding="latin-1", low_memory=False, usecols=columns,
    )
    df = df[df.Element.isin(elements)]
    crop_items = df.loc[df.Element == "Area harvested", "Item"].unique()
    df = df[df.Item.isin(crop_items)]
    return df.pivot_table(index=["Area", "Item", "Item Code", "Year"], columns="Element", values="Value").reset_index()


def load_groups(wide, sample_n=None, random_seed=0):
    """Return the (Area, Item, Item Code) groups to SCORE, optionally
    subsampled. This does not restrict which countries participate in the
    cross-country panel used for fixed-effects residualization -- see
    build_item_area_data, which always uses every country growing an item
    regardless of whether that country's own series was sampled here."""
    groups = list(wide.groupby(["Area", "Item", "Item Code"]))
    if sample_n is not None and len(groups) > sample_n:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(groups), size=sample_n, replace=False)
        groups = [groups[i] for i in idx]
    return groups


def build_item_area_data(g_item):
    """g_item: rows of load_wide()'s output for one Item. Returns
    (area_data, panel): area_data maps Area -> {"g": cleaned per-country
    frame, "diff_years": calendar years of its diffs}; panel is the raw
    (non-demeaned) diff panel across EVERY area growing this item, needed so
    residualize_two_way_fe_oos can use every area's contemporaneous data to
    identify year effects, even for areas this run isn't scoring."""
    area_data = {}
    panel_rows = []
    for area, g in g_item.groupby("Area"):
        g = g.sort_values("Year").dropna(subset=elements)
        if len(g) < 2:
            continue
        with np.errstate(divide="ignore"):
            log_AH = np.log(g["Area harvested"].values)
            log_P = np.log(g["Production"].values)
        years = g["Year"].values.astype(int)
        diff_years, a_t, p_t = build_raw_diffs(years, log_AH, log_P)
        area_data[area] = {"g": g, "diff_years": diff_years}
        panel_rows.append(pd.DataFrame({"entity": area, "time": diff_years, "a": a_t, "p": p_t}))

    if not panel_rows:
        return area_data, pd.DataFrame(columns=["entity", "time", "a", "p"])
    return area_data, pd.concat(panel_rows, ignore_index=True)


def residualized_target_series(area, area_entry, panel, train_end):
    """Residualize `area`'s series against two-way (country + year) fixed
    effects fit on the cross-country panel, excluding `area`'s own held-out
    (test, i.e. diff_years[train_end:]) rows from the fit. Returns
    (a_resid, p_resid) aligned to area_entry["diff_years"]."""
    diff_years = area_entry["diff_years"]
    test_years = set(diff_years[train_end:].tolist())
    fit_mask = ~((panel["entity"] == area) & (panel["time"].isin(test_years))).to_numpy()

    resid_panel = residualize_two_way_fe_oos(panel, fit_mask)
    sub = resid_panel[resid_panel["entity"] == area].sort_values("time")
    return sub["a_resid"].to_numpy(dtype=float), sub["p_resid"].to_numpy(dtype=float)


def time_fe_target_series(area, area_entry, panel, train_end):
    """Leak-safe counterfactual to residualized_target_series with a
    YEAR-ONLY fixed effect -- no country (entity) effect. Excludes `area`'s
    own held-out rows from the year-effect fit, same as
    residualized_target_series. Used to test whether the entity effect in
    two-way FE residualization is pulling its weight, or whether any benefit
    of residualizing comes entirely from absorbing year-level shocks."""
    diff_years = area_entry["diff_years"]
    test_years = set(diff_years[train_end:].tolist())
    fit_mask = ~((panel["entity"] == area) & (panel["time"].isin(test_years))).to_numpy()

    resid_panel = residualize_time_fe_oos(panel, fit_mask)
    sub = resid_panel[resid_panel["entity"] == area].sort_values("time")
    return sub["a_resid"].to_numpy(dtype=float), sub["p_resid"].to_numpy(dtype=float)


def demeaned_target_series(area, area_entry, panel, train_end):
    """Leak-safe counterfactual to residualized_target_series: instead of
    two-way (country+year) fixed-effects residualization, apply the main
    pipeline's non-residualization fallback (build_annual_diffs-style plain
    per-series demeaning), but computed from TRAIN-ONLY statistics so no
    test-period value leaks into the demeaning constant. Returns
    (a_demeaned, p_demeaned) aligned to area_entry["diff_years"], directly
    comparable to residualized_target_series's output for the same series
    and train_end -- used to test whether the fixed-effects step actually
    earns its keep over plain demeaning."""
    sub = panel[panel["entity"] == area].sort_values("time")
    a = sub["a"].to_numpy(dtype=float)
    p = sub["p"].to_numpy(dtype=float)
    train_valid = np.isfinite(a[:train_end]) & np.isfinite(p[:train_end])
    if train_valid.sum() == 0:
        return np.full_like(a, np.nan), np.full_like(p, np.nan)
    a_mean = np.nanmean(a[:train_end][train_valid])
    p_mean = np.nanmean(p[:train_end][train_valid])
    return a - a_mean, p - p_mean
