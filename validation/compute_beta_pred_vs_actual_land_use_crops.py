"""
Validates beta against real history: for every valid pair of consecutive
years in a (Area, Item) series, does the beta-implied area change from the
ACTUALLY observed production change match the area change that ACTUALLY
happened?

    A_t_predicted = A_{t-1} * (P_t / P_{t-1}) ** beta_t

using that series' own contemporaneous beta_t (its beta_<t> column in
outputs/beta_crops.csv -- the beta the TVP model had at year t, not today's
current beta applied retroactively), compared against the actually observed
A_t.

Every series contributes every valid transition it has (not just its most
recent year), because the point is to see how beta's real-world accuracy
holds up across different SIZES of actual production change. Each
transition's actual |% change in production| is classified into whichever of
3 scenarios it's closest to:
  Scenario A -- "typical": that series' own historical MEDIAN |% change|
                (median, not mean -- a series with even one near-zero-to-
                large production jump, e.g. a crop just starting to be
                reported, can otherwise blow the mean up to absurd values)
  Scenario B -- a fixed target (scenario_fixed_f[0] in plot_config.json, 5%)
  Scenario C -- a fixed target (scenario_fixed_f[1] in plot_config.json, 20%)
so routine years mostly land in Scenario A while large-shock years land in
B/C, with no arbitrary tolerance window needed.

Brazil analog: ../plotting/brazil/plot_beta_pred_vs_actual_land_use_brazil.py
(which only checks each region's single most recent year, since Brazil's
pasture panel is shorter; this crops version generalizes to every valid
year).

Plotting counterpart: ../plotting/land_use/plot_beta_pred_vs_actual_land_use_crops.py,
which reads this script's two CSV outputs and produces the figures -- run
this script first.

Produces:
  outputs/land_use_impact/land_use_pred_vs_actual_transitions_crops.csv
      -- one row per valid (Area, Item) year-over-year transition, with the
         predicted/actual area change, error, and scenario assignment.
  outputs/land_use_impact/land_use_pred_vs_actual_scenarios_crops.csv
      -- one row per (Area, Item) series, with per-scenario columns
         answering "how does beta perform under different production-change
         scenarios" for that series.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _utils import filter_list as _filter_list, load_aggregate_matcher, load_plot_config

SCRIPT_NAME = "compute_beta_pred_vs_actual_land_use_crops"

BETA_PATH = Path("outputs") / "beta_crops.csv"
RAW_DATA_PATH = Path("data") / "inputs" / "Production_Crops_Livestock_E_All_Data_(Normalized).csv"
TRANSITIONS_OUT_PATH = Path("outputs") / "land_use_impact" / "land_use_pred_vs_actual_transitions_crops.csv"
CSV_OUT_PATH = Path("outputs") / "land_use_impact" / "land_use_pred_vs_actual_scenarios_crops.csv"
SAVE = True

_lui_cfg = load_plot_config()["land_use_impact"][SCRIPT_NAME]
SCENARIO_FIXED_F = _lui_cfg["scenario_fixed_f"]                # [0.05, 0.20], editable
MIN_TRANSITIONS_PER_SERIES = _lui_cfg.get("min_transitions_per_series", 3)

ifilt = _lui_cfg.get("ifilt", [])  # empty = no item filtering (every crop with enough history)
iexcl = _lui_cfg.get("iexcl", [])

ID_COLS = ["Area", "Area Code", "Item", "Item Code"]

SCENARIO_KEYS = ["a", "b", "c"]
SCENARIO_LABELS = {
    "a": "Historical median |% production change|",
    "b": f"{SCENARIO_FIXED_F[0]:.0%} production change",
    "c": f"{SCENARIO_FIXED_F[1]:.0%} production change",
}

is_aggregate = load_aggregate_matcher()

# ---------------------------------------------------------------------------
# Load beta history (long format: one row per series per year with a beta)
# ---------------------------------------------------------------------------

beta_df = pd.read_csv(BETA_PATH)
beta_df = beta_df[~beta_df["is_aggregate"]].copy()

if ifilt:
    items = _filter_list(beta_df["Item"].unique(), ifilt, iexcl)
    beta_df = beta_df[beta_df["Item"].isin(items)].copy()

area_col = next(c for c in beta_df.columns if c.startswith("current_area_harvested_"))
prod_col = next(c for c in beta_df.columns if c.startswith("current_production_"))

beta_year_cols = sorted((c for c in beta_df.columns if c.startswith("beta_")), key=lambda c: int(c.split("_")[1]))
beta_long = beta_df[ID_COLS + beta_year_cols].melt(
    id_vars=ID_COLS, value_vars=beta_year_cols, var_name="beta_col", value_name="beta_t")
beta_long["Year"] = beta_long["beta_col"].str.split("_").str[1].astype(int)
beta_long = beta_long.drop(columns="beta_col").dropna(subset=["beta_t"])

print(f"{len(beta_long)} (series, year) beta values across {beta_df[['Area','Item','Item Code']].drop_duplicates().shape[0]} "
      f"non-aggregate series in {BETA_PATH.name}.")

# ---------------------------------------------------------------------------
# Load raw Production + Area harvested history, restricted to these series
# ---------------------------------------------------------------------------

raw = pd.read_csv(RAW_DATA_PATH, encoding="latin-1", low_memory=False,
                   usecols=["Area", "Area Code", "Item", "Item Code", "Element", "Year", "Value"])
raw = raw[raw["Element"].isin(["Area harvested", "Production"])]

keys = beta_long[ID_COLS].drop_duplicates()
raw = raw.merge(keys, on=ID_COLS, how="inner")

wide = raw.pivot_table(index=ID_COLS + ["Year"], columns="Element", values="Value").reset_index()
wide = wide.sort_values(ID_COLS + ["Year"])

grp = wide.groupby(ID_COLS, sort=False)
wide["Year_prev"] = grp["Year"].shift(1)
wide["Production_prev"] = grp["Production"].shift(1)
wide["Area harvested_prev"] = grp["Area harvested"].shift(1)

valid = (
    ((wide["Year"] - wide["Year_prev"]) == 1)
    & wide["Production"].gt(0) & wide["Production_prev"].gt(0)
    & wide["Area harvested"].gt(0) & wide["Area harvested_prev"].gt(0)
)
trans = wide[valid].copy()
trans = trans.merge(beta_long, on=ID_COLS + ["Year"], how="inner")

n_series_pre = trans[ID_COLS].drop_duplicates().shape[0]
print(f"{len(trans)} valid year-over-year transitions (positive Production/Area harvested in "
      f"both years, a contemporaneous beta_t) across {n_series_pre} series.")

series_counts = trans.groupby(ID_COLS).size().rename("n_transitions_series")
trans = trans.merge(series_counts.reset_index(), on=ID_COLS, how="left")
trans = trans[trans["n_transitions_series"] >= MIN_TRANSITIONS_PER_SERIES].copy()

n_series = trans[ID_COLS].drop_duplicates().shape[0]
print(f"{len(trans)} transitions across {n_series} series with >= {MIN_TRANSITIONS_PER_SERIES} "
      f"valid transitions (of {n_series_pre} series above).")

# ---------------------------------------------------------------------------
# Predicted vs. actual, per transition
# ---------------------------------------------------------------------------

P_t = trans["Production"].to_numpy(float)
P_prev = trans["Production_prev"].to_numpy(float)
A_t = trans["Area harvested"].to_numpy(float)
A_prev = trans["Area harvested_prev"].to_numpy(float)
beta_t = trans["beta_t"].to_numpy(float)

prod_ratio = P_t / P_prev
predicted_A_t = A_prev * prod_ratio ** beta_t

trans["prod_change_pct"] = prod_ratio - 1.0
trans["abs_prod_change_pct"] = np.abs(trans["prod_change_pct"])
trans["predicted_A_t"] = predicted_A_t
trans["error_ha"] = A_t - predicted_A_t
trans["error_pct"] = 100.0 * trans["error_ha"] / A_t
trans["actual_a"] = np.log(A_t / A_prev)
trans["predicted_a"] = beta_t * np.log(prod_ratio)
trans["correct_direction"] = np.sign(A_t - A_prev) == np.sign(predicted_A_t - A_prev)

weight = P_t
r2 = np.corrcoef(trans["actual_a"], trans["predicted_a"])[0, 1] ** 2
dir_acc = trans["correct_direction"].mean()
dir_acc_w = np.average(trans["correct_direction"], weights=weight)
print(f"\nR-squared (predicted vs. actual log-change in area): {r2:.3f}")
print(f"Directional accuracy (unweighted): {dir_acc:.1%}")
print(f"Directional accuracy (production-weighted): {dir_acc_w:.1%}")

# ---------------------------------------------------------------------------
# Scenario classification: each transition's actual |% change| is assigned
# to whichever of the 3 targets (series' own historical mean, or the 2 fixed
# config values) it's closest to.
# ---------------------------------------------------------------------------

f_hist_by_series = trans.groupby(ID_COLS)["abs_prod_change_pct"].median().rename("f_hist")
trans = trans.merge(f_hist_by_series.reset_index(), on=ID_COLS, how="left")

targets = np.stack([
    trans["f_hist"].to_numpy(float),
    np.full(len(trans), SCENARIO_FIXED_F[0]),
    np.full(len(trans), SCENARIO_FIXED_F[1]),
], axis=1)
dist = np.abs(targets - trans["abs_prod_change_pct"].to_numpy(float)[:, None])
trans["scenario"] = np.array(SCENARIO_KEYS)[np.argmin(dist, axis=1)]
trans["scenario_label"] = trans["scenario"].map(SCENARIO_LABELS)

for key in SCENARIO_KEYS:
    sub = trans[trans["scenario"] == key]
    w = sub["Production"].to_numpy(float)
    print(f"\n{SCENARIO_LABELS[key]}: {len(sub)} transitions")
    print(f"  Actual |% production change| in these transitions: mean {sub['abs_prod_change_pct'].mean():.1%}")
    print(f"  Land-use prediction error (% of actual area) -- unweighted mean/median: "
          f"{sub['error_pct'].mean():.1f}% / {sub['error_pct'].median():.1f}%")
    print(f"  Land-use prediction error (% of actual area) -- production-weighted mean: "
          f"{np.average(sub['error_pct'], weights=w):.1f}%")

# ---------------------------------------------------------------------------
# CSV 1: one row per valid transition, for the plotting script to re-derive
# the scatter and error-distribution figures from.
# ---------------------------------------------------------------------------

TRANSITIONS_OUT_COLS = ID_COLS + [
    "Year", "Production", "Production_prev", "Area harvested", "Area harvested_prev",
    "beta_t", "prod_change_pct", "abs_prod_change_pct", "predicted_A_t",
    "error_ha", "error_pct", "actual_a", "predicted_a", "correct_direction",
    "scenario", "scenario_label",
]

if SAVE:
    TRANSITIONS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    trans[TRANSITIONS_OUT_COLS].to_csv(TRANSITIONS_OUT_PATH, index=False)
    print(f"\nWrote {len(trans)} transitions to {TRANSITIONS_OUT_PATH}")

# ---------------------------------------------------------------------------
# CSV 2: one row per (Area, Item) series, per-scenario performance columns
# ---------------------------------------------------------------------------

csv_base = trans[["Area", "Item", "Item Code"]].drop_duplicates().merge(
    f_hist_by_series.reset_index()[["Area", "Item", "Item Code", "f_hist"]],
    on=["Area", "Item", "Item Code"], how="left",
).rename(columns={"f_hist": "f_hist_abs_pct_change"})

csv_base = csv_base.merge(
    beta_df[["Area", "Item", "Item Code", prod_col, area_col]]
        .drop_duplicates()
        .rename(columns={prod_col: "current_production", area_col: "current_area_harvested"}),
    on=["Area", "Item", "Item Code"], how="left",
)

for key in SCENARIO_KEYS:
    sub = trans[trans["scenario"] == key]
    summ = sub.groupby(["Area", "Item", "Item Code"]).agg(
        **{
            f"n_transitions_scenario_{key}": ("error_pct", "size"),
            f"mean_error_pct_scenario_{key}": ("error_pct", "mean"),
            f"mean_abs_error_pct_scenario_{key}": ("error_pct", lambda s: s.abs().mean()),
            f"mean_error_ha_scenario_{key}": ("error_ha", "mean"),
        }
    ).reset_index()
    csv_base = csv_base.merge(summ, on=["Area", "Item", "Item Code"], how="left")

if SAVE:
    CSV_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    csv_base.to_csv(CSV_OUT_PATH, index=False)
    print(f"Wrote {len(csv_base)} series to {CSV_OUT_PATH}")
