"""
Animals counterpart to compute_beta_pred_vs_actual_land_use_crops.py -- see
that module's docstring for the full methodology (same formula, same use of
every valid historical year-over-year transition, same 3-scenario
classification by the size of the actual production change). Reads
outputs/beta_animals.csv for the beta_<year> history instead of
outputs/beta_crops.csv.

Unlike crops, there is no per-Item dimension here (a single pooled "All
pasture-based animal products" series per Area/country -- see
beta_scripts/_compute_beta_animals.py), so series = Area alone, and there's
no per-item ifilt/iexcl filtering.

The raw historical panel is rebuilt directly from the two source FAOSTAT
files rather than from outputs/beta_animals.csv's single current-year
snapshot, replicating _compute_beta_animals.py's own pooling exactly:
  - Production: data/inputs/Production_Crops_Livestock_E_All_Data_(Normalized).csv,
    summed per Area/Year across the Item Codes flagged "grazing": true in
    config.json's "raw_animal_products" section.
  - Pasture area: data/inputs/Inputs_LandUse_E_All_Data.csv (wide Y<year>
    format, reported in "1000 ha" -- converted to ha here), summed per
    Area/Year across the Item Codes flagged "grazing": true in config.json's
    "land_use_codes" section.

Plotting counterpart: ../plotting/land_use/plot_beta_pred_vs_actual_land_use_animals.py,
which reads this script's two CSV outputs and produces the figures -- run
this script first.

Produces:
  outputs/land_use_impact/land_use_pred_vs_actual_transitions_animals.csv
      -- one row per valid Area year-over-year transition, with the
         predicted/actual pasture-area change, error, and scenario
         assignment.
  outputs/land_use_impact/land_use_pred_vs_actual_scenarios_animals.csv
      -- one row per Area, with per-scenario columns answering "how does
         beta perform under different production-change scenarios" for
         that area.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _utils import load_grazing_item_codes, load_grazing_land_use_codes, load_plot_config

SCRIPT_NAME = "compute_beta_pred_vs_actual_land_use_animals"

BETA_PATH = Path("outputs") / "beta_animals.csv"
DATA_PATH = Path("data") / "inputs"
PRODUCTION_PATH = DATA_PATH / "Production_Crops_Livestock_E_All_Data_(Normalized).csv"
LAND_USE_PATH = DATA_PATH / "Inputs_LandUse_E_All_Data.csv"
TRANSITIONS_OUT_PATH = Path("outputs") / "land_use_impact" / "land_use_pred_vs_actual_transitions_animals.csv"
CSV_OUT_PATH = Path("outputs") / "land_use_impact" / "land_use_pred_vs_actual_scenarios_animals.csv"
SAVE = True

_lui_cfg = load_plot_config()["land_use_impact"][SCRIPT_NAME]
SCENARIO_FIXED_F = _lui_cfg["scenario_fixed_f"]                # [0.05, 0.20], editable
MIN_TRANSITIONS_PER_SERIES = _lui_cfg.get("min_transitions_per_series", 3)

ID_COLS = ["Area", "Area Code"]
YEAR_COL_RE = re.compile(r"^Y\d{4}$")  # excludes flag (YxxxxF) and note (YxxxxN) columns
LAND_USE_HA_MULTIPLIER = 1000.0  # Inputs_LandUse_E_All_Data.csv reports "1000 ha"

SCENARIO_KEYS = ["a", "b", "c"]
SCENARIO_LABELS = {
    "a": "Historical median |% production change|",
    "b": f"{SCENARIO_FIXED_F[0]:.0%} production change",
    "c": f"{SCENARIO_FIXED_F[1]:.0%} production change",
}

# ---------------------------------------------------------------------------
# Load beta history (long format: one row per Area per year with a beta)
# ---------------------------------------------------------------------------

beta_df = pd.read_csv(BETA_PATH)

beta_year_cols = sorted((c for c in beta_df.columns if c.startswith("beta_")), key=lambda c: int(c.split("_")[1]))
beta_long = beta_df[ID_COLS + beta_year_cols].melt(
    id_vars=ID_COLS, value_vars=beta_year_cols, var_name="beta_col", value_name="beta_t")
beta_long["Year"] = beta_long["beta_col"].str.split("_").str[1].astype(int)
beta_long = beta_long.drop(columns="beta_col").dropna(subset=["beta_t"])

print(f"{len(beta_long)} (area, year) beta values across {beta_df[ID_COLS].drop_duplicates().shape[0]} "
      f"areas in {BETA_PATH.name}.")

keys = beta_long[ID_COLS].drop_duplicates()

# ---------------------------------------------------------------------------
# Pooled production per Area/Year (grazing-flagged commodities only)
# ---------------------------------------------------------------------------

grazing_item_codes = load_grazing_item_codes()

prod_raw = pd.read_csv(PRODUCTION_PATH, encoding="latin-1", low_memory=False,
                        usecols=["Area", "Area Code", "Item Code", "Element", "Year", "Value"])
prod_raw = prod_raw[(prod_raw["Element"] == "Production") & (prod_raw["Item Code"].isin(grazing_item_codes))]
prod_raw = prod_raw.merge(keys, on=ID_COLS, how="inner")

pooled_prod = (
    prod_raw.groupby(ID_COLS + ["Year"])["Value"].sum(min_count=1)
    .reset_index().rename(columns={"Value": "Production"})
)

# ---------------------------------------------------------------------------
# Pooled pasture area per Area/Year (grazing-flagged land-use codes only)
# ---------------------------------------------------------------------------

grazing_land_use_codes = load_grazing_land_use_codes()

land_use_df = pd.read_csv(
    LAND_USE_PATH,
    usecols=lambda c: c in ["Area", "Area Code", "Item Code", "Element"] or YEAR_COL_RE.match(c),
)
land_use_df = land_use_df[
    (land_use_df["Element"] == "Area") & (land_use_df["Item Code"].isin(grazing_land_use_codes))
]
land_use_df = land_use_df.merge(keys, on=ID_COLS, how="inner")

year_cols = [c for c in land_use_df.columns if YEAR_COL_RE.match(c)]
land_use_long = land_use_df.melt(id_vars=ID_COLS + ["Item Code"], value_vars=year_cols,
                                  var_name="YearCol", value_name="Value")
land_use_long["Year"] = land_use_long["YearCol"].str[1:].astype(int)

pooled_area = (
    land_use_long.groupby(ID_COLS + ["Year"])["Value"].sum(min_count=1)
    .reset_index().rename(columns={"Value": "Pasture area"})
)
pooled_area["Pasture area"] *= LAND_USE_HA_MULTIPLIER

# ---------------------------------------------------------------------------
# Build the per-Area historical panel and its year-over-year transitions
# ---------------------------------------------------------------------------

panel = pooled_prod.merge(pooled_area, on=ID_COLS + ["Year"], how="inner")
panel = panel.sort_values(ID_COLS + ["Year"])

grp = panel.groupby(ID_COLS, sort=False)
panel["Year_prev"] = grp["Year"].shift(1)
panel["Production_prev"] = grp["Production"].shift(1)
panel["Pasture area_prev"] = grp["Pasture area"].shift(1)

valid = (
    ((panel["Year"] - panel["Year_prev"]) == 1)
    & panel["Production"].gt(0) & panel["Production_prev"].gt(0)
    & panel["Pasture area"].gt(0) & panel["Pasture area_prev"].gt(0)
)
trans = panel[valid].copy()
trans = trans.merge(beta_long, on=ID_COLS + ["Year"], how="inner")

n_series_pre = trans[ID_COLS].drop_duplicates().shape[0]
print(f"{len(trans)} valid year-over-year transitions (positive pooled Production/pasture area in "
      f"both years, a contemporaneous beta_t) across {n_series_pre} areas.")

series_counts = trans.groupby(ID_COLS).size().rename("n_transitions_series")
trans = trans.merge(series_counts.reset_index(), on=ID_COLS, how="left")
trans = trans[trans["n_transitions_series"] >= MIN_TRANSITIONS_PER_SERIES].copy()

n_series = trans[ID_COLS].drop_duplicates().shape[0]
print(f"{len(trans)} transitions across {n_series} areas with >= {MIN_TRANSITIONS_PER_SERIES} "
      f"valid transitions (of {n_series_pre} areas above).")

# ---------------------------------------------------------------------------
# Predicted vs. actual, per transition
# ---------------------------------------------------------------------------

P_t = trans["Production"].to_numpy(float)
P_prev = trans["Production_prev"].to_numpy(float)
A_t = trans["Pasture area"].to_numpy(float)
A_prev = trans["Pasture area_prev"].to_numpy(float)
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
print(f"\nR-squared (predicted vs. actual log-change in pasture area): {r2:.3f}")
print(f"Directional accuracy (unweighted): {dir_acc:.1%}")
print(f"Directional accuracy (production-weighted): {dir_acc_w:.1%}")

# ---------------------------------------------------------------------------
# Scenario classification: each transition's actual |% change| is assigned
# to whichever of the 3 targets (area's own historical median, or the 2
# fixed config values) it's closest to.
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
    "Year", "Production", "Production_prev", "Pasture area", "Pasture area_prev",
    "beta_t", "prod_change_pct", "abs_prod_change_pct", "predicted_A_t",
    "error_ha", "error_pct", "actual_a", "predicted_a", "correct_direction",
    "scenario", "scenario_label",
]

if SAVE:
    TRANSITIONS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    trans[TRANSITIONS_OUT_COLS].to_csv(TRANSITIONS_OUT_PATH, index=False)
    print(f"\nWrote {len(trans)} transitions to {TRANSITIONS_OUT_PATH}")

# ---------------------------------------------------------------------------
# CSV 2: one row per Area, per-scenario performance columns
# ---------------------------------------------------------------------------

csv_base = trans[ID_COLS].drop_duplicates().merge(
    f_hist_by_series.reset_index()[ID_COLS + ["f_hist"]], on=ID_COLS, how="left",
).rename(columns={"f_hist": "f_hist_abs_pct_change"})

for key in SCENARIO_KEYS:
    sub = trans[trans["scenario"] == key]
    summ = sub.groupby(ID_COLS).agg(
        **{
            f"n_transitions_scenario_{key}": ("error_pct", "size"),
            f"mean_error_pct_scenario_{key}": ("error_pct", "mean"),
            f"mean_abs_error_pct_scenario_{key}": ("error_pct", lambda s: s.abs().mean()),
            f"mean_error_ha_scenario_{key}": ("error_ha", "mean"),
        }
    ).reset_index()
    csv_base = csv_base.merge(summ, on=ID_COLS, how="left")

if SAVE:
    CSV_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    csv_base.to_csv(CSV_OUT_PATH, index=False)
    print(f"Wrote {len(csv_base)} areas to {CSV_OUT_PATH}")
