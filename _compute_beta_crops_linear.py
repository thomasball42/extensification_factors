"""
Static (non-time-varying) OLS counterpart to compute_beta_crops.py.

Fits a single full-sample beta per (Area, Item) series:

    Delta ln(area)_t = beta * Delta ln(Production)_t + eps_t

by OLS, with the area set by config.json's area_method (FAOSTAT Area
harvested, or for "physical_area" Area harvested / national cropping
intensity -- see _utils.add_physical_area), using the same
reindexed/demeaned annual differences (build_annual_diffs) that the TVP
pipeline uses, so this output is a like-for-like comparison baseline for
outputs/<area_method>/time_varying/beta_crops.csv -- validation/holdout.py
compares the two out of sample.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress
from _stats import build_annual_diffs
from _utils import AREA_COL, AREA_METHOD, LINEAR_OUTPUTS_PATH, add_physical_area, load_aggregate_matcher

DATA_PATH: Path = Path("data") / "inputs"
OUT_PATH: Path = LINEAR_OUTPUTS_PATH / "beta_crops_linear.csv"

MIN_OBS: int = 15  # matches compute_beta_crops.py so both outputs cover the same series

is_aggregate = load_aggregate_matcher()

elements = ["Area harvested", "Production", "Yield"]
series_cols = [AREA_COL, "Production", "Yield"]  # a year is used only if all of these are present
columns = ["Area", "Area Code", "Item", "Item Code", "Element", "Year", "Value", "Unit"]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

df = pd.read_csv(
    DATA_PATH / "Production_Crops_Livestock_E_All_Data_(Normalized).csv",
    encoding="latin-1",
    low_memory=False,
    usecols=columns,
)

ha_unit = df.loc[df.Element == "Area harvested", "Unit"].values[0]
prod_unit = df.loc[df.Element == "Production", "Unit"].values[0]
yield_unit = df.loc[df.Element == "Yield", "Unit"].values[0]

df = df.drop(columns=["Unit"])
df = df[df.Element.isin(elements)]

# restrict to items that actually report "Area harvested" (i.e. crops, not
# livestock/animal-product items that reuse the "Production" element label)
crop_items = df.loc[df.Element == "Area harvested", "Item"].unique()
df = df[df.Item.isin(crop_items)]

wide = df.pivot_table(
    index=["Area", "Area Code", "Item", "Item Code", "Year"], columns="Element", values="Value"
).reset_index()

if AREA_METHOD == "physical_area":
    wide = add_physical_area(wide)

records = []
groups = wide.groupby(["Area", "Area Code", "Item", "Item Code"])
for i, ((area, area_code, item, item_code), g) in enumerate(groups):

    print(i + 1, "/", len(groups), area, item, item_code)

    g = g.sort_values("Year").dropna(subset=series_cols)
    if len(g) < MIN_OBS + 1:
        continue

    with np.errstate(divide="ignore"):
        log_A = np.log(g[AREA_COL].values)
        log_P = np.log(g["Production"].values)

    years = g["Year"].values.astype(int)

    _, a_t, p_t = build_annual_diffs(years, log_A, log_P)

    valid = np.isfinite(a_t) & np.isfinite(p_t)
    if valid.sum() < MIN_OBS:
        continue

    try:
        beta, intercept, r_value, p_value, std_err = linregress(p_t[valid], a_t[valid])
    except Exception as e:
        print(f"Error for {area} - {item}: {e}")
        continue

    record = {
        "Area": area,
        "Area Code": area_code,
        "Item": item,
        "Item Code": item_code,
        "is_aggregate": is_aggregate(item),
        "current_year": int(years[-1]),
        f"current_area_harvested_{ha_unit}": g["Area harvested"].values[-1],
        f"current_production_{prod_unit}": g["Production"].values[-1],
        f"current_yield_{yield_unit}": g["Yield"].values[-1],
        "beta": beta,
        "beta_se": std_err,
        "n_obs": int(valid.sum()),
    }
    if AREA_METHOD == "physical_area":
        record[f"current_physical_area_{ha_unit}"] = g["Physical area"].values[-1]
    records.append(record)

out = pd.DataFrame.from_records(records)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(out)} rows to {OUT_PATH}")
