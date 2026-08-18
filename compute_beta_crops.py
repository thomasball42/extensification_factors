"""
Time-varying-parameter (Kalman filter/smoother) version of the extensification-
factor pipeline.

Replaces fixed-width rolling-window OLS with a single state-space model per
(Area, Item) series:

    a_t = beta_t * p_t + u_t          u_t ~ N(0, R)      [observation eq.]
    beta_t = beta_{t-1} + eta_t       eta_t ~ N(0, Q)     [state eq., random walk]

where a_t = Delta ln(Area harvested), p_t = Delta ln(Production). Q and R are
estimated per series by maximum likelihood. Output is the *smoothed* beta_t
(uses the whole series, not just past data) plus its standard error, for
every year.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from _functions import kalman_filter, rts_smoother, fit_tvp_beta, build_annual_diffs, load_aggregate_matcher

DATA_PATH: Path = Path("data") / "inputs" # this needs the full production dataset from FAO
OUT_PATH: Path = Path("data", "outputs", "beta_crops.csv")

MIN_OBS: int = 15  # minimum number of valid (non-missing) yearly diff pairs required to fit

is_aggregate = load_aggregate_matcher()

elements = ["Area harvested", "Production", "Yield"]
columns = ["Area", "Area Code", "Item", "Item Code", "Element", "Year", "Value", "Unit"]


# ---------------------------------------------------------------------------
# Main
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

records = []
groups = wide.groupby(["Area", "Area Code", "Item", "Item Code"])
for i, ((area, area_code, item, item_code), g) in enumerate(groups): 

    print(i + 1, "/", len(groups), area, item, item_code)

    g = g.sort_values("Year").dropna(subset=elements)
    if len(g) < MIN_OBS + 1:
        continue

    with np.errstate(divide="ignore"):
        log_AH = np.log(g["Area harvested"].values)
        log_P = np.log(g["Production"].values)

    years = g["Year"].values.astype(int)

    diff_years, a_t, p_t = build_annual_diffs(years, log_AH, log_P)

    fit = fit_tvp_beta(a_t, p_t, min_obs=MIN_OBS)
    if fit is None:
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
        "Q_hat": fit["Q"],
        "R_hat": fit["R"],
    }

    for yr, beta, se in zip(diff_years, fit["beta"], fit["se"]):
        record[f"beta_{int(yr)}"] = beta
        record[f"se_{int(yr)}"] = se

    records.append(record)

out = pd.DataFrame.from_records(records)

id_cols = [c for c in out.columns if not (c.startswith("beta_") or c.startswith("se_"))]
beta_cols = sorted((c for c in out.columns if c.startswith("beta_")), key=lambda c: int(c.split("_")[1]))

year_ordered_cols = []
for bc in beta_cols:
    yr = bc.split("_", 1)[1]
    year_ordered_cols.append(bc)
    year_ordered_cols.append(f"se_{yr}")

out = out[id_cols + year_ordered_cols]

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(out)} rows to {OUT_PATH}")