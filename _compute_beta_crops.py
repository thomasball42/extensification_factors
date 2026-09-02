"""
Time-varying-parameter (Kalman filter/smoother) version of the extensification-
factor pipeline.

Replaces OLS with a single state-space model per
(Area, Item) series:

    a_t = beta_t * p_t + u_t          u_t ~ N(0, R)      [observation eq.]
    beta_t = beta_{t-1} + eta_t       eta_t ~ N(0, Q)     [state eq., random walk]

where a_t = Delta ln(Area harvested), p_t = Delta ln(Production). Before
fitting, a_t and p_t are residualized against two-way (country + year) fixed
effects -- estimated jointly across every country growing a given Item -- so
that a shock hitting many countries in the same year (a global price spike, a
widespread drought) is absorbed into a year effect instead of leaking into a
country's estimated beta_t; see residualize_two_way_fe. Q and R are then
estimated per series by maximum likelihood. Output is the *smoothed* beta_t
(uses the whole series, not just past data) plus its standard error, for
every year.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from _stats import (
    fit_tvp_beta, build_raw_diffs, build_annual_diffs, residualize_two_way_fe,
)
from _utils import load_aggregate_matcher, load_q_prior_config

DATA_PATH: Path = Path("data") / "inputs" # this needs the full production dataset from FAO
OUT_PATH: Path = Path("outputs", "beta_crops.csv")

MIN_OBS: int = 15  # minimum number of valid (non-missing) yearly diff pairs required to fit
USE_Q_PRIOR: bool = True  # shrinkage prior on Q (see _stats.q_shrinkage_penalty); False = unregularized MLE
USE_RESIDUALIZATION: bool = False # two-way (country + year) FE residualization; False = plain demeaning (build_annual_diffs)
                                  # validation steps indicate this doesn't improve things so it's turned off 
is_aggregate = load_aggregate_matcher()
Q_PRIOR_SCALE, Q_PRIOR_STRENGTH = load_q_prior_config() if USE_Q_PRIOR else (None, 1.0)

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

# restrict to items that report "Area harvested"
crop_items = df.loc[df.Element == "Area harvested", "Item"].unique()
df = df[df.Item.isin(crop_items)]

wide = df.pivot_table(
    index=["Area", "Area Code", "Item", "Item Code", "Year"], columns="Element", values="Value"
).reset_index()

records = []
items = wide[["Item", "Item Code"]].drop_duplicates().values.tolist()

for i, (item, item_code) in enumerate(items):
    g_item = wide[(wide["Item"] == item) & (wide["Item Code"] == item_code)]

    area_data = {}
    for (area, area_code), g in g_item.groupby(["Area", "Area Code"]):
        g = g.sort_values("Year").dropna(subset=elements)
        if len(g) < MIN_OBS + 1:
            continue
        area_data[area_code] = {"area": area, "g": g}

    if not area_data:
        continue

    print(i + 1, "/", len(items), item, item_code, f"({len(area_data)} areas)")

    if USE_RESIDUALIZATION:
        # build the raw (non-demeaned), calendar-year-reindexed diff panel
        # across every country growing this item, so the fixed effects are
        # identified from cross-country variation, not just one series in isolation.
        panel_rows = []
        for area_code, meta in area_data.items():
            g = meta["g"]
            with np.errstate(divide="ignore"):
                log_AH = np.log(g["Area harvested"].values)
                log_P = np.log(g["Production"].values)
            years = g["Year"].values.astype(int)
            diff_years, a_t, p_t = build_raw_diffs(years, log_AH, log_P)
            panel_rows.append(pd.DataFrame({
                "entity": area_code, "time": diff_years, "a": a_t, "p": p_t,
            }))

        panel = residualize_two_way_fe(pd.concat(panel_rows, ignore_index=True))
        diffs_by_area = {}
        for area_code, sub in panel.groupby("entity"):
            sub = sub.sort_values("time")
            diffs_by_area[area_code] = (sub["time"].values.astype(int), sub["a_resid"].values, sub["p_resid"].values)
    else:
        # plain demeaning, no cross-country fixed-effects adjustment
        diffs_by_area = {}
        for area_code, meta in area_data.items():
            g = meta["g"]
            with np.errstate(divide="ignore"):
                log_AH = np.log(g["Area harvested"].values)
                log_P = np.log(g["Production"].values)
            years = g["Year"].values.astype(int)
            diffs_by_area[area_code] = build_annual_diffs(years, log_AH, log_P)

    # fit the TVP Kalman model per country on its residualized/demeaned diffs.
    for area_code, meta in area_data.items():
        area, g = meta["area"], meta["g"]
        diff_years, a_t, p_t = diffs_by_area[area_code]

        fit = fit_tvp_beta(a_t, p_t, min_obs=MIN_OBS, q_prior_scale=Q_PRIOR_SCALE, q_prior_strength=Q_PRIOR_STRENGTH)
        if fit is None:
            continue

        years = g["Year"].values.astype(int)
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
            "q_prior_scale": Q_PRIOR_SCALE,
            "q_prior_strength": Q_PRIOR_STRENGTH,
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