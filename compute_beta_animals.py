import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from _kalman_functions import kalman_filter, rts_smoother, fit_tvp_beta, build_annual_diffs

DATA_PATH: Path = Path("data") / "inputs" # this needs the full production dataset from FAO
OUT_PATH: Path = Path("outputs", "beta_animals.csv")
CONFIG_PATH: Path = Path(__file__).parent / "config.json"

with open(CONFIG_PATH) as f:
    _config = json.load(f)

MIN_OBS: int = 15  # minimum number of valid (non-missing) yearly diff pairs required to fit

elements = ["Production"]
columns = ["Area", "Area Code", "Item", "Item Code", "Element", "Year", "Value", "Unit"]

# ---------------------------------------------------------------------------
# Main 
# ---------------------------------------------------------------------------

raw_animal_products = {int(k): v for k, v in _config["raw_animal_products"].items()}
land_use_codes = {int(k): v for k, v in _config["land_use_codes"].items()}

grazing_animal_products = {k:v for k,v in raw_animal_products.items() if v["grazing"]}
grazing_land_types = {k:v for k,v in land_use_codes.items() if v["grazing"]} # not comprehensive because some would be double counting


df = pd.read_csv(
    DATA_PATH / "Production_Crops_Livestock_E_All_Data_(Normalized).csv",
    encoding="latin-1",
    low_memory=False,
    # usecols=columns,
)

year_value_col = re.compile(r"^Y\d{4}$")  # excludes flag (YxxxxF) and note (YxxxxN) columns
land_use_df = pd.read_csv(
    DATA_PATH / "Inputs_LandUse_E_All_Data.csv",
    usecols=lambda c: c in ["Area", "Area Code", "Item", "Item Code", "Element", "Unit"] or year_value_col.match(c),
)

# filter commodities
df = df[df["Item Code"].isin(grazing_animal_products.keys())]

# filter land use
land_use_df = land_use_df[(land_use_df["Element"]=="Area") & (land_use_df["Item Code"].isin(grazing_land_types.keys()))]

# get units
ha_unit = land_use_df.loc[land_use_df.Element=="Area", "Unit"].values[0]
prod_unit = df.loc[df.Element=="Production", "Unit"].values[0]

wide = df.pivot_table(
    index=["Area", "Area Code", "Item", "Item Code", "Year"], columns="Element", values="Value"
).reset_index()

# sum production across all pasture-based commodities per area/year (NaN only if all commodities are NaN)
wide = wide.groupby(["Area", "Area Code", "Year"], as_index=False)["Production"].sum(min_count=1)

records = []
groups = wide.groupby(["Area", "Area Code"])
for i, ((area, area_code), g) in enumerate(groups):

    print(i + 1, "/", len(groups), area)

    g = g.sort_values("Year").dropna(subset=elements)
    if len(g) < MIN_OBS + 1:
        continue

    years = g["Year"].values.astype(int)

    pasture_year_cols = [c for c in land_use_df.columns
                          if c not in ["Area", "Area Code", "Item", "Item Code", "Element", "Unit"]]
    area_pasture_wide = land_use_df.loc[land_use_df["Area Code"] == area_code, pasture_year_cols]

    pasture_values = area_pasture_wide.values
    all_nan = np.all(np.isnan(pasture_values), axis=0)

    pasture_sum = np.nansum(pasture_values, axis=0)  # sum across all grazing land types
    pasture_sum[all_nan] = np.nan

    pasture_years = np.array([int(c[1:]) for c in pasture_year_cols])
    # align pasture years onto the production years (NaN where a production year has no land-use value)
    area_pasture = pd.Series(pasture_sum, index=pasture_years).reindex(years).values

    with np.errstate(divide="ignore"):
        log_pasture_ha = np.log(area_pasture)
        log_P = np.log(g["Production"].values)

    diff_years, a_t, p_t = build_annual_diffs(years, log_pasture_ha, log_P)

    fit = fit_tvp_beta(a_t, p_t, min_obs=MIN_OBS)
    if fit is None:
        continue

    record = {
            "Area": area,
            "Area Code": area_code,
            "Item": "All pasture-based animal products",
            "Item Code": 0,
            "current_year": int(years[-1]),
            f"current_area_harvested_{ha_unit}": area_pasture[~np.isnan(area_pasture)][-1],
            f"current_production_{prod_unit}": g["Production"].values[-1],
            f"current_yield_{None}": np.nan,
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