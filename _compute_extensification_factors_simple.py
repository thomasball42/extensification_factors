import pandas as pd
from pathlib import Path
import numpy as np
import scipy.stats as stats

DATA_PATH: Path = Path("mrio_pipeline", "input_data")
OUT_PATH: Path = Path("intensification_analysis", "calculated_parameters", "intensification_factors_simple.csv")

elements = ["Area harvested", "Production", "Yield"]
columns = ["Area", "Area Code", "Item", "Item Code", "Element", "Year", "Value", "Unit"]

df = pd.read_csv(
    DATA_PATH / "Production_Crops_Livestock_E_All_Data_(Normalized).csv",
    encoding="latin-1",
    low_memory=False,
    usecols=columns,
)

ha_unit = df.loc[df.Element=="Area harvested", "Unit"].values[0]
prod_unit = df.loc[df.Element=="Production", "Unit"].values[0]
yield_unit = df.loc[df.Element=="Yield", "Unit"].values[0]

df = df.drop(columns=["Unit"])
df = df[df.Element.isin(elements)]

# restrict to items that actually report "Area harvested" (i.e. crops, not
# livestock/animal-product items that reuse the "Production" element label)
crop_items = df.loc[df.Element == "Area harvested", "Item"].unique()
df = df[df.Item.isin(crop_items)]

wide = df.pivot_table(index=["Area", "Area Code", "Item", "Item Code", "Year"], columns="Element", values="Value").reset_index()

records = []
groups = wide.groupby(["Area", "Area Code", "Item", "Item Code"])
for i, ((area, area_code, item, item_code), g) in enumerate(groups):

    print(i + 1, "/", len(groups), area, area_code, item, item_code)

    g = g.sort_values("Year").dropna(subset=elements)
    if len(g) < 4:
        continue

    with np.errstate(divide="ignore"):
        log_AH = np.log(g["Area harvested"].values)
        log_P = np.log(g["Production"].values)

    diff_log_AH = np.diff(log_AH)
    diff_log_P = np.diff(log_P)

    mask = np.isfinite(diff_log_AH) & np.isfinite(diff_log_P)
    if mask.sum() < 3:
        continue

    try:
        beta, intercept, r_value, p_value, std_err = stats.linregress(diff_log_P[mask], diff_log_AH[mask])
    except Exception as e:
        print(f"Error for {area} - {item}: {e}")
        continue

    
    records.append({
        "Area": area,
        "Area Code": area_code,
        "Item": item,
        "Item Code": item_code,
        "current_year": int(g["Year"].values[-1]),
        f"current_area_harvested_{ha_unit}": g["Area harvested"].values[-1],
        f"current_production_{prod_unit}": g["Production"].values[-1],
        f"current_yield_{yield_unit}": g["Yield"].values[-1],
        "beta": beta,
        "beta_std_err": std_err,
        "n_obs": int(mask.sum()),
    })

out = pd.DataFrame.from_records(records)
out.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(out)} rows to {OUT_PATH}")
