"""
Animals counterpart to _holdout_common.py: builds the leak-safe
cross-country panel (pasture area vs. grazing-animal production) that
holdout_comparison_animals.py scores, porting the animals data-prep logic
from _compute_beta_animals.py (config.json-driven grazing filters,
production summed across grazing commodities, pasture land summed across
grazing land-use codes). There is only one implicit "item" here -- all
pasture-based animal products, pooled per Area, exactly as
_compute_beta_animals.py pools them -- so there's no per-Item loop, unlike
the crops version.

residualized_target_series (imported by holdout_comparison_animals.py
from _holdout_common.py, unchanged) is fully generic over the panel's
entity/time columns and has no crop-specific assumptions, so it applies
here as-is.
"""

import json
import re

import numpy as np
import pandas as pd
from pathlib import Path

from _stats import build_raw_diffs

DATA_PATH: Path = Path("data") / "inputs"
CONFIG_PATH: Path = Path(__file__).resolve().parent.parent / "config.json"
MIN_RAW_OBS = 2  # matches _holdout_common.build_item_area_data's floor; real length filtering happens downstream in score_series


def load_area_data():
    """Returns (area_data, panel): area_data maps Area -> {"g": per-area
    frame with Year/Production columns, "diff_years": diff-year array};
    panel is the raw (non-demeaned) diff panel across every area, needed
    for two-way fixed-effects residualization. Mirrors
    _holdout_common.build_item_area_data's return shape, but for the
    single pooled animals series instead of per-crop-item groups."""
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
    year_value_col = re.compile(r"^Y\d{4}$")  # excludes flag (YxxxxF) and note (YxxxxN) columns
    land_use_df = pd.read_csv(
        DATA_PATH / "Inputs_LandUse_E_All_Data.csv",
        usecols=lambda c: c in ["Area", "Area Code", "Item", "Item Code", "Element", "Unit"] or year_value_col.match(c),
    )

    df = df[df["Item Code"].isin(grazing_animal_products.keys())]
    land_use_df = land_use_df[(land_use_df["Element"] == "Area") & (land_use_df["Item Code"].isin(grazing_land_types.keys()))]

    wide = df.pivot_table(
        index=["Area", "Area Code", "Item", "Item Code", "Year"], columns="Element", values="Value"
    ).reset_index()
    # sum production across all pasture-based commodities per area/year (NaN only if all commodities are NaN)
    wide = wide.groupby(["Area", "Area Code", "Year"], as_index=False)["Production"].sum(min_count=1)

    pasture_year_cols = [c for c in land_use_df.columns
                          if c not in ["Area", "Area Code", "Item", "Item Code", "Element", "Unit"]]
    pasture_years = np.array([int(c[1:]) for c in pasture_year_cols])

    area_data = {}
    panel_rows = []
    for (area, area_code), g in wide.groupby(["Area", "Area Code"]):
        g = g.sort_values("Year").dropna(subset=["Production"])
        if len(g) < MIN_RAW_OBS:
            continue

        years = g["Year"].values.astype(int)

        area_pasture_wide = land_use_df.loc[land_use_df["Area Code"] == area_code, pasture_year_cols]
        pasture_values = area_pasture_wide.values
        all_nan = np.all(np.isnan(pasture_values), axis=0)
        pasture_sum = np.nansum(pasture_values, axis=0)  # sum across all grazing land types
        pasture_sum[all_nan] = np.nan
        # align pasture years onto the production years (NaN where a production year has no land-use value)
        area_pasture = pd.Series(pasture_sum, index=pasture_years).reindex(years).values

        with np.errstate(divide="ignore"):
            log_pasture = np.log(area_pasture)
            log_P = np.log(g["Production"].values)
        diff_years, a_t, p_t = build_raw_diffs(years, log_pasture, log_P)

        area_data[area] = {"g": g, "diff_years": diff_years}
        panel_rows.append(pd.DataFrame({"entity": area, "time": diff_years, "a": a_t, "p": p_t}))

    if not panel_rows:
        return area_data, pd.DataFrame(columns=["entity", "time", "a", "p"])
    return area_data, pd.concat(panel_rows, ignore_index=True)