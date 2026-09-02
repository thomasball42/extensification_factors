"""
Brazil-shaped data plumbing for holdout_comparison_animals_brazil.py,
mirroring _holdout_common.py's structure but flat: Brazil's local pasture
panel has a single series type ("All pasture-based animal products") per
immediate region, not a cross-product of country x FAOSTAT item, so there's
no per-item outer loop here -- build_region_data builds one region panel
directly, the analog of _holdout_common.build_item_area_data run once
globally instead of once per FAOSTAT Item.

residualized_target_series / demeaned_target_series are NOT redefined here --
the driver imports them straight from the sibling _holdout_common.py, since
they're already generic given the panel/region_entry shape build_region_data
produces (entity=cod_rgi in place of entity=Area/country).
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent                # .../extensification_factors/validation/brazil
VALIDATION_DIR = FILE_DIR.parent                            # .../extensification_factors/validation
EXT_FACTORS_DIR = VALIDATION_DIR.parent                      # .../extensification_factors
REPO_ROOT = EXT_FACTORS_DIR.parent                            # .../Extensification

sys.path.insert(0, str(EXT_FACTORS_DIR))
from _stats import build_raw_diffs

DATA_PATH: Path = REPO_ROOT / "brazil" / "data" / "NL_local_domain.csv"

AREA_COL = "area_past_ha"
PRODUCTION_COL = "P_T"


def load_data() -> pd.DataFrame:
    """Load Brazil's local immediate-region pasture panel (see
    brazil/data/readme for the schema)."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df["year"] = df["year"].astype(int)
    return df


def load_groups(df, sample_n=None, random_seed=0):
    """Return the cod_rgi values to SCORE, optionally subsampled. Unlike the
    FAOSTAT version there's no item dimension to cross with -- every region
    always participates in the shared panel regardless of sampling here (see
    build_region_data)."""
    regions = sorted(df["cod_rgi"].unique().tolist())
    if sample_n is not None and len(regions) > sample_n:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(regions), size=sample_n, replace=False)
        regions = [regions[i] for i in idx]
    return regions


def build_region_data(df):
    """df: load_data()'s output. Returns (region_data, panel): region_data
    maps cod_rgi -> {"g": cleaned per-region frame, "diff_years": calendar
    years of its diffs}; panel is the raw (non-demeaned) diff panel across
    EVERY region, needed so residualize_two_way_fe_oos/demeaned_target_series
    can use every region's contemporaneous data, even for regions this run
    isn't scoring. Direct analog of _holdout_common.build_item_area_data,
    minus the outer per-item loop -- Brazil has a single series type."""
    region_data = {}
    panel_rows = []
    for cod_rgi, g in df.groupby("cod_rgi"):
        g = g.sort_values("year")
        g = g[np.isfinite(g[AREA_COL]) & np.isfinite(g[PRODUCTION_COL])]
        g = g[(g[AREA_COL] > 0) & (g[PRODUCTION_COL] > 0)]
        if len(g) < 2:
            continue
        with np.errstate(divide="ignore"):
            log_area = np.log(g[AREA_COL].values)
            log_prod = np.log(g[PRODUCTION_COL].values)
        years = g["year"].values.astype(int)
        diff_years, a_t, p_t = build_raw_diffs(years, log_area, log_prod)
        region_data[cod_rgi] = {"g": g, "diff_years": diff_years}
        panel_rows.append(pd.DataFrame({"entity": cod_rgi, "time": diff_years, "a": a_t, "p": p_t}))

    if not panel_rows:
        return region_data, pd.DataFrame(columns=["entity", "time", "a", "p"])
    return region_data, pd.concat(panel_rows, ignore_index=True)
