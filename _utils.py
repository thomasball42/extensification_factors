import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Item/Area name filtering (plotting scripts)
# ---------------------------------------------------------------------------

def filter_list(all_, filt_, excl_):
    """Substring-filter `all_` (case-insensitive) down to entries matching
    any of `filt_`, then drop any of those that also match `excl_` --
    unless the entry is an exact (lowercased) match for one of `filt_`
    entries, which bypasses exclusion. `filt_` empty means "no filtering",
    returning `all_` unchanged (exclusions don't apply in that case)."""
    if not filt_:
        return all_
    filt = [f.lower() for f in filt_]
    excl = [f.lower() for f in excl_]
    return [
        i for i in all_
        if any(f in i.lower() for f in filt)
        and (i.lower() in filt or not any(f in i.lower() for f in excl))
    ]


# ---------------------------------------------------------------------------
# FAOSTAT aggregate-item flagging
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_q_prior_config(config_path: Path = CONFIG_PATH):
    """Returns (q_prior_scale, q_prior_strength) from the "q_prior" section of
    config.json, for use as fit_tvp_beta's shrinkage-prior kwargs."""
    with open(config_path) as f:
        config = json.load(f)

    q_prior_cfg = config.get("q_prior", {})
    return q_prior_cfg.get("q_prior_scale"), q_prior_cfg.get("q_prior_strength", 1.0)


def load_area_method(config_path: Path = CONFIG_PATH) -> str:
    """Returns the "area_method" from config.json (e.g. "harvested_area"),
    which names the outputs/ subfolder that results are written to/read from."""
    with open(config_path) as f:
        config = json.load(f)

    return config["area_method"]["area_method"]


AREA_METHOD: str = load_area_method()
OUTPUTS_PATH: Path = Path("outputs") / AREA_METHOD
LINEAR_OUTPUTS_PATH: Path = OUTPUTS_PATH / "linear"  # static OLS betas (_compute_beta_*_linear.py)
NONLINEAR_OUTPUTS_PATH: Path = OUTPUTS_PATH / "time_varying"  # TVP betas (_compute_beta_*.py)
VALIDATION_OUTPUTS_PATH: Path = OUTPUTS_PATH / "validation"  # validation/*.py


# ---------------------------------------------------------------------------
# Crop area series (area_method)
# ---------------------------------------------------------------------------

# wide-frame column each crop's area series is taken from (see add_physical_area)
AREA_COL: str = {"harvested_area": "Area harvested", "physical_area": "Physical area"}[AREA_METHOD]

NATIONAL_CI_PATH: Path = Path("data") / "processed" / "national_cropping_intensity.csv"  # 00_build_national_ci.py
# ci_cropped, not ci_cropland: ci_cropland's denominator is a static (~2010)
# cropland map, so its year-to-year change is just the satellite harvested
# area -- it would book cropland expansion/abandonment inside that map as a
# change in cropping intensity
PHYSICAL_AREA_CI: str = "ci_cropped"


def add_physical_area(wide: pd.DataFrame) -> pd.DataFrame:
    """Adds a "Physical area" column to a wide (one column per Element)
    FAOSTAT crop frame: Area harvested / CI, with CI the national cropping
    intensity for that Area Code and Year -- i.e. the country's physical
    cropland, sum(Area harvested) / CI, allocated to crops by harvested-area
    share. NaN where there's no CI: outside the GCI years (2001-2019) and for
    areas the GCI doesn't cover (regional aggregates, former states)."""
    ci = pd.read_csv(NATIONAL_CI_PATH, usecols=["Area Code", "Year", PHYSICAL_AREA_CI])
    ci = ci.dropna(subset=["Area Code"]).astype({"Area Code": int})
    wide = wide.merge(ci, on=["Area Code", "Year"], how="left", validate="many_to_one")
    wide["Physical area"] = wide["Area harvested"] / wide.pop(PHYSICAL_AREA_CI)
    return wide


FAOSTAT_CROPS_PATH: Path = Path("data") / "inputs" / "Production_Crops_Livestock_E_All_Data_(Normalized).csv"
CROP_ELEMENTS: list[str] = ["Area harvested", "Production", "Yield"]
# a year is used only if all of these are present
CROP_SERIES_COLS: list[str] = [AREA_COL, "Production", "Yield"]


def load_crop_wide(with_physical: bool = AREA_METHOD == "physical_area") -> pd.DataFrame:
    """FAOSTAT crop series (Area harvested / Production / Yield) in wide (one
    column per Element) form, one row per (Area, Area Code, Item, Item Code,
    Year), restricted to items that report Area harvested -- plus "Physical
    area" if with_physical (see add_physical_area). Used by the validation
    scripts; the compute scripts build the same frame inline."""
    df = pd.read_csv(
        FAOSTAT_CROPS_PATH, encoding="latin-1", low_memory=False,
        usecols=["Area", "Area Code", "Item", "Item Code", "Element", "Year", "Value"],
    )
    df = df[df.Element.isin(CROP_ELEMENTS)]
    crop_items = df.loc[df.Element == "Area harvested", "Item"].unique()
    df = df[df.Item.isin(crop_items)]
    wide = df.pivot_table(
        index=["Area", "Area Code", "Item", "Item Code", "Year"], columns="Element", values="Value"
    ).reset_index()
    if with_physical:
        wide = add_physical_area(wide)
    return wide


def load_aggregate_matcher(config_path: Path = CONFIG_PATH):
    """Returns an is_aggregate(item_name) -> bool function built from the
    "faostat_aggregates" section of config.json: an exact-match set of known
    aggregate names plus a set of case-insensitive substring patterns."""
    with open(config_path) as f:
        config = json.load(f)

    agg_cfg = config.get("faostat_aggregates", {})
    known_names = set(agg_cfg.get("known_names", []))
    name_patterns = [p.lower() for p in agg_cfg.get("name_patterns", [])]

    def is_aggregate(item_name: str) -> bool:
        if item_name in known_names:
            return True
        item_lower = item_name.lower()
        return any(pattern in item_lower for pattern in name_patterns)

    return is_aggregate
