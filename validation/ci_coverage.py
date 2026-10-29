"""
Which areas the physical-area crop betas lose relative to the harvested-area
ones, and why. Physical area needs a national cropping intensity (CI, from
00_build_national_ci.py), which covers only 2001-2019 and only areas where
the GCI rasters detect cropping, so an area with harvested-area betas
(outputs/harvested_area/time_varying/beta_crops.csv) can be missing from
outputs/physical_area/time_varying/beta_crops.csv. Each missing area gets the
first status that applies:

  regional aggregate                 FAOSTAT region (Area Code >= 5000), not a country
  FAOSTAT composite                  sum of areas that are reported separately (COMPOSITE_AREAS)
  no FAOSTAT data in CI years        former states (USSR, Yugoslav SFR, ...)
  too few FAOSTAT years in CI years  no crop has MIN_OBS consecutive-year diffs inside the CI
                                     years even before the CI is considered: lost to the
                                     window, not to the CI
  no CI for this area code           not in the CI file (outside the boundary set, or no FAOSTAT code)
  no cropping detected               in the CI file, but ci_cropped is missing in every year
  CI in too few years                ci_cropped is missing in enough years that no crop
                                     reaches MIN_OBS diffs

Writes one row per area with harvested-area betas (ci_coverage_by_area.csv)
and a count per status (ci_coverage_summary.csv). Series counts and areas
are over non-aggregate crops; the area is each series' current (last-year)
harvested area, so former states contribute their last year's area.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _utils import CROP_ELEMENTS, NATIONAL_CI_PATH, PHYSICAL_AREA_CI, load_crop_wide

HARVESTED_PATH: Path = Path("outputs") / "harvested_area" / "time_varying" / "beta_crops.csv"
PHYSICAL_PATH: Path = Path("outputs") / "physical_area" / "time_varying" / "beta_crops.csv"
OUT_DIR: Path = Path("outputs") / "physical_area" / "validation"

MIN_OBS: int = 15  # matches _compute_beta_crops.py
REGION_CODE_MIN: int = 5000  # FAOSTAT region codes start here
COMPOSITE_AREAS: dict[int, str] = {351: "China = mainland + Taiwan + Hong Kong + Macao"}
STATUSES = [
    "kept",
    "regional aggregate",
    "FAOSTAT composite",
    "no FAOSTAT data in CI years",
    "too few FAOSTAT years in CI years",
    "no CI for this area code",
    "no cropping detected",
    "CI in too few years",
]
SERIES_KEYS = ["Area Code", "Item Code"]


def betas_by_area(path):
    """Series with at least one beta in a beta_crops.csv, as per-area counts
    and current harvested area (non-aggregate crops)."""
    out = pd.read_csv(path)
    out = out[out[[c for c in out.columns if c.startswith("beta_")]].notna().any(axis=1)]
    area_col = next(c for c in out.columns if c.startswith("current_area_harvested_"))
    crops = out[~out["is_aggregate"]]
    return out.groupby(["Area Code", "Area"]).agg(current_year=("current_year", "max")).join(
        crops.groupby(["Area Code", "Area"]).agg(n_series=("Item", "size"), harvested_area_ha=(area_col, "sum"))
    ).fillna({"n_series": 0}).astype({"n_series": int}).reset_index()


def max_consecutive_diffs(rows):
    """Per Area Code: the most consecutive-year diffs any one crop has in `rows`."""
    rows = rows.sort_values(SERIES_KEYS + ["Year"])
    consecutive = rows.groupby(SERIES_KEYS)["Year"].diff().eq(1)
    return consecutive.groupby([rows["Area Code"], rows["Item Code"]]).sum().groupby(level="Area Code").max()


if __name__ == "__main__":
    harv = betas_by_area(HARVESTED_PATH)
    phys = betas_by_area(PHYSICAL_PATH).set_index("Area Code")

    ci = pd.read_csv(NATIONAL_CI_PATH).dropna(subset=["Area Code"]).astype({"Area Code": int})
    ci_first, ci_last = ci["Year"].min(), ci["Year"].max()
    ci_years = ci.dropna(subset=[PHYSICAL_AREA_CI]).groupby("Area Code")["Year"].nunique()

    wide = load_crop_wide(with_physical=False).dropna(subset=CROP_ELEMENTS)
    fao_last_year = wide.groupby("Area Code")["Year"].max()
    in_window = wide[wide["Year"].between(ci_first, ci_last)]
    diffs_fao_only = max_consecutive_diffs(in_window)
    with_ci = in_window.merge(ci[["Area Code", "Year", PHYSICAL_AREA_CI]], on=["Area Code", "Year"])
    diffs_with_ci = max_consecutive_diffs(with_ci.dropna(subset=[PHYSICAL_AREA_CI]))

    def status(r):
        code = r["Area Code"]
        if code in phys.index:
            return "kept"
        if code >= REGION_CODE_MIN:
            return "regional aggregate"
        if code in COMPOSITE_AREAS:
            return "FAOSTAT composite"
        if fao_last_year.get(code, 0) < ci_first:
            return "no FAOSTAT data in CI years"
        if diffs_fao_only.get(code, 0) < MIN_OBS:
            return "too few FAOSTAT years in CI years"
        if code not in ci["Area Code"].values:
            return "no CI for this area code"
        if ci_years.get(code, 0) == 0:
            return "no cropping detected"
        if diffs_with_ci.get(code, 0) < MIN_OBS:
            return "CI in too few years"
        raise ValueError(f"{r['Area']} ({code}) has enough CI years but no physical-area betas")

    by_area = harv.rename(columns={"n_series": "n_series_harvested", "current_year": "current_year_harvested"})
    by_area["status"] = by_area.apply(status, axis=1)
    by_area["n_series_physical"] = by_area["Area Code"].map(phys["n_series"]).fillna(0).astype(int)
    by_area["fao_last_year"] = by_area["Area Code"].map(fao_last_year)
    by_area["ci_years"] = by_area["Area Code"].map(ci_years).fillna(0).astype(int)
    by_area["max_diffs_fao_in_ci_years"] = by_area["Area Code"].map(diffs_fao_only).fillna(0).astype(int)
    by_area["max_diffs_with_ci"] = by_area["Area Code"].map(diffs_with_ci).fillna(0).astype(int)
    by_area["note"] = by_area["Area Code"].map(COMPOSITE_AREAS)
    by_area = by_area[["Area", "Area Code", "status", "n_series_harvested", "n_series_physical", "harvested_area_ha",
                       "current_year_harvested", "fao_last_year", "ci_years", "max_diffs_fao_in_ci_years",
                       "max_diffs_with_ci", "note"]].copy()
    by_area["status"] = pd.Categorical(by_area["status"], categories=STATUSES, ordered=True)
    by_area = by_area.sort_values(["status", "harvested_area_ha"], ascending=[True, False])

    not_in_harvested = sorted(set(phys["Area"]) - set(harv["Area"]))
    assert not not_in_harvested, f"physical-area betas for areas without harvested-area betas: {not_in_harvested}"

    summary = by_area.groupby("status", observed=False).agg(
        n_areas=("Area", "size"),
        n_series_harvested=("n_series_harvested", "sum"),
        n_series_physical=("n_series_physical", "sum"),
        harvested_area_mha=("harvested_area_ha", lambda x: x.sum() / 1e6),
        areas=("Area", lambda x: "; ".join(x) if len(x) <= 25 else ""),
    )
    # regions and composites double count the countries they're made of
    summary.loc[["regional aggregate", "FAOSTAT composite"], "harvested_area_mha"] = np.nan

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_area.to_csv(OUT_DIR / "ci_coverage_by_area.csv", index=False)
    summary.to_csv(OUT_DIR / "ci_coverage_summary.csv")

    print(f"Areas with harvested-area betas: {len(by_area)}; with physical-area betas: {len(phys)} "
          f"(CI years {ci_first}-{ci_last}, MIN_OBS = {MIN_OBS} diffs)\n")
    print(summary.drop(columns="areas").to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nWritten to {OUT_DIR}: ci_coverage_by_area.csv, ci_coverage_summary.csv")
