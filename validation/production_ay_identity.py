"""
Checks the identity the method rests on, Production = Area harvested * Yield
(paper Eq. 1), in the FAOSTAT data, over the rows the compute scripts draw
on: every year with all of CROP_SERIES_COLS present -- so under
area_method = "physical_area" only the 2001-2019 rows of countries with a
cropping-intensity estimate.

Physical area needs no separate check: Physical area = Area harvested / CI,
so Production = Physical area * CI * Yield holds exactly when this does (in
log differences, p_t = a_t + kappa_t + y_t).

Writes the rows whose relative discrepancy exceeds TOLERANCE, plus histogram
data for the |relative discrepancy| distribution (plotting/plot_ay_identity.py).
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _utils import AREA_METHOD, CROP_ELEMENTS, CROP_SERIES_COLS, FAOSTAT_CROPS_PATH, VALIDATION_OUTPUTS_PATH, load_crop_wide

OUT_PATH: Path = VALIDATION_OUTPUTS_PATH / "ay_identity_flagged.csv"
OUT_HIST_PATH: Path = VALIDATION_OUTPUTS_PATH / "ay_identity_histogram.csv"

TOLERANCE = 0.01  # 1% relative discrepancy flagged for review
HIST_CLIP = 0.05  # histogram range; larger discrepancies land in the last bin
YIELD_TO_T_PER_HA = {"kg/ha": 1e-3, "hg/ha": 1e-4, "t/ha": 1.0}


if __name__ == "__main__":
    wide = load_crop_wide()

    units = pd.read_csv(FAOSTAT_CROPS_PATH, encoding="latin-1", usecols=["Item Code", "Element", "Unit"])
    units = units[units["Item Code"].isin(wide["Item Code"].unique()) & units["Element"].isin(CROP_ELEMENTS)]
    units = units.drop_duplicates(["Element", "Unit"]).set_index("Element")["Unit"]
    assert units.index.is_unique, f"more than one unit per element: {units.to_dict()}"
    assert units["Area harvested"] == "ha" and units["Production"] == "t", units.to_dict()
    print(f"Units -- Area: {units['Area harvested']}, Production: {units['Production']}, Yield: {units['Yield']}")

    wide = wide.dropna(subset=CROP_SERIES_COLS)
    wide["implied_production"] = wide["Area harvested"] * wide["Yield"] * YIELD_TO_T_PER_HA[units["Yield"]]
    wide["relative_discrepancy"] = (wide["implied_production"] - wide["Production"]) / wide["Production"]

    abs_disc = wide["relative_discrepancy"].abs()

    print(f"=== Identity check: Production vs. Area harvested * Yield ({AREA_METHOD} rows, "
          f"{wide['Year'].min()}-{wide['Year'].max()}) ===")
    print(f"Rows checked: {len(wide):,}")
    print(f"Median |relative discrepancy|:  {abs_disc.median():.4%}")
    print(f"95th percentile:                {abs_disc.quantile(0.95):.4%}")
    print(f"99th percentile:                {abs_disc.quantile(0.99):.4%}")
    print(f"Max:                            {abs_disc.max():.4%}")
    print(f"Rows exceeding {TOLERANCE:.0%} tolerance: "
          f"{(abs_disc > TOLERANCE).sum():,} ({(abs_disc > TOLERANCE).mean():.2%})")

    flagged = wide.loc[abs_disc > TOLERANCE].sort_values("relative_discrepancy", key=abs, ascending=False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(OUT_PATH, index=False)
    print(f"\nFlagged rows (> {TOLERANCE:.0%} discrepancy) written to {OUT_PATH}")

    # histogram data for the Technical Validation figure -- distribution of
    # |relative discrepancy| across all rows (expect a sharp spike near 0)
    finite_abs_disc = abs_disc[np.isfinite(abs_disc)]
    n_non_finite = len(abs_disc) - len(finite_abs_disc)
    if n_non_finite:
        print(f"Excluding {n_non_finite:,} rows with non-finite discrepancy "
              f"(Production == 0) from histogram")
    hist_counts, hist_edges = np.histogram(np.clip(finite_abs_disc, 0, HIST_CLIP), bins=50)
    hist_out = pd.DataFrame({"bin_left": hist_edges[:-1], "bin_right": hist_edges[1:], "count": hist_counts})
    hist_out.to_csv(OUT_HIST_PATH, index=False)
    print(f"Histogram data written to {OUT_HIST_PATH}")
