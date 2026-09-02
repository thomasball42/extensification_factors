"""
Compares Brazil's local, region-level beta estimates (brazil/outputs/
beta_animals_local_Brazil.csv, beta_animals_linear_local_Brazil.csv --
produced by _compute_beta_animals_brazil.py / _compute_beta_animals_brazil_
linear.py from Brazil's own municipality/region pasture data) against the
single Brazil row in the main suite's FAOSTAT country-level results
(outputs/beta_animals.csv, beta_animals_linear.csv).

The two are different resolutions of essentially the same quantity (pasture
area growth vs. pasture-based production growth), so this asks: does the
national FAOSTAT-derived beta for Brazil sit inside the distribution of betas
estimated directly from Brazil's own regions?

The regional data (2011-2021) only overlaps the FAOSTAT series (1962-2023)
over 2012-2021 (the first valid annual-diff year); the TVP comparison is
restricted to that overlap. The "regional weighted mean" uses each region's
current_production_kg as a fixed weight across years -- the wide beta output
doesn't carry a production series, only the latest value -- so it approximates
rather than exactly reproduces a production-weighted average for earlier years.
Run with working directory = extensification_factors/.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

NATIONAL_TVP_PATH = Path("outputs") / "beta_animals.csv"
NATIONAL_LINEAR_PATH = Path("outputs") / "beta_animals_linear.csv"
REGIONAL_TVP_PATH = Path("..") / "brazil" / "outputs" / "beta_animals_local_Brazil.csv"
REGIONAL_LINEAR_PATH = Path("..") / "brazil" / "outputs" / "beta_animals_linear_local_Brazil.csv"

OUT_PATH = Path("outputs") / "validation" / "brazil" / "beta_comparison_summary.csv"


def weighted_mean(values, weights):
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

national_tvp = pd.read_csv(NATIONAL_TVP_PATH)
national_linear = pd.read_csv(NATIONAL_LINEAR_PATH)
regional_tvp = pd.read_csv(REGIONAL_TVP_PATH)
regional_linear = pd.read_csv(REGIONAL_LINEAR_PATH)

national_tvp_bra = national_tvp[national_tvp["Area"] == "Brazil"]
national_linear_bra = national_linear[national_linear["Area"] == "Brazil"]
if len(national_tvp_bra) != 1 or len(national_linear_bra) != 1:
    raise ValueError(
        f"Expected exactly one 'Brazil' row in each national file, got "
        f"{len(national_tvp_bra)} (tvp) and {len(national_linear_bra)} (linear)."
    )
national_tvp_bra = national_tvp_bra.iloc[0]
national_linear_bra = national_linear_bra.iloc[0]

# ---------------------------------------------------------------------------
# TVP: national beta_<year>/se_<year> vs. the regional distribution, per
# overlapping year
# ---------------------------------------------------------------------------

national_years = {int(c.split("_")[1]) for c in national_tvp.columns if c.startswith("beta_")}
regional_years = {int(c.split("_")[1]) for c in regional_tvp.columns if c.startswith("beta_")}
overlap_years = sorted(national_years & regional_years)

if not overlap_years:
    raise ValueError("No overlapping beta_<year> columns between the national and regional TVP outputs.")

tvp_rows = []
for year in overlap_years:
    beta_col, se_col = f"beta_{year}", f"se_{year}"
    regional_beta = regional_tvp[beta_col]
    valid = regional_beta.notna()

    tvp_rows.append({
        "model": "tvp",
        "year": year,
        "national_beta": national_tvp_bra[beta_col],
        "national_se": national_tvp_bra[se_col],
        "n_regions": int(valid.sum()),
        "regional_mean": regional_beta[valid].mean(),
        "regional_median": regional_beta[valid].median(),
        "regional_std": regional_beta[valid].std(),
        "regional_weighted_mean": weighted_mean(
            regional_beta[valid].values, regional_tvp.loc[valid, "current_production_kg"].values
        ),
    })

tvp_df = pd.DataFrame(tvp_rows)

# ---------------------------------------------------------------------------
# Linear: single full-sample national beta vs. the regional distribution
# (one beta per region, not time-varying)
# ---------------------------------------------------------------------------

linear_valid = regional_linear["beta"].notna()
linear_row = {
    "model": "linear",
    "year": np.nan,
    "national_beta": national_linear_bra["beta"],
    "national_se": national_linear_bra["beta_se"],
    "n_regions": int(linear_valid.sum()),
    "regional_mean": regional_linear.loc[linear_valid, "beta"].mean(),
    "regional_median": regional_linear.loc[linear_valid, "beta"].median(),
    "regional_std": regional_linear.loc[linear_valid, "beta"].std(),
    "regional_weighted_mean": weighted_mean(
        regional_linear.loc[linear_valid, "beta"].values,
        regional_linear.loc[linear_valid, "current_production_kg"].values,
    ),
}

summary = pd.concat([tvp_df, pd.DataFrame([linear_row])], ignore_index=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT_PATH, index=False)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

pd.set_option("display.width", 120)
print(f"Overlap years: {overlap_years[0]}-{overlap_years[-1]} ({len(overlap_years)} years)")
print()
print("TVP beta by year (national FAOSTAT vs. Brazil regions):")
print(tvp_df.drop(columns="model").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()
print("Linear (full-sample) beta:")
print(pd.DataFrame([linear_row]).drop(columns=["model", "year"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()
print(f"Written to {OUT_PATH}")
