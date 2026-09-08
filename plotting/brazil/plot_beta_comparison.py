"""
Plots Brazil's local, region-level TVP beta estimates against the single
Brazil row in the main suite's FAOSTAT country-level TVP results -- the
visual counterpart to validation/brazil/compare_beta_animals.py.

Most recent year common to both the national and regional panels only:
national beta (+- SE) as a single reference value, vs. the distribution of
that same year's regional betas.

Run with working directory = extensification_factors/.
"""

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

NATIONAL_TVP_PATH = Path("outputs") / "beta_animals.csv"
REGIONAL_TVP_PATH = Path("outputs") / "brazil" / "beta_animals_local_Brazil.csv"
FIGS_PATH = Path("..") / "figs" / "brazil"
SAVE = True

NATIONAL_COLOR = "#2a78d6"  # blue -- matches BASE_PALETTE[0] used across sibling plotting scripts
REGIONAL_COLOR = "#eb6834"  # orange -- matches BASE_PALETTE[1]

os.makedirs(FIGS_PATH, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

national_tvp = pd.read_csv(NATIONAL_TVP_PATH)
regional_tvp = pd.read_csv(REGIONAL_TVP_PATH)

national_tvp_bra = national_tvp[national_tvp["Area"] == "Brazil"].iloc[0]

# ---------------------------------------------------------------------------
# Most recent shared year only: national point (+- SE) vs. the distribution
# of that year's regional betas
# ---------------------------------------------------------------------------

national_years = {int(c.split("_")[1]) for c in national_tvp.columns if c.startswith("beta_")}
regional_years = {int(c.split("_")[1]) for c in regional_tvp.columns if c.startswith("beta_")}
latest_year = max(national_years & regional_years)

national_beta = float(national_tvp_bra[f"beta_{latest_year}"])
national_se = float(national_tvp_bra[f"se_{latest_year}"])
regional_beta = regional_tvp[f"beta_{latest_year}"].dropna().values

fig, ax = plt.subplots(figsize=(9, 6))

for xline in (0, 1):
    ax.axvline(x=xline, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

ax.hist(regional_beta, bins=30, color=REGIONAL_COLOR, alpha=0.6, edgecolor="white",
        label=f"Regional (n={len(regional_beta)} immediate regions)")
ax.axvspan(national_beta - national_se, national_beta + national_se, color=NATIONAL_COLOR, alpha=0.15, linewidth=0)
ax.axvline(national_beta, color=NATIONAL_COLOR, linewidth=2.5,
           label=f"National (FAOSTAT): {national_beta:.3f} +/- {national_se:.3f}")

ax.set_xlabel(r"Extensification parameter $\beta$ (pasture-based animal products, TVP/Kalman)")
ax.set_ylabel("Number of regions")
ax.set_title(f"Brazil {latest_year}: national (FAOSTAT) vs. regional (local pasture data) extensification beta")
ax.legend(loc="best")

fig.tight_layout()
if SAVE:
    fig.savefig(FIGS_PATH / "beta_tvp_comparison_animals_brazil.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to {FIGS_PATH}")
