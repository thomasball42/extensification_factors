"""
Distribution of the current (most recent year) extensification parameter
beta across countries, grouped by continent -- the cross-sectional
complement to beta_map_crops.py's time-windowed maps.

One point (+/- SE) per country: FAOSTAT aggregate items are excluded and
each country's items are combined via a production-weighted mean/SE (see
_geo.weighted_mean_se), using each item's own most recent (current_year)
beta/se rather than a fixed calendar year, since not every item's series
necessarily runs through the same last year.
"""

import os
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _geo import load_country_codes, load_world, weighted_mean_se
from _functions import filter_list as _filter_list

dat_path = Path("outputs") / "beta_crops.csv"
figs_path = Path("..") / "figs" / "beta_distribution_by_region_crops"
SAVE = True

# crop filtering (kept in sync with beta_map_crops.py / beta_timeseries_crops.py /
# beta_linear_crops.py); ifilt empty means "no filtering" -- mean across all crops
ifilt = [
        # "rice",
        "wheat",
        # "maize",
        # "sorghum",
        # "soy",
        # "barley",
        # "Cereal"
        ]

iexcl = [
        "buckwheat", "green corn", "n.e.c."
         ]

os.makedirs(figs_path, exist_ok=True)

BASE_PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# ---------------------------------------------------------------------------
# Load + prep
# ---------------------------------------------------------------------------

df = pd.read_csv(dat_path)
df = df[~df["is_aggregate"]].copy()

items = _filter_list(df["Item"].unique(), ifilt, iexcl)
df = df[df["Item"].isin(items)].copy()

prod_col = next(c for c in df.columns if c.startswith("current_production_"))

# each item's own most recent beta/se, not a fixed calendar year
beta_year_cols = sorted((c for c in df.columns if c.startswith("beta_")), key=lambda c: int(c.split("_")[1]))
years_avail = [int(c.split("_")[1]) for c in beta_year_cols]
col_of_year = {y: i for i, y in enumerate(years_avail)}

current_year = df["current_year"].astype(int)
col_pos = current_year.map(col_of_year)
has_col = col_pos.notna()
df["beta_current"] = np.nan
df["se_current"] = np.nan
row_pos = np.flatnonzero(has_col.values)
col_pos_valid = col_pos[has_col].astype(int).values
df.loc[has_col, "beta_current"] = df[beta_year_cols].values[row_pos, col_pos_valid]
df.loc[has_col, "se_current"] = df[[f"se_{y}" for y in years_avail]].values[row_pos, col_pos_valid]

country_codes = load_country_codes()
df = df.merge(country_codes, left_on="Area Code", right_on="FAOSTAT", how="inner")

# production-weighted aggregate across items, per country
country_stats = df.groupby("ISO3").apply(
    lambda g: pd.Series(dict(zip(
        ["beta", "se"], weighted_mean_se(g["beta_current"], g["se_current"], g[prod_col])
    ))),
    include_groups=False,
).dropna(subset=["beta"])

world = load_world()
country_stats = country_stats.merge(
    world[["ISO3", "CONTINENT", "NAME"]], on="ISO3", how="inner"
)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

continents = (
    country_stats.groupby("CONTINENT")["beta"].median().sort_values().index.tolist()
)
continent_color = {c: col for c, col in zip(continents, BASE_PALETTE)}
continent_n = country_stats["CONTINENT"].value_counts()

rng = np.random.default_rng(0)
x_of_continent = {c: i for i, c in enumerate(continents)}
jitter = rng.uniform(-0.3, 0.3, size=len(country_stats))
x = country_stats["CONTINENT"].map(x_of_continent).values + jitter

fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(continents) + 2), 7))

for _ in [0, 1]:
    ax.axhline(y=_, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

for c in continents:
    mask = country_stats["CONTINENT"].values == c
    ax.errorbar(
        x[mask], country_stats.loc[mask, "beta"], yerr=country_stats.loc[mask, "se"],
        fmt="o", markersize=5, capsize=2.5, linewidth=1, alpha=0.75,
        color=continent_color[c], label=c,
    )

ax.set_xticks(range(len(continents)))
ax.set_xticklabels([f"{c}\n(n={continent_n[c]})" for c in continents])
ax.set_xlim(-0.6, len(continents) - 0.4)

# y-limits from the point estimates only (not the SEs) so a handful of very
# uncertain series don't compress the rest of the plot into a thin band --
# their error bars are left to extend past the visible range
lo, hi = np.nanpercentile(country_stats["beta"], [1, 99])
pad = 0.15 * (hi - lo)
ax.set_ylim(lo - pad, hi + pad)

ax.set_ylabel(r"Extensification parameter $\beta$ (prior'd TVP/Kalman, current year, "
              r"production-weighted across crops)")
# ax.set_title("Distribution of country-level crop extensification betas, by continent\n"
#               "(a few very uncertain series have SE bars extending past the plotted range)")

fig.tight_layout()

if SAVE:
    fig.savefig(figs_path / "beta_distribution_by_region_crops.png", dpi=300, bbox_inches="tight")

plt.show()
