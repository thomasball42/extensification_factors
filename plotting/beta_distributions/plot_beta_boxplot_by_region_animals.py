"""
Distribution of the current (most recent year) extensification parameter
beta across countries, grouped by continent -- box-and-whisker variant of
plot_beta_distribution_by_region_animals.py (same data prep, each scatter
point there becomes part of a box here).

beta_animals.csv has one row per Area already (single item, "All
pasture-based animal products"), so unlike the crops boxplot there's no
cross-item aggregation and no by-crop colour option -- just one box per
continent.

Groups with fewer than MIN_N_FOR_BOX countries (e.g. Northern America, often
just Canada + the US) are drawn as raw points instead of a box: with n=2,
matplotlib's linear-interpolated Q1/Q3 fall strictly between the two values,
so the box would cover a stretch with no data behind it while the whiskers
reach out to the two real points -- misleading rather than just uninformative.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "map"))
from _geo import load_country_codes, load_world
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from _utils import label_helper

dat_path = Path("outputs") / "beta_animals.csv"
figs_path = Path("..") / "figs" / "beta_distributions" / "beta_distribution_by_region_animals"
SAVE = True

os.makedirs(figs_path, exist_ok=True)

BOX_COLOR = "#2a78d6"  # blue

# ---------------------------------------------------------------------------
# Load + prep
# ---------------------------------------------------------------------------

df = pd.read_csv(dat_path)

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

world = load_world()
country_stats = df[["ISO3", "beta_current", "se_current"]].dropna(subset=["beta_current"]).merge(
    world[["ISO3", "CONTINENT", "NAME"]], on="ISO3", how="inner"
).rename(columns={"beta_current": "beta", "se_current": "se"})

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

continents = (
    country_stats.groupby("CONTINENT")["beta"].median().sort_values().index.tolist()
)
continent_n = country_stats["CONTINENT"].value_counts()

x_of_continent = {c: i for i, c in enumerate(continents)}

fig, ax = plt.subplots(figsize=(1.4 * len(continents), 5))

for _ in [0, 1]:
    ax.axhline(y=_, color="k", linestyle="--", linewidth=0.5, alpha=0.5)


# below this many countries, quartiles/whiskers are just interpolation
# artifacts with no data behind them -- plot the raw points instead
MIN_N_FOR_BOX = 3
_jitter_rng = np.random.default_rng(0)


def _draw_group(values, position, width, color):
    if len(values) < MIN_N_FOR_BOX:
        jitter = _jitter_rng.uniform(-width * 0.25, width * 0.25, size=len(values)) if len(values) > 1 else [0.0]
        ax.plot(
            position + np.asarray(jitter), values, "o", markersize=6, markerfacecolor=color,
            markeredgecolor="black", markeredgewidth=0.75, alpha=0.85, linestyle="none",
        )
        return None
    return ax.boxplot(
        values, positions=[position], widths=width, patch_artist=True,
        showfliers=True, whis=1.5,
        flierprops=dict(marker="o", markersize=3, markerfacecolor=color,
                         markeredgecolor="none", alpha=0.6),
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor=color, edgecolor="black", alpha=0.75, linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
    )


for c in continents:
    mask = country_stats["CONTINENT"].values == c
    values = country_stats.loc[mask, "beta"].dropna().values
    if len(values) == 0:
        continue
    _draw_group(values, x_of_continent[c], width=0.5, color=BOX_COLOR)

ax.set_xticks(range(len(continents)))
ax.set_xticklabels([f"{label_helper(c)}\n(n={continent_n[c]})" for c in continents], rotation=35)
ax.set_xlim(-0.6, len(continents) - 0.4)

# y-limits from the point estimates only so a handful of very uncertain
# series don't compress the rest of the plot into a thin band
lo, hi = np.nanpercentile(country_stats["beta"], [1, 99])
pad = 0.15 * (hi - lo)
ax.set_ylim(lo - pad, hi + pad)

ax.set_ylabel(r"Extensification parameter $\beta$" +  "\n" + r"(prior'd TVP/Kalman, current year)")
# ax.set_title("Distribution of country-level animal extensification betas, by continent")

fig.tight_layout()

if SAVE:
    fig.savefig(figs_path / "beta_boxplot_by_region_animals.png", dpi=300, bbox_inches="tight")

plt.show()
