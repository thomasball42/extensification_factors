"""
Distribution of the current (most recent year) extensification parameter
beta across countries, grouped by continent -- box-and-whisker variant of
plot_beta_distribution_by_region_crops.py (same data prep, each scatter
"blob" there becomes a box here).

One point per country: FAOSTAT aggregate items are excluded and, when
take_mean is True, each country's items are combined via a
production-weighted mean/SE (see _geo.weighted_mean_se), using each item's
own most recent (current_year) beta rather than a fixed calendar year, since
not every item's series necessarily runs through the same last year.
"""

import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "map"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from _geo import load_country_codes, load_world, weighted_mean_se
from _utils import filter_list as _filter_list, load_plot_config, resolve_filter_preset

SCRIPT_NAME = "plot_beta_boxplot_by_region_crops"

dat_path = Path("outputs") / "beta_crops.csv"
figs_path = Path("..") / "figs" / "beta_distributions" / "beta_distribution_by_region_crops"
SAVE = True

# True: production-weighted mean across the selected crops, one box per
# continent (as before). False: no aggregation across crops -- each selected
# crop gets its own box, colour-coded by crop, within each continent's slot.
_plot_cfg = load_plot_config()
ifilt, iexcl, _, _ = resolve_filter_preset(_plot_cfg, SCRIPT_NAME)
take_mean = _plot_cfg["scripts"][SCRIPT_NAME].get("take_mean", False)

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

if take_mean:
    # production-weighted aggregate across items, per country
    country_stats = df.groupby("ISO3").apply(
        lambda g: pd.Series(dict(zip(
            ["beta", "se"], weighted_mean_se(g["beta_current"], g["se_current"], g[prod_col])
        ))),
        include_groups=False,
    ).dropna(subset=["beta"])
else:
    # keep each crop's own point, no aggregation across items
    country_stats = (
        df[["ISO3", "Item", "beta_current", "se_current"]]
        .rename(columns={"beta_current": "beta", "se_current": "se"})
        .dropna(subset=["beta"])
        .set_index("ISO3")
    )

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
continent_n = country_stats.groupby("CONTINENT")["NAME"].nunique()

x_of_continent = {c: i for i, c in enumerate(continents)}

fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(continents) + 2), 7))

for _ in [0, 1]:
    ax.axhline(y=_, color="k", linestyle="--", linewidth=0.5, alpha=0.5)


def _draw_box(values, position, width, color):
    bp = ax.boxplot(
        values, positions=[position], widths=width, patch_artist=True,
        showfliers=True, whis=1.5,
        flierprops=dict(marker="o", markersize=3, markerfacecolor=color,
                         markeredgecolor="none", alpha=0.6),
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor=color, edgecolor="black", alpha=0.75, linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
    )
    return bp


if take_mean:
    for c in continents:
        mask = country_stats["CONTINENT"].values == c
        values = country_stats.loc[mask, "beta"].dropna().values
        if len(values) == 0:
            continue
        _draw_box(values, x_of_continent[c], width=0.5, color=continent_color[c])
else:
    # colour by crop instead of continent; each crop gets its own sub-band
    # within the continent's slot, so same-crop boxes line up across regions
    n_items = len(items)
    band_width = 0.6 / n_items
    crop_color = {crop: col for crop, col in zip(items, BASE_PALETTE)}
    for c in continents:
        for i, crop in enumerate(items):
            mask = (country_stats["CONTINENT"].values == c) & (country_stats["Item"].values == crop)
            values = country_stats.loc[mask, "beta"].dropna().values
            if len(values) == 0:
                continue
            band_center = -0.3 + (i + 0.5) * band_width
            _draw_box(values, x_of_continent[c] + band_center, width=band_width * 0.8, color=crop_color[crop])

    legend_handles = [mpatches.Patch(facecolor=crop_color[crop], edgecolor="black", alpha=0.75, label=crop)
                       for crop in items]
    ax.legend(handles=legend_handles, loc="best", fontsize=8, framealpha=0.9)

ax.set_xticks(range(len(continents)))
ax.set_xticklabels([f"{c}\n(n={continent_n[c]})" for c in continents])
ax.set_xlim(-0.6, len(continents) - 0.4)

# y-limits from the point estimates only so a handful of very uncertain
# series don't compress the rest of the plot into a thin band
lo, hi = np.nanpercentile(country_stats["beta"], [1, 99])
pad = 0.15 * (hi - lo)
ax.set_ylim(lo - pad, hi + pad)

if take_mean:
    ax.set_ylabel(r"Mean extensification parameter $\beta$ "
                  r"(production-weighted across crops)")
else:
    ax.set_ylabel(r"Extensification parameter $\beta$")
# ax.set_title("Distribution of country-level crop extensification betas, by continent")

fig.tight_layout()

if SAVE:
    suffix = "" if take_mean else "_by_crop"
    fig.savefig(figs_path / f"beta_boxplot_by_region_crops{suffix}.png", dpi=300, bbox_inches="tight")

plt.show()
