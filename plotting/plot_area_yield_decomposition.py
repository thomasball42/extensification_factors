"""
Year-over-year decomposition of production growth into its area (extensive)
and yield (intensive) margins, for a single item across one or more areas.

Since Production = Area harvested * Yield, working in log-space makes the
decomposition exactly additive:

    Dln(Production) = Dln(Area harvested) + Dln(Yield)

Each year is plotted as a point at (%D Area harvested, %D Yield). The y = x
line marks equal contribution: points above it are yield-driven that year,
points below it are area-driven. Point size encodes |%D Production| so
big-swing years stand out; point color encodes year.

`afilt`/`aexcl` may resolve to more than one Area -- each matched area gets
its own subplot (shared year color scale and point-size legend) in a single
figure.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _utils import filter_list as _filter_list

DATA_PATH = Path("data") / "inputs"
figs_path = Path("..") /  "figs" / "area_yield_decomp"

SAVE = True

# main

os.makedirs(figs_path, exist_ok=True)

elements = ["Area harvested", "Production", "Yield"]
columns = ["Area", "Area Code", "Item", "Item Code", "Element", "Year", "Value", "Unit"]

df = pd.read_csv(
    DATA_PATH / "Production_Crops_Livestock_E_All_Data_(Normalized).csv",
    encoding="latin-1",
    low_memory=False,
    usecols=columns,
)

ha_unit = df.loc[df.Element == "Area harvested", "Unit"].values[0]
prod_unit = df.loc[df.Element == "Production", "Unit"].values[0]
yield_unit = df.loc[df.Element == "Yield", "Unit"].values[0]

df = df.drop(columns=["Unit"])
df = df[df.Element.isin(elements)]

# restrict to items that actually report "Area harvested" (i.e. crops, not
# livestock/animal-product items that reuse the "Production" element label)
crop_items = df.loc[df.Element == "Area harvested", "Item"].unique()
df = df[df.Item.isin(crop_items)]

all_items = df.Item.unique()
all_areas = df.Area.unique()

ifilt = [
        "Wheat"
        ]

iexcl = [
        "buckwheat",
        "n.e.c."
        ]

# afilt = [
#         "United States of America",
#         ]

afilt = [
        "United Kingdom of Great Britain and Northern Ireland", 
        "France", 
        # "Brazil", "China", "India", "United States"
        # "World",
        # "United Kingdom",
        # "Brazil",
        # "Argentina",
        "India",
        "China, mainland",
        # "Europe",
        # "Africa",
        # # "Latin America"
        # "South America",
        # "Northern America",
        # "Eastern Asia",
        # "Southern Asia",
        # "United States of America",
        ]

aexcl = [
        "taiwan",
        "southern",
        "northern",
        "western",
        "eastern",
        "republic",
        "union",
        "middle",
        "South Africa",
        "central"
        ]


item_matches = _filter_list(all_items, ifilt, iexcl)
assert len(item_matches) == 1, (
    f"ifilt/iexcl must resolve to exactly one Item, got {item_matches}"
)
item = item_matches[0]

area_matches = _filter_list(all_areas, afilt, aexcl)
assert len(area_matches) >= 1, (
    f"afilt/aexcl must resolve to at least one Area, got {area_matches}"
)
areas = area_matches


def compute_decomp(item, area):
    g = df[(df.Item == item) & (df.Area == area)].pivot_table(
        index="Year", columns="Element", values="Value"
    ).sort_index()

    full_years = np.arange(g.index.min(), g.index.max() + 1)
    g = g.reindex(full_years)

    with np.errstate(divide="ignore"):
        log_area = np.log(g["Area harvested"].values)
        log_yield = np.log(g["Yield"].values)
        log_prod = np.log(g["Production"].values)

    d_area = 100 * np.diff(log_area)
    d_yield = 100 * np.diff(log_yield)
    d_prod = 100 * np.diff(log_prod)
    years = full_years[1:]

    valid = np.isfinite(d_area) & np.isfinite(d_yield) & np.isfinite(d_prod)
    d_area, d_yield, d_prod, years = (
        d_area[valid], d_yield[valid], d_prod[valid], years[valid]
    )

    # holds exactly in theory (Production = Area harvested * Yield), but
    # FAOSTAT rounds Area/Yield/Production independently, so allow slack
    residual = np.abs((d_area + d_yield) - d_prod)
    assert residual.max() < 0.5, (
        f"{area}: log-diff decomposition should be ~additive, "
        f"max residual = {residual.max():.3f} pct points"
    )

    return d_area, d_yield, d_prod, years


decomps = {area: compute_decomp(item, area) for area in areas}

all_years = np.concatenate([years for *_, years in decomps.values()])
norm = plt.Normalize(vmin=all_years.min(), vmax=all_years.max())
cmap = plt.get_cmap("viridis")
size_k = 8


def plot_panel(ax, area, d_area, d_yield, d_prod, years):
    lo = min(d_area.min(), d_yield.min())
    hi = max(d_area.max(), d_yield.max())
    pad = 0.1 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    ax.axhline(0, color="0.7", linestyle="--", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.7", linestyle="--", linewidth=0.8, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="0.4", linewidth=1, zorder=1)
    ax.annotate("equal contribution", (hi, hi), fontsize=8, color="0.4",
                ha="right", va="bottom", rotation=45, rotation_mode="anchor")

    step = 10
    c_min = (d_area + d_yield).min()
    c_max = (d_area + d_yield).max()
    c_levels = np.arange(np.floor(c_min / step) * step, np.ceil(c_max / step) * step + step, step)
    for c in c_levels:
        if c == 0:
            continue
        ax.plot([lo, hi], [c - lo, c - hi], color="0.75", linestyle=":", linewidth=0.7, zorder=0)
        ax.annotate(f"{c:+.0f}%", (hi, c - hi), fontsize=7, color="0.6",
                    ha="right", va="bottom")

    sizes = 20 + size_k * np.abs(d_prod)
    sc = ax.scatter(d_area, d_yield, c=years, cmap=cmap, norm=norm, s=sizes,
                     edgecolor="0.3", linewidth=0.3, zorder=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    ax.set_xlabel("Year-over-year %$\\Delta$ Area harvested (log-diff)")
    ax.set_ylabel("Year-over-year %$\\Delta$ Yield (log-diff)")
    ax.set_title(area)

    return sc


ncols = int(np.ceil(np.sqrt(len(areas))))
nrows = int(np.ceil(len(areas) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 8 * nrows), squeeze=False)
axes_flat = axes.flatten()

sc = None
for ax, area in zip(axes_flat, areas):
    d_area, d_yield, d_prod, years = decomps[area]
    sc = plot_panel(ax, area, d_area, d_yield, d_prod, years)

for ax in axes_flat[len(areas):]:
    ax.set_visible(False)

fig.suptitle(f"{item}: area vs yield contribution to production growth")

cbar = fig.colorbar(sc, ax=axes_flat[:len(areas)], shrink=0.8)
cbar.set_label("Year")

size_ref_vals = [5, 15, 30]
size_handles = [
    axes_flat[0].scatter([], [], s=20 + size_k * v, color="0.5", edgecolor="0.3", linewidth=0.3)
    for v in size_ref_vals
]
fig.legend(size_handles, [f"{v}% ΔProduction" for v in size_ref_vals],
           title="|%$\\Delta$ Production|", loc="upper left", framealpha=0.9)

plt.show()

if SAVE:
    area_tag = areas[0] if len(areas) == 1 else f"{len(areas)}areas"
    filename = f"area_yield_decomposition_{item}_{area_tag}.png"
    fig.savefig(figs_path / filename, dpi=300, bbox_inches="tight")
