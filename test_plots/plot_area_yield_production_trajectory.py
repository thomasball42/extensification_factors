"""
Phase/trajectory plot of a single commodity's Area harvested vs Yield and
Area harvested vs Production, one point per year, connected in year order.

Since Production = Area harvested * Yield, this makes it visually obvious
whether a region's production growth over time has come from extensification
(moving right = more area) or intensification (moving up = higher yield).
Multiple areas can be overlaid on the same panel; each panel is scoped to a
single commodity (set via ifilt/iexcl, must resolve to exactly one Item).
"""

import colorsys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pathlib import Path

DATA_PATH = Path("data") / "inputs"
figs_path = Path("..", "figs")

SAVE = True

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
        "Cereal"
        ]

iexcl = [
        "buckwheat",
        "n.e.c."
        ]

afilt = [
        "Africa",
        "Europe",
        "Northern America",
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


def _filter_list(all_, filt_, excl_):
    if not filt_:
        return all_
    filt = [f.lower() for f in filt_]
    excl = [f.lower() for f in excl_]
    return [
        i for i in all_
        if any(f in i.lower() for f in filt)
        and (i.lower() in filt or not any(f in i.lower() for f in excl))
    ]


item_matches = _filter_list(all_items, ifilt, iexcl)
assert len(item_matches) == 1, (
    f"ifilt/iexcl must resolve to exactly one Item, got {item_matches}"
)
item = item_matches[0]

areas = _filter_list(all_areas, afilt, aexcl)
assert areas, f"afilt/aexcl matched no Area, candidates were {list(all_areas)[:20]}..."

wide = df[(df.Item == item) & (df.Area.isin(areas))].pivot_table(
    index=["Area", "Year"], columns="Element", values="Value"
).reset_index()
wide = wide.sort_values(["Area", "Year"]).dropna(subset=elements)

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


def distinct_colors(n):
    """n visually distinct colors: the hand-picked base palette for small n,
    golden-angle-spaced HSV hues (with alternating sat/val) beyond that so
    consecutive categories never land on near-identical hues."""
    if n <= len(BASE_PALETTE):
        return BASE_PALETTE[:n]
    colors = list(BASE_PALETTE)
    for i in range(len(BASE_PALETTE), n):
        hue = (i * 0.6180339887) % 1.0
        sat = 0.55 if (i // len(BASE_PALETTE)) % 2 else 0.85
        val = 0.65 if (i // len(BASE_PALETTE)) % 2 else 0.9
        colors.append(mcolors.to_hex(colorsys.hsv_to_rgb(hue, sat, val)))
    return colors


def _light_tint(hex_color, frac=0.75):
    """Blend hex_color toward white by frac (0=color, 1=white) — a visible
    light tint rather than pure white, so the earliest (lightest) points
    don't disappear against a white axes background."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


base_colors = dict(zip(areas, distinct_colors(len(areas))))
area_cmaps = {
    a: mcolors.LinearSegmentedColormap.from_list(f"{a}_cmap", [_light_tint(c), c])
    for a, c in base_colors.items()
}

norm = mcolors.Normalize(vmin=wide.Year.min(), vmax=wide.Year.max())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for area in areas:
    g = wide[wide.Area == area]
    if g.empty:
        continue

    x = g["Area harvested"].values / g["Area harvested"].values[0]
    y_yield = g["Yield"].values / g["Yield"].values[0]  
    y_prod = g["Production"].values / g["Production"].values[0]
    years = g["Year"].values
    cmap = area_cmaps[area]
    base_color = base_colors[area]

    for ax, y in [(ax1, y_yield), (ax2, y_prod)]:
        ax.plot(x, y, color=base_color, alpha=0.3, lw=1, zorder=1)
        ax.scatter(x, y, c=years, cmap=cmap, norm=norm, s=30, zorder=2, edgecolor="none")
        ax.annotate(str(years[0]), (x[0], y[0]), fontsize=7, color=base_color,
                    xytext=(4, 4), textcoords="offset points")
        ax.annotate(str(years[-1]), (x[-1], y[-1]), fontsize=7, color=base_color,
                    xytext=(4, 4), textcoords="offset points")

ax1.set_xlabel(f"Area harvested ({ha_unit})")
ax1.set_ylabel(f"Yield ({yield_unit})")
ax2.set_xlabel(f"Area harvested ({ha_unit})")
ax2.set_ylabel(f"Production ({prod_unit})")

legend_handles = [Line2D([0], [0], color=c, lw=2) for c in base_colors.values()]
fig.legend(legend_handles, base_colors.keys(), title="Area", loc="upper center",
           ncol=min(len(areas), 6), bbox_to_anchor=(0.5, 1.0))

fig.suptitle(item, y=1.08)
fig.tight_layout()
plt.show()

if SAVE:
    filename = f"area_yield_production_trajectory_{item}_{'_'.join(areas)}.png"
    fig.savefig(figs_path / filename, dpi=300, bbox_inches="tight")
