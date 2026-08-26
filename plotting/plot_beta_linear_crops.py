"""
Plots the static (full-sample OLS) beta values from
outputs/beta_crops_linear.csv -- the linear counterpart to the
Kalman/TVP beta timeseries plotted by beta_timeseries_crops.py.

One point (+/- beta_se) per (Item, Area), grouped into clusters by crop
along the x-axis with areas distinguished by color, using the same
crop/area filters as beta_timeseries_crops.py.
"""

import colorsys
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _utils import filter_list as _filter_list

dat_path = Path("outputs") / "beta_crops_linear.csv"
figs_path = Path("..") / "figs" / "beta_linear_crops"
SAVE = True

# filtering (kept in sync with beta_timeseries_crops.py)
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

afilt = [
        "United Kingdom of Great Britain and Northern Ireland", 
        "France", 
        # "Brazil", "China", "India", "United States"
        # "World",
        # "United Kingdom",
        # "Brazil",
        # "Argentina",
        # "India",
        "China, mainland",
        # "Europe",
        # "Africa",
        # # "Latin America"
        # "South America",
        # "Northern America",
        # "Eastern Asia",
        # "Southern Asia",
        # "United States of America",
        "Japan"
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

# main

country_codes = pd.read_excel(Path("data") / "inputs" / "nocsDataExport_20251021-164754.xlsx")

os.makedirs(figs_path, exist_ok=True)

df = pd.read_csv(dat_path)

df = df.merge(country_codes[["FAOSTAT", "ISO3"]],
              left_on="Area Code", right_on="FAOSTAT",
              how="left",
              )
df = df.drop(columns=["FAOSTAT"])


items = _filter_list(df.Item.unique(), ifilt, iexcl)
areas = _filter_list(df.Area.unique(), afilt, aexcl)

print(items)
print(areas)

dfx = df[df.Item.isin(items) & df.Area.isin(areas)].copy()

dfx["Item"] = pd.Categorical(dfx["Item"], categories=items, ordered=True)
dfx["Area"] = pd.Categorical(dfx["Area"], categories=areas, ordered=True)
dfx = dfx.sort_values(["Item", "Area"]).reset_index(drop=True)

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


area_color = {a: c for a, c in zip(areas, distinct_colors(len(areas)))}

# cluster layout: one x-slot per Item, areas jittered within the slot
n_areas = len(areas)
width = 0.8
offsets = (
    [0.0] if n_areas == 1
    else [width * (i / (n_areas - 1) - 0.5) for i in range(n_areas)]
)
area_offset = dict(zip(areas, offsets))

fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(items) + 2), 6))

for _ in [0, 1]:
    ax.axhline(y=_, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

for area in areas:
    sub = dfx[dfx["Area"] == area]
    if sub.empty:
        continue
    x = [items.index(it) + area_offset[area] for it in sub["Item"]]
    ax.errorbar(
        x, sub["beta"], yerr=sub["beta_se"],
        fmt="o", markersize=6, capsize=3, linewidth=1.25,
        color=area_color[area], label=area,
    )

ax.set_xticks(range(len(items)))
ax.set_xticklabels(items, rotation=20, ha="right")
ax.set_xlim(-0.5, len(items) - 0.5)
ax.set_ylabel(r"Extensification parameter $\beta$ (static OLS, full sample)")
ax.legend(title="Area", loc="best", framealpha=0.9)

fig.tight_layout()
plt.show()

if SAVE:
    filename = f"beta_linear{f'_{items[0]}' if len(items) == 1 else '_multi'}_{'_'.join(areas)}.png"
    fig.savefig(figs_path / filename, dpi=300, bbox_inches="tight")
