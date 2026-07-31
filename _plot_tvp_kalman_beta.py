import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

dat_path = Path("calculated_parameters") / "intensification_factors_tvp.csv"
figs_path = Path("..", "..", "figs")

SAVE = True

country_codes = pd.read_excel(Path("..", "mrio_pipeline", "input_data", "nocsDataExport_20251021-164754.xlsx"))

print(country_codes)

df = pd.read_csv(dat_path)

df = df.merge(country_codes[["FAOSTAT", "ISO3"]], 
              left_on="Area Code", right_on="FAOSTAT", 
              how="left",
              )
df = df.drop(columns=["FAOSTAT"])


all_items = df.Item.unique()
all_areas = df.Area.unique()

ifilt = [
        "rice", 
        "wheat", 
        # "maize",
        # "sorghum",
        "soy",
        # "barley",
        ]

iexcl = [
        "buckwheat", "green corn"
         ]

afilt = [
        # "United Kingdom", "France", "Brazil", "China", "India", "United States"
        "World",
        # "United Kingdom",
        # "Brazil",
        # "Argentina",
        # "India",
        # "China, mainland",
        # "Europe",
        # "Africa", 
        # "South America",
        # "North America",
        # "Asia",
        # "United States of America",
        ]
 
aexcl = [
        # "taiwan",
        # "southern",
        # "northern",
        # "western", 
        # "eastern",
        # "republic",
        # # "union",
        # "middle",
        # "South Africa",
        # "central"
        ]

def _filter_list(all_, filt_, excl_):
    if not filt_:
        return all_
    filt = [f.lower() for f in filt_]
    excl = [f.lower() for f in excl_]
    return [
        i for i in all_
        if any(f in i.lower() for f in filt)
        and not any(f in i.lower() for f in excl)
    ]

items = _filter_list(all_items, ifilt, iexcl)
areas = _filter_list(all_areas, afilt, aexcl)

print(items)
print(areas)

dfx = df[df.Item.isin(items) & df.Area.isin(areas)].copy()

# group by crop type (in the order given by `items`), areas grouped together within each crop
dfx["Item"] = pd.Categorical(dfx["Item"], categories=items, ordered=True)
dfx = dfx.sort_values(["Item", "Area"]).reset_index(drop=True)

import colorsys
import matplotlib.colors as mcolors

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
LINESTYLES = [
    "-", "--", ":", "-.",
    (0, (3, 1, 1, 1)),        # densely dashdotted
    (0, (5, 1)),               # densely dashed
    (0, (1, 1)),               # densely dotted
    (0, (3, 5, 1, 5, 1, 5)),   # dashdotdotted
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


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


# Color gets the dimension with more categories (scales to any count via
# distinct_colors), linestyle gets the dimension with fewer (only a handful
# of dash patterns read as distinct at a glance). If the linestyle dimension
# still outgrows LINESTYLES, marker shape is layered on to disambiguate the
# repeats instead of letting two lines become pixel-identical.
if len(items) <= len(areas):
    color_key, color_values = "Area", areas
    ls_key, ls_values = "Item", items
else:
    color_key, color_values = "Item", items
    ls_key, ls_values = "Area", areas

color_map = {v: c for v, c in zip(color_values, distinct_colors(len(color_values)))}
linestyle_map = {v: LINESTYLES[i % len(LINESTYLES)] for i, v in enumerate(ls_values)}
marker_map = {
    v: (MARKERS[(i // len(LINESTYLES) - 1) % len(MARKERS)] if i >= len(LINESTYLES) else None)
    for i, v in enumerate(ls_values)
}

fig, ax = plt.subplots(figsize=(10, 6))

for _ in [0, 1]:
    ax.axhline(y=_, color="k", linestyle="--", linewidth=0.5, alpha = 0.5)

labels = []
area_labels = []
for i, (idx, row) in enumerate(dfx.iterrows()):

    item = row.Item
    area = row.Area

    beta_vals = row[row.index.str.startswith("beta_")].values.astype(float)
    beta_years = [int(c.split("_")[-1]) for c in row.index if c.startswith("beta_")]
    se_vals = row[[f"se_{y}" for y in beta_years]].values.astype(float)

    area_label = row.ISO3 if pd.notnull(row.ISO3) else row.Area

    color_val = area if color_key == "Area" else item
    ls_val = area if ls_key == "Area" else item
    marker = marker_map[ls_val]
    line_color = color_map[color_val]

    ax.fill_between(
        beta_years, beta_vals - se_vals, beta_vals + se_vals,
        color=line_color, alpha=0.15, linewidth=0, zorder=1,
    )

    ax.plot(
        beta_years, beta_vals,
        label=f"{item} ({area_label})", alpha=0.7,
        color=line_color, linestyle=linestyle_map[ls_val],
        marker=marker, markevery=max(1, len(beta_years) // 6) if marker else None,
        markersize=5, markerfacecolor="none" if marker else None,
        zorder=2,
    )

    if area_label not in area_labels:
        area_labels.append(area_label)

ax.set_xlabel("Year")


ax.set_ylabel(f"Extensification parameter $\\beta$ (kalman estimated)")

from matplotlib.lines import Line2D
color_handles = [Line2D([0], [0], color=color_map[v], lw=2) for v in color_values]
ls_handles = [
    Line2D([0], [0], color="k", linestyle=linestyle_map[v], marker=marker_map[v],
           markerfacecolor="none" if marker_map[v] else None, lw=2)
    for v in ls_values
]
color_legend = ax.legend(color_handles, color_values, title=color_key, loc="upper left")
ax.add_artist(color_legend)

if len(ls_values) > 0:
    ax.legend(ls_handles, ls_values, title=ls_key, loc="upper right")

fig.tight_layout()
plt.show()

if SAVE:
    filename = f"intensification_factors_tvp{f"_{items[0]}" if len(items) == 1 else "multi"}_{'_'.join(area_labels)}.png"
    fig.savefig(figs_path / filename, dpi=300, bbox_inches="tight")