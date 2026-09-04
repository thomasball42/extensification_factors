import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from _utils import filter_list as _filter_list, load_plot_config, resolve_filter_preset

SCRIPT_NAME = "plot_beta_timeseries_animals"

dat_path = Path("outputs") / "beta_animals.csv"
figs_path = Path("..") / "figs" / "other" / "beta_timeseries_animals"


SAVE = True
# filtering (only afilt/aexcl apply -- beta_animals.csv has a single pooled
# Item, so any ifilt/iexcl in the active preset is ignored here)
_plot_cfg = load_plot_config()
_, _, afilt, aexcl = resolve_filter_preset(_plot_cfg, SCRIPT_NAME)

# main
os.makedirs(figs_path, exist_ok=True)

country_codes = pd.read_excel(Path("data") / "inputs" / "nocsDataExport_20251021-164754.xlsx")

df = pd.read_csv(dat_path)

df = df.merge(country_codes[["FAOSTAT", "ISO3"]],
              left_on="Area Code", right_on="FAOSTAT",
              how="left",
              )
df = df.drop(columns=["FAOSTAT"])

# note: unlike the crop pipeline, beta_animals.csv has a single "Item"
# ("All pasture-based animal products") per Area, so there's no item
# dimension to filter/color by here -- only Area.
all_areas = df.Area.unique()



areas = _filter_list(all_areas, afilt, aexcl)

print(areas)

dfx = df[df.Area.isin(areas)].copy()
dfx = dfx.sort_values("Area").reset_index(drop=True)

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


color_map = {v: c for v, c in zip(areas, distinct_colors(len(areas)))}

fig, ax = plt.subplots(figsize=(10, 6))

for _ in [0, 1]:
    ax.axhline(y=_, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

area_labels = []
for i, (idx, row) in enumerate(dfx.iterrows()):

    area = row.Area

    beta_vals = row[row.index.str.startswith("beta_")].values.astype(float)
    beta_years = [int(c.split("_")[-1]) for c in row.index if c.startswith("beta_")]
    se_vals = row[[f"se_{y}" for y in beta_years]].values.astype(float)

    area_label = row.ISO3 if pd.notnull(row.ISO3) else row.Area
    line_color = color_map[area]

    ax.fill_between(
        beta_years, beta_vals - se_vals, beta_vals + se_vals,
        color=line_color, alpha=0.15, linewidth=0, zorder=1,
    )

    ax.plot(
        beta_years, beta_vals,
        label=area_label, alpha=0.7,
        color=line_color,
        zorder=2,
    )

    if area_label not in area_labels:
        area_labels.append(area_label)

ax.set_xlabel("Year")
ax.set_ylabel("Extensification parameter $\\beta$ (pasture-based animal products, kalman estimated)")
ax.legend(title="Area", loc="upper left")

fig.tight_layout()
plt.show()

if SAVE:
    filename = f"extensification_factors_tvp_animals_{'_'.join(area_labels)}.png"
    fig.savefig(figs_path / filename, dpi=300, bbox_inches="tight")
