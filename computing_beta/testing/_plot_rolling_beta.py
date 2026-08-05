import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

dat_path = Path("calculated_parameters") / "intensification_factors_rolling.csv"
figs_path = Path("..", "..", "figs")

country_codes = pd.read_excel(Path(".", "mrio_pipeline", "input_data", "nocsDataExport_20251021-164754.xlsx"))

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
        # "rice", 
        # "wheat", 
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
        # "World",
        # "United Kingdom",
        "Brazil",
        "Argentina",
        "India",
        # "India",
        "China, mainland",
        # "Europe",
        # "Africa", 
        # "South America",
        # "North America",
        # "Asia",
        "United States of America",
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

palette = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]

# color encodes area, linestyle encodes crop (there are usually many more
# areas than crops, and same-crop lines were indistinguishable when color
# only tracked crop)
area_colors = {area: palette[i % len(palette)] for i, area in enumerate(areas)}
item_linestyles = {item: linestyles[i % len(linestyles)] for i, item in enumerate(items)}

fig, ax = plt.subplots(figsize=(10, 6))

for _ in [0, 1]:
    ax.axhline(y=_, color="k", linestyle="--", linewidth=0.5, alpha = 0.5)

labels = []
area_labels = []
for i, (idx, row) in enumerate(dfx.iterrows()):

    item = row.Item
    area = row.Area

    beta_vals = row[row.index.str.startswith("beta_")].values
    beta_years = [int(c.split("_")[-1]) for c in row.index if c.startswith("beta_")]

    area_label = row.ISO3 if pd.notnull(row.ISO3) else row.Area

    ax.plot(
        beta_years, beta_vals,
        label=f"{item} ({area_label})", alpha=0.7,
        color=area_colors[area], linestyle=item_linestyles[item],
    )

    if area_label not in area_labels:
        area_labels.append(area_label)

ax.set_xlabel("Year")
ax.set_ylabel(f"Extensification parameter $\\beta$ (5-year window)")

from matplotlib.lines import Line2D
area_handles = [Line2D([0], [0], color=area_colors[a], lw=2) for a in areas]
item_handles = [Line2D([0], [0], color="k", linestyle=item_linestyles[it], lw=2) for it in items]
area_legend = ax.legend(area_handles, areas, title="Area", loc="upper left")
ax.add_artist(area_legend)

if len(items) > 0:
    ax.legend(item_handles, items, title="Crop", loc="upper right")

fig.tight_layout()
plt.show()

save_path = figs_path / f"intensification_factors_rolling_{'_'.join(area_labels)}.png"
fig.savefig(save_path, dpi=300, bbox_inches="tight")