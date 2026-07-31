import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

dat_path = Path("calculated_parameters") / "intensification_factors_simple.csv"
figs_path = Path("..", "..", "figs")
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
        "maize",
        "sorghum",
        "soybeans",
        "barley",
        ]

iexcl = [
        "buckwheat", "green corn"
         ]

afilt = [
        # "United Kingdom", "France", "Brazil", "China", "India", "United States"
        "World",
        # "United Kingdom",
        # "Brazil",
        # "India",
        # "China, mainland",
        "Europe",
        "Africa", 
        "South America",
        "North America",
        "Asia",
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
item_colors = {item: palette[i % len(palette)] for i, item in enumerate(items)}

fig, ax = plt.subplots(figsize=(10, 6))

for _ in [0, 1]:
    ax.axhline(y=_, color="k", linestyle="--", linewidth=0.5, alpha = 0.5)

labels = []
area_labels = []
plotted_items = set()
for i, (idx, row) in enumerate(dfx.iterrows()):

    item = row.Item
    beta_vals = row.beta
    beta_sem = row.beta_std_err

    area_label = row.ISO3 if pd.notnull(row.ISO3) else row.Area

    label = f"{item} ({area_label})"
    labels.append(label)
    if area_label not in area_labels:
        area_labels.append(area_label)

    ax.errorbar(
        x=i,
        y=beta_vals,
        yerr=beta_sem,
        fmt="o",
        alpha=0.7,
        color=item_colors[item],
        label=item if item not in plotted_items else None,
    )
    plotted_items.add(item)

ax.set_xticks(np.arange(len(labels)), labels = labels, rotation = 90, ha = "right")


ax.set_ylabel(f"Extensification parameter $\\beta$ (FULL REGR)")
ax.legend(title="Crop")

fig.tight_layout()
plt.show()


fig.savefig(figs_path / f"intensification_factors_simple_{"_".join(area_labels)}.png", dpi=300, bbox_inches="tight")