import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH: Path = Path("data") / "inputs" 
OUT_PATH: Path = Path("outputs") / "validation" / "p_a_indentity.csv" 

TOLERANCE = 0.01  # 1% relative discrepancy flagged for review

elements = ["Area harvested", "Production", "Yield"]
columns = ["Area", "Item", "Item Code", "Element", "Year", "Value", "Unit"]

df = pd.read_csv(
    DATA_PATH / "Production_Crops_Livestock_E_All_Data_(Normalized).csv",
    encoding="latin-1",
    low_memory=False,
    usecols=columns,
)

ha_unit = df.loc[df.Element == "Area harvested", "Unit"].values[0]
prod_unit = df.loc[df.Element == "Production", "Unit"].values[0]
yield_unit = df.loc[df.Element == "Yield", "Unit"].values[0]
print(f"Units found in file -- Area: {ha_unit}, Production: {prod_unit}, Yield: {yield_unit}")
print("Dividing yield by 1000 to get tonnes")

df = df[df.Element.isin(elements)]
crop_items = df.loc[df.Element == "Area harvested", "Item"].unique()
df = df[df.Item.isin(crop_items)]

wide = df.pivot_table(
    index=["Area", "Item", "Item Code", "Year"], columns="Element", values="Value"
).reset_index().dropna(subset=elements)

# kg/ha -> t/ha
implied_production = wide["Area harvested"] * wide["Yield"] / 1000

wide["implied_production"] = implied_production
wide["relative_discrepancy"] = (
    (wide["implied_production"] - wide["Production"]) / wide["Production"]
)

abs_disc = wide["relative_discrepancy"].abs()

print("=== Identity check: Production vs. Area*Yield (raw FAOSTAT data) ===")
print(f"Rows checked: {len(wide):,}")
print(f"Median |relative discrepancy|:  {abs_disc.median():.4%}")
print(f"95th percentile:                {abs_disc.quantile(0.95):.4%}")
print(f"99th percentile:                {abs_disc.quantile(0.99):.4%}")
print(f"Max:                            {abs_disc.max():.4%}")
print(f"Rows exceeding {TOLERANCE:.0%} tolerance: "
      f"{(abs_disc > TOLERANCE).sum():,} ({(abs_disc > TOLERANCE).mean():.2%})")

flagged = wide.loc[abs_disc > TOLERANCE].sort_values("relative_discrepancy", key=abs, ascending=False)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
flagged.to_csv(OUT_PATH, index=False)
print(f"\nFlagged rows (> {TOLERANCE:.0%} discrepancy) written to {OUT_PATH}")

# Histogram data for a Technical Validation figure -- distribution of
# |relative discrepancy| across all rows (expect a sharp spike near 0)
finite_abs_disc = abs_disc[np.isfinite(abs_disc)]
n_non_finite = len(abs_disc) - len(finite_abs_disc)
if n_non_finite:
    print(f"Excluding {n_non_finite:,} rows with non-finite discrepancy "
          f"(Production == 0) from histogram")
hist_counts, hist_edges = np.histogram(np.clip(finite_abs_disc, 0, 0.05), bins=50)
hist_out = pd.DataFrame({"bin_left": hist_edges[:-1], "bin_right": hist_edges[1:], "count": hist_counts})
hist_out.to_csv(OUT_PATH.with_name("p_a_identity_histogram.csv"), index=False)
print(f"Histogram data written to {OUT_PATH.with_name('p_a_identity_histogram.csv')}")