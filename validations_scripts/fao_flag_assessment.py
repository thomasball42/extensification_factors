import numpy as np
import pandas as pd
from pathlib import Path

TVP_OUTPUT_PATH: Path = Path("outputs") / "beta_crops.csv"
OUT_LONG_PATH: Path = Path("outputs") / "validation" / "fao_flag_assessment_long_format.csv"
OUT_SUMMARY_PATH: Path = Path("outputs") / "validation" / "fao_flag_assessment_summary.csv"
OUT_QHAT_PATH: Path = Path("outputs") / "validation" / "fao_flag_assessment_qhat.csv"

# A series-year is flagged "reliable" if its SE is below this absolute
# threshold. 
SE_RELIABILITY_THRESHOLD = 0.2

# A series is flagged "essentially constant" if Q_hat is this many orders of
# magnitude below R_hat (a practical stand-in for "MLE found no evidence of drift")
QR_RATIO_FLAT_THRESHOLD = 1e-3

df = pd.read_csv(TVP_OUTPUT_PATH)

id_cols = [c for c in df.columns if not (c.startswith("beta_") or c.startswith("se_"))]
beta_cols = sorted((c for c in df.columns if c.startswith("beta_")), key=lambda c: int(c.split("_")[1]))

long_records = []
for _, row in df.iterrows():
    for bc in beta_cols:
        yr = bc.split("_", 1)[1]
        se_c = f"se_{yr}"
        if se_c not in df.columns or pd.isna(row[bc]) or pd.isna(row.get(se_c, np.nan)):
            continue
        long_records.append({
            "Area": row["Area"], "Item": row["Item"], "Item Code": row["Item Code"],
            "year": int(yr), "beta": row[bc], "se": row[se_c],
        })

long_df = pd.DataFrame(long_records)
long_df["reliable"] = long_df["se"] < SE_RELIABILITY_THRESHOLD

OUT_LONG_PATH.parent.mkdir(parents=True, exist_ok=True)
long_df.to_csv(OUT_LONG_PATH, index=False)

# --- overall SE distribution ---
print("=== SE distribution across all series-years ===")
print(long_df["se"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
print(f"\nShare of series-years meeting SE < {SE_RELIABILITY_THRESHOLD} reliability threshold: "
      f"{long_df['reliable'].mean():.1%}\n")

# --- SE vs series length (years of data available per series) ---
n_years_per_series = long_df.groupby(["Area", "Item", "Item Code"])["year"].transform("count")
long_df["n_years_in_series"] = n_years_per_series
length_bins = [0, 10, 15, 20, 25, 30, 100]
length_labels = ["<=10", "11-15", "16-20", "21-25", "26-30", "30+"]
long_df["length_bucket"] = pd.cut(long_df["n_years_in_series"], bins=length_bins, labels=length_labels)

se_by_length = long_df.groupby("length_bucket", observed=True)["se"].agg(["mean", "median", "count"])
print("=== SE by series length ===")
print(se_by_length)

OUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
se_by_length.to_csv(OUT_SUMMARY_PATH)
print(f"\nWritten to {OUT_SUMMARY_PATH}")

# --- Q_hat / R_hat characterization, if present in the output file ---
if "Q_hat" in df.columns and "R_hat" in df.columns:
    qr = df[["Area", "Item", "Item Code", "Q_hat", "R_hat"]].dropna()
    qr["qr_ratio"] = qr["Q_hat"] / qr["R_hat"]
    qr["essentially_flat"] = qr["qr_ratio"] < QR_RATIO_FLAT_THRESHOLD

    print(f"\n=== Q_hat / R_hat characterization ({len(qr)} series) ===")
    print(f"Share of series classified as essentially flat (Q/R < {QR_RATIO_FLAT_THRESHOLD:.0e}): "
          f"{qr['essentially_flat'].mean():.1%}")
    print(qr["qr_ratio"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))

    OUT_QHAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    qr.to_csv(OUT_QHAT_PATH, index=False)
    print(f"\nWritten to {OUT_QHAT_PATH}")
else:
    print("\nQ_hat / R_hat columns not found in output file -- add them to the main "
          "pipeline's per-series record if you want this characterization "
          "(the current tvp_intensification_factors.py already computes them, "
          "just confirm they're being written to the output CSV).")

# --- histogram data for the SE distribution figure ---
hist_counts, hist_edges = np.histogram(np.clip(long_df["se"], 0, 2), bins=50)
hist_out = pd.DataFrame({"bin_left": hist_edges[:-1], "bin_right": hist_edges[1:], "count": hist_counts})
hist_out.to_csv(OUT_SUMMARY_PATH.with_name("validation_4_se_histogram.csv"), index=False)
print(f"SE histogram data written to {OUT_SUMMARY_PATH.with_name('validation_4_se_histogram.csv')}")