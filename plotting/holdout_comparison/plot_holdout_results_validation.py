"""
Summary plot for the TVP hold-out calibration check
(validation/holdout_results_validation.py).

For each qualifying series, (Q, R) are fit by MLE on the training years only
and the one-step-ahead predictions are scored on the held-out years. A
well-calibrated model should have standardised residuals (v_t / sqrt(F_t))
with mean ~0 and variance ~1 across series, alongside small plain-units
error (MAE, bias).

Reads:
  outputs/validation/holdout_validation.csv

Produces:
  holdout_results_validation_summary.png -- 2x2 grid: MAE, bias, standardised
    residual mean, and standardised residual variance, each as a histogram
    across series with the target/aggregate values marked.

Run with working directory = extensification_factors/.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("outputs") / "validation" / "holdout_validation.csv"
FIGS_PATH = Path("..") / "figs" / "holdout_comparison"
SAVE = True

BASE_PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e34948",  # red
]

os.makedirs(FIGS_PATH, exist_ok=True)

df = pd.read_csv(DATA_PATH)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 6))

# ---------------------------------------------------------------------------
# A. MAE (plain units, log points) -- always positive, right-skewed
# ---------------------------------------------------------------------------

mae_hi = df["mae"].quantile(0.99)
mae_clipped = df["mae"].clip(upper=mae_hi)
ax1.hist(mae_clipped, bins=60, color=BASE_PALETTE[0], alpha=0.85)
ax1.axvline(df["mae"].mean(), color=BASE_PALETTE[4], linestyle="-", linewidth=1.5,
            label=f"mean {df['mae'].mean():.4f}")
ax1.axvline(df["mae"].median(), color="0.3", linestyle="--", linewidth=1.5,
            label=f"median {df['mae'].median():.4f}")
ax1.set_xlabel("Held-out MAE (log points)" + (" [99th pct clipped]" if mae_hi < df["mae"].max() else ""))
ax1.set_ylabel("Number of series")
ax1.set_title("(a) One-step-ahead absolute error")
ax1.legend()

# ---------------------------------------------------------------------------
# B. Bias (plain units) -- signed, should be centred near 0
# ---------------------------------------------------------------------------

bias_lo, bias_hi = df["bias"].quantile([0.005, 0.995])
bias_clipped = df["bias"].clip(bias_lo, bias_hi)
ax2.hist(bias_clipped, bins=60, color=BASE_PALETTE[1], alpha=0.85)
# ax2.axvline(0, color="0.3", linestyle="--", linewidth=1.5, label="target 0")
ax2.axvline(df["bias"].mean(), color=BASE_PALETTE[4], linestyle="-", linewidth=1.5,
            label=f"mean {df['bias'].mean():.4f}")
ax2.set_xlabel("Held-out bias (log points) [0.5th-99.5th pct clipped]")
ax2.set_ylabel("Number of series")
ax2.set_title("(b) One-step-ahead bias")
ax2.legend()

# ---------------------------------------------------------------------------
# C. Standardized residual mean, v_t / sqrt(F_t) -- target ~0
# ---------------------------------------------------------------------------

srm_lo, srm_hi = df["std_resid_mean"].quantile([0.005, 0.995])
srm_clipped = df["std_resid_mean"].clip(srm_lo, srm_hi)
ax3.hist(srm_clipped, bins=60, color=BASE_PALETTE[2], alpha=0.85)
# ax3.axvline(0, color="0.3", linestyle="--", linewidth=1.5, label="target 0")
ax3.axvline(df["std_resid_mean"].mean(), color=BASE_PALETTE[4], linestyle="-", linewidth=1.5,
            label=f"mean {df['std_resid_mean'].mean():.3f}")
ax3.set_xlabel("Mean standardised residual [0.5th-99.5th pct clipped]")
ax3.set_ylabel("Number of series")
ax3.set_title("(c) Calibration of location")
ax3.legend()

# ---------------------------------------------------------------------------
# D. Standardized residual variance, var(v_t / sqrt(F_t)) -- target ~1
# ---------------------------------------------------------------------------

srv_hi = df["std_resid_var"].quantile(0.99)
srv_clipped = df["std_resid_var"].clip(upper=srv_hi)
ax4.hist(srv_clipped, bins=60, color=BASE_PALETTE[3], alpha=0.85)
# ax4.axvline(1, color="0.3", linestyle="--", linewidth=1.5, label="target 1")
ax4.axvline(df["std_resid_var"].mean(), color=BASE_PALETTE[4], linestyle="-", linewidth=1.5,
            label=f"mean {df['std_resid_var'].mean():.3f}")
ax4.set_xlabel("Variance of standardised residuals [99th pct clipped]")
ax4.set_ylabel("Number of series")
ax4.set_title("(d) Calibration of uncertainty")
ax4.legend()

fig.tight_layout()

if SAVE:
    fig.savefig(FIGS_PATH / "holdout_results_validation_summary.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Evaluated {len(df)} series.")
print(f"Aggregate MAE (log points): {df['mae'].mean():.4f}")
print(f"Aggregate bias:             {df['bias'].mean():.4f}")
print(f"Mean standardised residual: {df['std_resid_mean'].mean():.3f}  (target ~0)")
print(f"Mean standardised resid var:{df['std_resid_var'].mean():.3f}  (target ~1)")
print(f"Saved figure to {FIGS_PATH}")
