"""
Plots for the Q-prior shrinkage validation
(validation/q_prior_shrinkage_validation.py): does adding a shrinkage prior
on Q fix the overfitting signature (Q4 -- the quartile with the most
detected drift -- underperforming the static baseline) without hurting the
other quartiles?

Reads:
  outputs/validation/q_prior_shrinkage/quartile_comparison_all_settings.csv

Produces:
  quartile_pct_improvement.png   -- pct_improvement by quartile, one line per candidate setting
  quartile_win_rate.png          -- win_rate by quartile, one line per candidate setting

Both x-axes are candidate settings in the order they were run, labelled with
their (q_prior_scale, q_prior_strength); both plots break the metric out by
Q_hat/R_hat quartile (Q1 = least detected drift .. Q4 = most).
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

dat_path = Path("outputs") / "validation" / "q_prior_shrinkage" / "quartile_comparison_all_settings.csv"
figs_path = Path("..") / "figs" / "q_prior_shrinkage_validation"

SAVE = True

os.makedirs(figs_path, exist_ok=True)

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
QUARTILE_COLORS = {"Q1": BASE_PALETTE[0], "Q2": BASE_PALETTE[1], "Q3": BASE_PALETTE[2], "Q4": BASE_PALETTE[3]}

df = pd.read_csv(dat_path)


def fmt_scale(x):
    return "None" if pd.isna(x) else f"{x:g}"


settings = list(df["setting"].unique())  # preserves the run order (CANDIDATES order)
setting_meta = df.drop_duplicates("setting").set_index("setting")
x_labels = [
    f"{s}\n(scale={fmt_scale(setting_meta.loc[s, 'q_prior_scale'])}, "
    f"strength={setting_meta.loc[s, 'q_prior_strength']:g})"
    for s in settings
]
x_pos = range(len(settings))


def quartile_lines(ax, value_col, ylabel):
    for q, color in QUARTILE_COLORS.items():
        sub = df[df["quartile"] == q].set_index("setting").loc[settings]
        ax.plot(x_pos, sub[value_col], marker="o", markersize=6, linewidth=1.75,
                 color=color, label=q)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.legend(title="Q_hat/R_hat quartile", loc="best", framealpha=0.9)


# ---------------------------------------------------------------------------
# 1. Pct improvement (TVP vs static) by quartile, across candidate settings
# ---------------------------------------------------------------------------

fig1, ax1 = plt.subplots(figsize=(11, 6))
quartile_lines(ax1, "pct_improvement", "Mean relative MAE improvement, TVP vs. static\n(positive = TVP better)")
ax1.axhline(0, color="0.3", linestyle="--", linewidth=1, zorder=0)
ax1.set_title("Does the Q-prior fix Q4 overfitting without hurting Q1-Q3?\n"
              "(Q4 crossing above 0 = fixed; watch that Q1-Q3 don't drop)")
fig1.tight_layout()

# ---------------------------------------------------------------------------
# 2. Win rate by quartile, across candidate settings
# ---------------------------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(11, 6))
quartile_lines(ax2, "win_rate", "Share of series where TVP beats static")
ax2.axhline(0.5, color="0.3", linestyle="--", linewidth=1, zorder=0)
ax2.set_ylim(0, 1)
ax2.set_title("TVP win rate by quartile, across candidate Q-prior settings")
fig2.tight_layout()

plt.show()

if SAVE:
    fig1.savefig(figs_path / "quartile_pct_improvement.png", dpi=300, bbox_inches="tight")
    fig2.savefig(figs_path / "quartile_win_rate.png", dpi=300, bbox_inches="tight")
