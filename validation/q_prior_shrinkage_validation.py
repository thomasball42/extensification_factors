"""
Validates whether an optional shrinkage prior on Q (see q_shrinkage_penalty
in _functions.py) fixes the overfitting signature found by
hold_out_static_comparison.py: series in the top quartile of fitted Q_hat/
R_hat ratio (most detected drift) perform worse out-of-sample than a static
OLS baseline, while the other three quartiles improve on it.

Loads a fixed sample of series ONCE (so every candidate setting is compared
on exactly the same series -- an apples-to-apples comparison), then for each
candidate (q_prior_scale, q_prior_strength) re-runs the held-out comparison
and reports n / win_rate / mae_static / mae_tvp / pct_improvement broken
down by Q_hat/R_hat quartile.

This is a one-off diagnostic -- it does not change compute_beta_crops.py's
default output. Outputs are written under outputs/validation/q_prior_shrinkage/,
separate from hold_out_static_comparison.py's own default output paths.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hold_out_static_comparison import load_groups, run_holdout_comparison

SAMPLE_N = None
RANDOM_SEED = 0
OUT_DIR: Path = Path("outputs") / "validation" / "q_prior_shrinkage"

CANDIDATES = [
    # {"label": "weak", "q_prior_scale": 0.01, "q_prior_strength": 1.0},
    {"label": "baseline",         "q_prior_scale": None,    "q_prior_strength": 1.0},
    {"label": "str_0.9", "q_prior_scale": 0.02, "q_prior_strength": 0.9},
    {"label": "str_0.8", "q_prior_scale": 0.02, "q_prior_strength": 0.8},
    {"label": "str_0.7", "q_prior_scale": 0.02, "q_prior_strength": 0.7},
    # {"label": "strong",           "q_prior_scale": 0.001,   "q_prior_strength": 0.5},    # previous best, kept as anchor
    # {"label": "strength_0.25",    "q_prior_scale": 0.001,   "q_prior_strength": 0.25},
    # {"label": "strength_0.1",     "q_prior_scale": 0.001,   "q_prior_strength": 0.1},
    # {"label": "scale_1e-4",       "q_prior_scale": 0.0001,  "q_prior_strength": 0.5},
    # {"label": "scale_1e-5",       "q_prior_scale": 0.00001, "q_prior_strength": 0.5},
    # {"label": "scale_1e-4_tight", "q_prior_scale": 0.0001,  "q_prior_strength": 0.25},
]


def quartile_summary(res_df):
    """Bin series into quartiles by fitted Q_hat/R_hat ratio and report
    n, win_rate, mean mae_static, mean mae_tvp, and mean pct_improvement
    per quartile (Q1 = lowest ratio / least detected drift, Q4 = highest)."""
    df = res_df.copy()
    df["qr_ratio"] = df["Q_hat"] / df["R_hat"]
    df["quartile"] = pd.qcut(df["qr_ratio"], 4, labels=["Q1", "Q2", "Q3", "Q4"])

    def _agg(g):
        pct_improvement = (g["mae_static"] - g["mae_tvp"]) / g["mae_static"]
        return pd.Series({
            "n": len(g),
            "win_rate": g["tvp_beats_static"].mean(),
            "mae_static": g["mae_static"].mean(),
            "mae_tvp": g["mae_tvp"].mean(),
            "pct_improvement": pct_improvement.mean(),
        })

    return df.groupby("quartile", observed=True).apply(_agg, include_groups=False)


if __name__ == "__main__":
    groups = load_groups(SAMPLE_N, RANDOM_SEED)
    print(f"Loaded {len(groups)} series (sample_n={SAMPLE_N}, seed={RANDOM_SEED}); "
          f"reusing this exact sample for every candidate setting below.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tables = {}
    for cfg in CANDIDATES:
        print(f"\n=== {cfg['label']} (q_prior_scale={cfg['q_prior_scale']}, "
              f"q_prior_strength={cfg['q_prior_strength']}) ===")
        res_df, agg_df, _ = run_holdout_comparison(
            groups,
            q_prior_scale=cfg["q_prior_scale"], q_prior_strength=cfg["q_prior_strength"],
            out_path=OUT_DIR / f"holdout_{cfg['label']}.csv",
            out_by_item_path=None,
            out_aggregates_path=OUT_DIR / f"holdout_{cfg['label']}_aggregates.csv",
            production_weighting=False, verbose=False,
        )
        table = quartile_summary(res_df)
        table.insert(0, "q_prior_strength", cfg["q_prior_strength"])
        table.insert(0, "q_prior_scale", cfg["q_prior_scale"])
        tables[cfg["label"]] = table
        print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    combined = pd.concat(tables, names=["setting", "quartile"])
    combined.to_csv(OUT_DIR / "quartile_comparison_all_settings.csv")
    print("\n=== Combined quartile comparison across all settings ===")
    print(combined.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nWritten per-setting CSVs and quartile_comparison_all_settings.csv to {OUT_DIR}")
