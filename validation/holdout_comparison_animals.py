"""
Animals counterpart to holdout_comparison.py: the same leak-safe
TVP-vs-static held-out comparison methodology (see that module's
docstring for the full rationale -- residualization, train/test split,
production weighting), applied to the animals/pasture pipeline instead of
crops -- pasture area vs. grazing-animal production, pooled per Area
exactly as _compute_beta_animals.py pools it (there is no per-Item split
for animals, unlike crops).

All of the actual model-fitting/scoring logic (compare_on_series) and the
leak-safe residualization (residualized_target_series) are imported
unchanged from the crops modules -- only the data loading differs, via
_holdout_common_animals.load_area_data.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _holdout_common import residualized_target_series
from _holdout_common_animals import load_area_data
from holdout_comparison import (
    compare_on_series, reindex_levels,
    MIN_TRAIN_OBS, HOLDOUT_YEARS, Q_PRIOR_SCALE, Q_PRIOR_STRENGTH,
)

OUT_PATH: Path = Path("outputs") / "validation" / "holdout_static_comparison_animals_pweighted.csv"

ITEM_LABEL = "All pasture-based animal products"
ITEM_CODE = 0


def score_series(area, area_entry, panel, q_prior_scale=None, q_prior_strength=1.0):
    """Residualize `area`'s pasture-area/production series (against the
    cross-country panel, holding out its own test years from the
    fixed-effects fit) and run the held-out comparison on it. Returns the
    tagged result dict, or None if the series doesn't qualify."""
    g = area_entry["g"]
    diff_years = area_entry["diff_years"]
    if len(g) < MIN_TRAIN_OBS + HOLDOUT_YEARS + 1:
        return None
    T = len(diff_years)
    train_end = T - HOLDOUT_YEARS
    if train_end < MIN_TRAIN_OBS:
        return None

    a, p = residualized_target_series(area, area_entry, panel, train_end)
    prod_level = reindex_levels(g["Year"].values.astype(int), g["Production"].values)

    out = compare_on_series(a, p, prod_level, train_end, q_prior_scale, q_prior_strength)
    if out is None:
        return None
    out.update({"Area": area, "Item": ITEM_LABEL, "Item Code": ITEM_CODE})
    return out


if __name__ == "__main__":
    area_data, panel = load_area_data()
    total = len(area_data)

    results = []
    for i, (area, area_entry) in enumerate(area_data.items(), 1):
        print(i, "/", total, area)
        out = score_series(area, area_entry, panel, Q_PRIOR_SCALE, Q_PRIOR_STRENGTH)
        if out is None:
            continue
        results.append(out)

    res_df = pd.DataFrame(results)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUT_PATH, index=False)

    print(f"\nScored {len(res_df)} of {total} areas (held-out prediction, {HOLDOUT_YEARS} test years each).")
    if len(res_df) > 0:
        print(f"Aggregate MAE -- static baseline: {res_df['mae_static'].mean():.4f}")
        print(f"Aggregate MAE -- TVP:             {res_df['mae_tvp'].mean():.4f}")
        print(f"Share of series where TVP beats static baseline: {res_df['tvp_beats_static'].mean():.1%}")
        print(f"Weighted MAE -- TVP (production-weighted): "
              f"{(res_df['mae_tvp'] * res_df['test_avg_production']).sum() / res_df['test_avg_production'].sum():.4f}")
    print(f"Written to {OUT_PATH}")