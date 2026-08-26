"""
Technical Validation hold-out cross-validation.

For each qualifying series, hold out the final HOLDOUT_YEARS years. Fit
(Q, R) by MLE using ONLY the training years, then run the Kalman recursion
through the full series (train + test) with those fitted parameters, and
score the one-step-ahead predictions on the held-out test years.

The series fitted here are the same two-way (country + year) fixed-effects
residualized diffs the main pipeline (_compute_beta_crops.py) actually fits
its TVP model on -- not raw diffs, which would validate a different
data-generating process than the one that actually produces beta_crops.csv.
IMPORTANT: the residualization for a target country excludes that country's
own held-out years from the fixed-effects fit, so its own held-out values
can't leak into its own entity effect -- see
_holdout_common.residualized_target_series and
_stats.residualize_two_way_fe_oos.

If a fitted model is well calibrated, the standardized one-step-ahead
residuals (v_t / sqrt(F_t)) on the held-out years should have mean ~0 and
variance ~1 -- these are the numbers to report in the paper, alongside
plain-units error metrics (MAE, bias) that are easier for a reader to
interpret directly.

For tractability, this runs on a random sample of series by default. Set
SAMPLE_N = None to run on everything.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _stats import kalman_filter_diag
from _holdout_common import load_wide, load_groups, build_item_area_data, residualized_target_series

OUT_PATH: Path = Path("outputs") / "validation" / "holdout_validation.csv"

MIN_TRAIN_OBS = 15
HOLDOUT_YEARS = 10
SAMPLE_N = 5000       # None = run on every qualifying series
RANDOM_SEED = 0


def fit_on_train_evaluate_on_test(a, p, train_end):
    """a, p: two-way (country+year) fixed-effects-residualized diffs for one
    series, already centred by the residualization step (see
    residualized_target_series) -- no further demeaning needed here.
    train_end: index such that [0, train_end) is train, [train_end, T) is test."""
    T = len(a)
    valid = np.isfinite(a) & np.isfinite(p)
    train_valid = valid.copy()
    train_valid[train_end:] = False
    if train_valid.sum() < MIN_TRAIN_OBS:
        return None
    if not valid[train_end:].any():
        return None  # no usable test points

    var_a_train = np.nanvar(a[train_valid])
    if not (np.isfinite(var_a_train) and var_a_train > 0):
        return None

    def neg_ll(log_qr):
        Q, R = np.exp(log_qr)
        *_, ll = kalman_filter_diag(a, p, Q, R, fit_upto=train_end)
        return -ll if np.isfinite(ll) else 1e10

    res = minimize(
        neg_ll, np.log([0.05 * var_a_train, var_a_train]),
        method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 500},
    )
    if not res.success:
        return None
    Q_hat, R_hat = np.exp(res.x)

    _, v, F, _ = kalman_filter_diag(a, p, Q_hat, R_hat, fit_upto=T)

    test_mask = np.zeros(T, dtype=bool)
    test_mask[train_end:] = True
    test_mask &= valid

    v_test, F_test = v[test_mask], F[test_mask]
    return {
        "n_test": int(test_mask.sum()),
        "Q_hat": Q_hat, "R_hat": R_hat,
        "mae": float(np.mean(np.abs(v_test))),
        "bias": float(np.mean(v_test)),
        "std_resid_mean": float(np.mean(v_test / np.sqrt(F_test))),
        "std_resid_var": float(np.var(v_test / np.sqrt(F_test))),
    }


wide = load_wide()
groups = load_groups(wide, SAMPLE_N, RANDOM_SEED)

by_item = {}
for (area, item, item_code), _ in groups:
    by_item.setdefault((item, item_code), []).append(area)
total = sum(len(areas) for areas in by_item.values())

results = []
done = 0
for (item, item_code), target_areas in by_item.items():
    g_item = wide[(wide["Item"] == item) & (wide["Item Code"] == item_code)]
    area_data, panel = build_item_area_data(g_item)
    for area in target_areas:
        done += 1
        print(done, "/", total, area, item, item_code)
        area_entry = area_data.get(area)
        if area_entry is None:
            continue
        g = area_entry["g"]
        diff_years = area_entry["diff_years"]
        if len(g) < MIN_TRAIN_OBS + HOLDOUT_YEARS + 1:
            continue
        T = len(diff_years)
        train_end = T - HOLDOUT_YEARS
        if train_end < MIN_TRAIN_OBS:
            continue

        a, p = residualized_target_series(area, area_entry, panel, train_end)
        out = fit_on_train_evaluate_on_test(a, p, train_end)
        if out is None:
            continue
        out.update({"Area": area, "Item": item, "Item Code": item_code})
        results.append(out)

res_df = pd.DataFrame(results)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
res_df.to_csv(OUT_PATH, index=False)

print(f"\nEvaluated {len(res_df)} series.")
print(f"Aggregate MAE (a_t, log points): {res_df['mae'].mean():.4f}")
print(f"Aggregate bias:                  {res_df['bias'].mean():.4f}")
print(f"Mean standardized residual:      {res_df['std_resid_mean'].mean():.3f}  (target ~0)")
print(f"Mean standardized residual var:  {res_df['std_resid_var'].mean():.3f}  (target ~1)")
print(f"Written to {OUT_PATH}")
