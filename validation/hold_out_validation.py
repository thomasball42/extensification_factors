"""
Technical Validation hold-out cross-validation.

For each qualifying series, hold out the final HOLDOUT_YEARS years. Fit
(Q, R) by MLE using ONLY the training years, then run the Kalman recursion
through the full series (train + test) with those fitted parameters, and
score the one-step-ahead predictions on the held-out test years.

IMPORTANT: demeaning (the stand-in for the intercept -- see main pipeline)
must use train-only statistics. Demeaning with the full series' mean would
leak test-period information into the very data used to "predict" it,
undermining the whole point of a holdout test. This script computes the
mean from the training window only and applies it to both windows.

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
from _functions import kalman_filter_diag, build_raw_diffs

DATA_PATH: Path = Path("data") / "inputs"
OUT_PATH: Path = Path("outputs") / "validation" / "holdout_validation.csv"

MIN_TRAIN_OBS = 15
HOLDOUT_YEARS = 10
SAMPLE_N = 5000       # None = run on every qualifying series
RANDOM_SEED = 0

elements = ["Area harvested", "Production", "Yield"]
columns = ["Area", "Item", "Item Code", "Element", "Year", "Value", "Unit"]


def fit_on_train_evaluate_on_test(a_raw, p_raw, train_end):
    """a_raw, p_raw: full raw (non-demeaned) diffed series.
    train_end: index such that [0, train_end) is train, [train_end, T) is test."""
    T = len(a_raw)
    valid = np.isfinite(a_raw) & np.isfinite(p_raw)
    train_valid = valid.copy()
    train_valid[train_end:] = False
    if train_valid.sum() < MIN_TRAIN_OBS:
        return None
    if not valid[train_end:].any():
        return None  # no usable test points

    a_mean = np.nanmean(a_raw[train_valid])
    p_mean = np.nanmean(p_raw[train_valid])
    a = a_raw - a_mean
    p = p_raw - p_mean
    a[~valid] = np.nan
    p[~valid] = np.nan

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


df = pd.read_csv(
    DATA_PATH / "Production_Crops_Livestock_E_All_Data_(Normalized).csv",
    encoding="latin-1", low_memory=False, usecols=columns,
)

df = df[df.Element.isin(elements)]
crop_items = df.loc[df.Element == "Area harvested", "Item"].unique()
df = df[df.Item.isin(crop_items)]
wide = df.pivot_table(index=["Area", "Item", "Item Code", "Year"], columns="Element", values="Value").reset_index()

groups = list(wide.groupby(["Area", "Item", "Item Code"]))
if SAMPLE_N is not None and len(groups) > SAMPLE_N:
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(groups), size=SAMPLE_N, replace=False)
    groups = [groups[i] for i in idx]

results = []
for i, ((area, item, item_code), g) in enumerate(groups):
    print(i + 1, "/", len(groups), area, item, item_code)
    g = g.sort_values("Year").dropna(subset=elements)
    if len(g) < MIN_TRAIN_OBS + HOLDOUT_YEARS + 1:
        continue
    with np.errstate(divide="ignore"):
        log_AH = np.log(g["Area harvested"].values)
        log_P = np.log(g["Production"].values)
    years = g["Year"].values.astype(int)

    diff_years, a_raw, p_raw = build_raw_diffs(years, log_AH, log_P)
    T = len(diff_years)
    train_end = T - HOLDOUT_YEARS
    if train_end < MIN_TRAIN_OBS:
        continue

    out = fit_on_train_evaluate_on_test(a_raw, p_raw, train_end)
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