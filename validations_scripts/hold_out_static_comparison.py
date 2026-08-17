"""
Comparison of TVP vs. static (non-time-varying) OLS
baseline, via held-out cross-validation -- the actual test of whether
letting beta move over time earns its keep, as opposed to the descriptive
(and, on its own, non-diagnostic) comparison in
validate_3_descriptive_comparison.py.
ing the two summary outputs.

For each qualifying series: hold out the final HOLDOUT_YEARS years, fit (a)
the static OLS beta and (b) the TVP model's (Q, R) using ONLY the training
years, then compare one-step-ahead prediction error on the held-out years
for both. Demeaning (the intercept stand-in) uses train-only statistics for
both models, to avoid leaking test-period information into either fit.

For tractability this runs on a random sample of series by default (SAMPLE_N),
matching the convention used in comparable gridded-dataset validation papers
(e.g. cross-validating on a random subset of stations rather than all of
them). Set SAMPLE_N = None for a full run.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

DATA_PATH: Path = Path("data") / "inputs"
OUT_PATH: Path = Path("outputs") / "validation" / "holdout_static_comparison.csv"

MIN_TRAIN_OBS = 15
HOLDOUT_YEARS = 10
SAMPLE_N = 5000       # None = run on every qualifying series
RANDOM_SEED = 0

elements = ["Area harvested", "Production", "Yield"]
columns = ["Area", "Item", "Item Code", "Element", "Year", "Value", "Unit"]


def kalman_filter_diag(a, p, Q, R, P0=1e6, beta0=0.0, fit_upto=None):
    """Kalman recursion, restricted to accumulate log-likelihood only over
    indices < fit_upto (the training window), while still filtering through
    the full series so test-period one-step-ahead residuals are available."""
    T = len(a)
    if fit_upto is None:
        fit_upto = T
    v_arr = np.full(T, np.nan)
    F_arr = np.full(T, np.nan)
    beta_prev, P_prev = beta0, P0
    loglik = 0.0
    for t in range(T):
        beta_pred = beta_prev
        P_pred = P_prev + Q
        if np.isfinite(a[t]) and np.isfinite(p[t]):
            Z = p[t]
            F = Z * Z * P_pred + R
            v = a[t] - Z * beta_pred
            K = P_pred * Z / F
            beta_t = beta_pred + K * v
            P_t = P_pred - K * Z * P_pred
            v_arr[t], F_arr[t] = v, F
            if t < fit_upto:
                loglik += -0.5 * (np.log(2 * np.pi * F) + v * v / F)
        else:
            beta_t, P_t = beta_pred, P_pred
        beta_prev, P_prev = beta_t, P_t
    return v_arr, F_arr, loglik


def build_raw_diffs(years, log_AH, log_P):
    """Reindex onto full calendar years and difference -- NOT demeaned here;
    demeaning happens after the train/test split, using train-only stats,
    to avoid leaking test-period information into either model's fit."""
    full_years = np.arange(int(years.min()), int(years.max()) + 1)
    ah = pd.Series(log_AH, index=years).reindex(full_years)
    p_ = pd.Series(log_P, index=years).reindex(full_years)
    a_t = ah.diff().values[1:].copy()
    p_t = p_.diff().values[1:].copy()
    return full_years[1:], a_t, p_t


def compare_on_series(a_raw, p_raw, train_end):
    T = len(a_raw)
    valid = np.isfinite(a_raw) & np.isfinite(p_raw)
    train_valid = valid.copy()
    train_valid[train_end:] = False
    if train_valid.sum() < MIN_TRAIN_OBS or not valid[train_end:].any():
        return None

    a_mean = np.nanmean(a_raw[train_valid])
    p_mean = np.nanmean(p_raw[train_valid])
    a = a_raw - a_mean
    p = p_raw - p_mean
    a[~valid] = np.nan
    p[~valid] = np.nan

    a_tr, p_tr = a[train_valid], p[train_valid]
    var_p_tr = np.var(p_tr)
    if not (np.isfinite(var_p_tr) and var_p_tr > 0):
        return None

    # --- static OLS baseline, fit on train only, applied to test ---
    beta_static = np.sum(p_tr * a_tr) / np.sum(p_tr * p_tr)  # demeaned -> Cov/Var form

    test_mask = np.zeros(T, dtype=bool)
    test_mask[train_end:] = True
    test_mask &= valid
    a_test, p_test = a[test_mask], p[test_mask]

    resid_static = a_test - beta_static * p_test
    mae_static = float(np.mean(np.abs(resid_static)))
    bias_static = float(np.mean(resid_static))

    # --- TVP model, (Q,R) fit on train only, evaluated on test ---
    var_a_tr = np.var(a_tr)
    if not (np.isfinite(var_a_tr) and var_a_tr > 0):
        return None

    def neg_ll(log_qr):
        Q, R = np.exp(log_qr)
        *_, ll = kalman_filter_diag(a, p, Q, R, fit_upto=train_end)
        return -ll if np.isfinite(ll) else 1e10

    res = minimize(neg_ll, np.log([0.05 * var_a_tr, var_a_tr]),
                    method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 500})
    if not res.success:
        return None
    Q_hat, R_hat = np.exp(res.x)
    v, F, _ = kalman_filter_diag(a, p, Q_hat, R_hat, fit_upto=T)
    v_test = v[test_mask]
    mae_tvp = float(np.mean(np.abs(v_test)))
    bias_tvp = float(np.mean(v_test))

    return {
        "n_test": int(test_mask.sum()),
        "Q_hat": Q_hat, "R_hat": R_hat,
        "beta_static_train": beta_static,
        "mae_static": mae_static, "bias_static": bias_static,
        "mae_tvp": mae_tvp, "bias_tvp": bias_tvp,
        "tvp_beats_static": mae_tvp < mae_static,
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
    out = compare_on_series(a_raw, p_raw, train_end)
    if out is None:
        continue
    out.update({"Area": area, "Item": item, "Item Code": item_code})
    results.append(out)

res_df = pd.DataFrame(results)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
res_df.to_csv(OUT_PATH, index=False)

print(f"\nCompared {len(res_df)} series (held-out prediction, {HOLDOUT_YEARS} test years each).")
print(f"Aggregate MAE -- static baseline: {res_df['mae_static'].mean():.4f}")
print(f"Aggregate MAE -- TVP:             {res_df['mae_tvp'].mean():.4f}")
print(f"Share of series where TVP beats static baseline: {res_df['tvp_beats_static'].mean():.1%}")
print(f"Written to {OUT_PATH}")