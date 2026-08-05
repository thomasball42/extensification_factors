"""
Time-varying-parameter (Kalman filter/smoother) version of the extensification-
factor pipeline.

Replaces fixed-width rolling-window OLS with a single state-space model per
(Area, Item) series:

    a_t = beta_t * p_t + u_t          u_t ~ N(0, R)      [observation eq.]
    beta_t = beta_{t-1} + eta_t       eta_t ~ N(0, Q)     [state eq., random walk]

where a_t = Delta ln(Area harvested), p_t = Delta ln(Production). Q and R are
estimated per series by maximum likelihood. Output is the *smoothed* beta_t
(uses the whole series, not just past data) plus its standard error, for
every year.

Every recursion below was tested against synthetic data with a known
drifting beta before being wired into this pipeline (see accompanying test
script) -- statsmodels was not available to cross-check against directly,
so this is a from-scratch scalar Kalman filter/RTS smoother implementation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

DATA_PATH: Path = Path("..", "..", "..", "mrio_pipeline", "input_data") # this needs the full production dataset from FAO
OUT_PATH: Path = Path("..", "data", "extensification_factors_tvp.csv")

MIN_OBS: int = 8  # minimum number of valid (non-missing) yearly diff pairs required to fit

elements = ["Area harvested", "Production", "Yield"]
columns = ["Area", "Area Code", "Item", "Item Code", "Element", "Year", "Value", "Unit"]


# ---------------------------------------------------------------------------
# Kalman filter / RTS smoother for the scalar local-level regression model
# ---------------------------------------------------------------------------

def kalman_filter(a, p, Q, R, P0=1e6, beta0=0.0):
    """
    Forward filter for:
        a_t = p_t * beta_t + eps_t,  eps_t ~ N(0, R)
        beta_t = beta_{t-1} + eta_t, eta_t ~ N(0, Q)

    a[t] or p[t] == NaN is treated as a missing observation: the state
    prediction is carried forward (uncertainty grows by Q) but no update
    happens and no log-likelihood contribution is made for that step.

    Returns filtered beta/P, the one-step-ahead predicted beta/P (needed by
    the smoother), and the total log-likelihood over non-missing steps.
    """
    T = len(a)
    beta_filt = np.empty(T)
    P_filt = np.empty(T)
    beta_pred_arr = np.empty(T)
    P_pred_arr = np.empty(T)

    beta_prev, P_prev = beta0, P0
    loglik = 0.0

    for t in range(T):
        beta_pred = beta_prev
        P_pred = P_prev + Q
        beta_pred_arr[t] = beta_pred
        P_pred_arr[t] = P_pred

        obs_ok = np.isfinite(a[t]) and np.isfinite(p[t])
        if obs_ok:
            Z = p[t]
            F = Z * Z * P_pred + R
            v = a[t] - Z * beta_pred
            K = P_pred * Z / F
            beta_t = beta_pred + K * v
            P_t = P_pred - K * Z * P_pred
            loglik += -0.5 * (np.log(2 * np.pi * F) + v * v / F)
        else:
            beta_t = beta_pred
            P_t = P_pred

        beta_filt[t] = beta_t
        P_filt[t] = P_t
        beta_prev, P_prev = beta_t, P_t

    return beta_filt, P_filt, beta_pred_arr, P_pred_arr, loglik


def rts_smoother(beta_filt, P_filt, Q):
    """Backward RTS smoother for the random-walk (transition = 1) case."""
    T = len(beta_filt)
    beta_smooth = np.empty(T)
    P_smooth = np.empty(T)

    beta_smooth[-1] = beta_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in range(T - 2, -1, -1):
        P_pred_next = P_filt[t] + Q
        J = P_filt[t] / P_pred_next if P_pred_next > 0 else 0.0
        beta_smooth[t] = beta_filt[t] + J * (beta_smooth[t + 1] - beta_filt[t])
        P_smooth[t] = P_filt[t] + J * J * (P_smooth[t + 1] - P_pred_next)

    return beta_smooth, P_smooth


def fit_tvp_beta(a, p, P0=1e6):
    """
    Fit Q, R by MLE, then return smoothed beta_t and its SE for every t.
    a, p: 1-D arrays of equal length, calendar-year-aligned, NaN for gaps.
    Returns None if there aren't enough valid points to identify the model.
    """
    a = np.asarray(a, dtype=float)
    p = np.asarray(p, dtype=float)

    valid = np.isfinite(a) & np.isfinite(p)
    if valid.sum() < MIN_OBS:
        return None

    var_a = np.nanvar(a[valid])
    if not (np.isfinite(var_a) and var_a > 0):
        return None

    def neg_loglik(log_params):
        Q, R = np.exp(log_params)
        _, _, _, _, ll = kalman_filter(a, p, Q, R, P0=P0)
        if not np.isfinite(ll):
            return 1e10
        return -ll

    x0 = np.log([0.05 * var_a, var_a])
    res = minimize(
        neg_loglik, x0, method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 500},
    )
    if not res.success:
        return None

    Q_hat, R_hat = np.exp(res.x)
    beta_filt, P_filt, _, _, _ = kalman_filter(a, p, Q_hat, R_hat, P0=P0)
    beta_smooth, P_smooth = rts_smoother(beta_filt, P_filt, Q_hat)

    return {
        "beta": beta_smooth,
        "se": np.sqrt(np.clip(P_smooth, 0, None)),
        "Q": Q_hat,
        "R": R_hat,
    }


def build_annual_diffs(years, log_AH, log_P):
    """
    Reindex onto the full calendar-year range (filling gaps with NaN) before
    differencing, so a genuine multi-year gap grows the state uncertainty
    across that gap instead of being silently compressed into a 1-year step.
    """
    full_years = np.arange(int(years.min()), int(years.max()) + 1)

    ah = pd.Series(log_AH, index=years).reindex(full_years)
    p_ = pd.Series(log_P, index=years).reindex(full_years)

    a_t = np.array(ah.diff().values[1:], dtype=float)
    p_t = np.array(p_.diff().values[1:], dtype=float)
    diff_years = full_years[1:]

    # demean over available points -- equivalent to fitting a fixed intercept
    # in a constant-slope OLS regression; see note in module docstring
    valid = np.isfinite(a_t) & np.isfinite(p_t)
    if valid.sum() > 0:
        a_t = a_t - np.nanmean(a_t[valid])
        p_t = p_t - np.nanmean(p_t[valid])
    a_t[~valid] = np.nan
    p_t[~valid] = np.nan

    return diff_years, a_t, p_t


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

df = pd.read_csv(
    DATA_PATH / "Production_Crops_Livestock_E_All_Data_(Normalized).csv",
    encoding="latin-1",
    low_memory=False,
    usecols=columns,
)

ha_unit = df.loc[df.Element == "Area harvested", "Unit"].values[0]
prod_unit = df.loc[df.Element == "Production", "Unit"].values[0]
yield_unit = df.loc[df.Element == "Yield", "Unit"].values[0]

df = df.drop(columns=["Unit"])
df = df[df.Element.isin(elements)]

# restrict to items that actually report "Area harvested" (i.e. crops, not
# livestock/animal-product items that reuse the "Production" element label)
crop_items = df.loc[df.Element == "Area harvested", "Item"].unique()
df = df[df.Item.isin(crop_items)]

wide = df.pivot_table(
    index=["Area", "Area Code", "Item", "Item Code", "Year"], columns="Element", values="Value"
).reset_index()

records = []
groups = wide.groupby(["Area", "Area Code", "Item", "Item Code"])
for i, ((area, area_code, item, item_code), g) in enumerate(groups): 

    print(i + 1, "/", len(groups), area, item, item_code)

    g = g.sort_values("Year").dropna(subset=elements)
    if len(g) < MIN_OBS + 1:
        continue

    with np.errstate(divide="ignore"):
        log_AH = np.log(g["Area harvested"].values)
        log_P = np.log(g["Production"].values)

    years = g["Year"].values.astype(int)

    diff_years, a_t, p_t = build_annual_diffs(years, log_AH, log_P)

    fit = fit_tvp_beta(a_t, p_t)
    if fit is None:
        continue

    record = {
        "Area": area,
        "Area Code": area_code,
        "Item": item,
        "Item Code": item_code,
        "current_year": int(years[-1]),
        f"current_area_harvested_{ha_unit}": g["Area harvested"].values[-1],
        f"current_production_{prod_unit}": g["Production"].values[-1],
        f"current_yield_{yield_unit}": g["Yield"].values[-1],
        "Q_hat": fit["Q"],
        "R_hat": fit["R"],
    }

    for yr, beta, se in zip(diff_years, fit["beta"], fit["se"]):
        record[f"beta_{int(yr)}"] = beta
        record[f"se_{int(yr)}"] = se

    records.append(record)

out = pd.DataFrame.from_records(records)

id_cols = [c for c in out.columns if not (c.startswith("beta_") or c.startswith("se_"))]
beta_cols = sorted((c for c in out.columns if c.startswith("beta_")), key=lambda c: int(c.split("_")[1]))

year_ordered_cols = []
for bc in beta_cols:
    yr = bc.split("_", 1)[1]
    year_ordered_cols.append(bc)
    year_ordered_cols.append(f"se_{yr}")

out = out[id_cols + year_ordered_cols]

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(out)} rows to {OUT_PATH}")