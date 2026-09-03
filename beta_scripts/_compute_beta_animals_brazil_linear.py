"""
Static (non-time-varying) OLS counterpart to _compute_beta_animals_brazil.py,
for Brazil's local (immediate-region) pasture dataset. Mirrors
_compute_beta_animals_linear.py's method (single full-sample OLS beta per
series, on the same reindexed/demeaned annual diffs build_annual_diffs
produces), fit instead on Brazil's regional panel.

    Delta ln(area_past_ha)_t = beta * Delta ln(P_T)_t + eps_t

See brazil/data/readme for the input schema.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress
from _stats import build_annual_diffs

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = REPO_ROOT / "brazil" / "data" / "NL_local_domain.csv"
OUT_PATH: Path = REPO_ROOT / "brazil" / "outputs" / "beta_animals_linear_local_Brazil.csv"

MIN_OBS: int = 8  # see _compute_beta_animals_brazil.py

AREA_COL = "area_past_ha"
PRODUCTION_COL = "P_T"
GROUP_COLS = ["uf", "cod_rgi", "nome_rgi"]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH, encoding="utf-8")
missing_cols = [c for c in ["year", *GROUP_COLS, AREA_COL, PRODUCTION_COL] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df["year"] = df["year"].astype(int)

records = []
groups = list(df.groupby("cod_rgi"))
for i, (cod_rgi, g) in enumerate(groups):
    g = g.sort_values("year")
    g = g[np.isfinite(g[AREA_COL]) & np.isfinite(g[PRODUCTION_COL])]
    g = g[(g[AREA_COL] > 0) & (g[PRODUCTION_COL] > 0)]
    if len(g) < MIN_OBS + 1:
        continue

    current = g.iloc[-1]
    print(i + 1, "/", len(groups), current["uf"], current["nome_rgi"])

    years = g["year"].values
    with np.errstate(divide="ignore"):
        log_area = np.log(g[AREA_COL].values)
        log_prod = np.log(g[PRODUCTION_COL].values)

    _, a_t, p_t = build_annual_diffs(years, log_area, log_prod)

    valid = np.isfinite(a_t) & np.isfinite(p_t)
    if valid.sum() < MIN_OBS or np.var(p_t[valid]) == 0:
        continue

    beta, intercept, r_value, p_value, std_err = linregress(p_t[valid], a_t[valid])

    records.append({
        "uf": current["uf"],
        "cod_rgi": cod_rgi,
        "nome_rgi": current["nome_rgi"],
        "Item": "All pasture-based animal products",
        "Item Code": 0,
        "current_year": int(current["year"]),
        "current_area_pasture_ha": current[AREA_COL],
        "current_production_kg": current[PRODUCTION_COL],
        "beta": beta,
        "beta_se": std_err,
        "n_obs": int(valid.sum()),
    })

out = pd.DataFrame.from_records(records)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(out)} rows to {OUT_PATH}")
