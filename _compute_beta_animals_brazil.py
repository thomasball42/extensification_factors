"""
Time-varying beta estimates for Brazil's local (immediate-region) pasture
dataset -- the same TVP/Kalman model _compute_beta_animals.py fits on
FAOSTAT national data, fit instead on Brazil's higher-resolution regional
panel, reusing the same _stats.py functions so the model code is identical,
not just parameter-matched.

    Delta ln(area_past_ha)_t = beta_t * Delta ln(P_T)_t + eps_t
    beta_t = beta_{t-1} + eta_t

See brazil/data/readme for the input schema and brazil/R/scripts/
convert_rds_to_csv.R for how the input csv is produced from the source .rds.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from _stats import fit_tvp_beta, build_annual_diffs, build_raw_diffs, residualize_two_way_fe
from _utils import load_q_prior_config

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = REPO_ROOT / "brazil" / "data" / "NL_local_domain.csv"
OUT_PATH: Path = REPO_ROOT / "brazil" / "outputs" / "beta_animals_local_Brazil.csv"

# The FAOSTAT national analysis (_compute_beta_animals.py) uses 15. This local
# file covers 2011-2021, so a region can contribute at most 10 annual
# log-difference pairs -- 15 would exclude every region.
MIN_OBS: int = 8

USE_Q_PRIOR: bool = True  # shrinkage prior on Q (see _stats.q_shrinkage_penalty); False = unregularized MLE
USE_RESIDUALIZATION: bool = False  # two-way (region + year) FE residualization; off by default -- holdout validation on the main FAOSTAT series found this didn't improve results

Q_PRIOR_SCALE, Q_PRIOR_STRENGTH = load_q_prior_config() if USE_Q_PRIOR else (None, 1.0)

AREA_COL = "area_past_ha"
PRODUCTION_COL = "P_T"
GROUP_COLS = ["uf", "cod_rgi", "nome_rgi"]  # cod_rgi alone is nationally unique; uf/nome_rgi are carried through as labels

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH, encoding="utf-8")
missing_cols = [c for c in ["year", *GROUP_COLS, AREA_COL, PRODUCTION_COL] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df["year"] = df["year"].astype(int)

region_data = {}
for cod_rgi, g in df.groupby("cod_rgi"):
    g = g.sort_values("year")
    g = g[np.isfinite(g[AREA_COL]) & np.isfinite(g[PRODUCTION_COL])]
    g = g[(g[AREA_COL] > 0) & (g[PRODUCTION_COL] > 0)]
    if len(g) < MIN_OBS + 1:
        continue
    region_data[cod_rgi] = g

print(f"{len(region_data)} regions")

if USE_RESIDUALIZATION:
    print("fitting two-way fixed effects")
    panel_rows = []
    for cod_rgi, g in region_data.items():
        years = g["year"].values
        with np.errstate(divide="ignore"):
            log_area = np.log(g[AREA_COL].values)
            log_prod = np.log(g[PRODUCTION_COL].values)
        diff_years, a_t, p_t = build_raw_diffs(years, log_area, log_prod)
        panel_rows.append(pd.DataFrame({"entity": cod_rgi, "time": diff_years, "a": a_t, "p": p_t}))

    panel = residualize_two_way_fe(pd.concat(panel_rows, ignore_index=True))
    diffs_by_region = {}
    for cod_rgi, sub in panel.groupby("entity"):
        sub = sub.sort_values("time")
        diffs_by_region[cod_rgi] = (sub["time"].values.astype(int), sub["a_resid"].values, sub["p_resid"].values)
else:
    diffs_by_region = {}
    for cod_rgi, g in region_data.items():
        years = g["year"].values
        with np.errstate(divide="ignore"):
            log_area = np.log(g[AREA_COL].values)
            log_prod = np.log(g[PRODUCTION_COL].values)
        diffs_by_region[cod_rgi] = build_annual_diffs(years, log_area, log_prod)

records = []
for i, (cod_rgi, g) in enumerate(region_data.items()):
    current = g.iloc[-1]
    print(i + 1, "/", len(region_data), current["uf"], current["nome_rgi"])

    diff_years, a_t, p_t = diffs_by_region[cod_rgi]

    fit = fit_tvp_beta(a_t, p_t, min_obs=MIN_OBS, q_prior_scale=Q_PRIOR_SCALE, q_prior_strength=Q_PRIOR_STRENGTH)
    if fit is None:
        continue

    record = {
        "uf": current["uf"],
        "cod_rgi": cod_rgi,
        "nome_rgi": current["nome_rgi"],
        "Item": "All pasture-based animal products",
        "Item Code": 0,
        "current_year": int(current["year"]),
        "current_area_pasture_ha": current[AREA_COL],
        "current_production_kg": current[PRODUCTION_COL],
        "Q_hat": fit["Q"],
        "R_hat": fit["R"],
        "q_prior_scale": Q_PRIOR_SCALE,
        "q_prior_strength": Q_PRIOR_STRENGTH,
        "n_obs": int((np.isfinite(a_t) & np.isfinite(p_t)).sum()),
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
