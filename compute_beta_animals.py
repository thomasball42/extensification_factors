import numpy as np
import pandas as pd
from pathlib import Path
from _kalman_functions import kalman_filter, rts_smoother, fit_tvp_beta, build_annual_diffs

DATA_PATH: Path = Path("..", "mrio_pipeline", "input_data") # this needs the full production dataset from FAO
OUT_PATH: Path = Path("data", "beta_animals.csv")

MIN_OBS: int = 15  # minimum number of valid (non-missing) yearly diff pairs required to fit

elements = ["Area harvested", "Production", "Yield"]
columns = ["Area", "Area Code", "Item", "Item Code", "Element", "Year", "Value", "Unit"]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

df = pd.read_csv(
    DATA_PATH / "Production_Crops_Livestock_E_All_Data_(Normalized).csv",
    encoding="latin-1",
    low_memory=False,
    # usecols=columns,
)

# land_use_df = pd.read_csv(
#     DATA_PATH / "Inputs_LandUse_E_All_Data.csv",
# )

# print(land_use_df)

# ha_unit = df.loc[df.Element == "Area harvested", "Unit"].values[0]

# yield_unit = df.loc[df.Element == "Yield", "Unit"].values[0]

# df = df.drop(columns=["Unit"])
# df = df[df.Element.isin(elements)]

# need to get at 'intensity' of animal production somehow - look at producing animals

anims = [867, 882, 947, 951, 977, 
         1017, 1035, 1058, 1062, 1069, 
         1073, 1080, 1091, 1097, 1141, 
         1166]

df = df[df["Item Code"].isin(anims)]

prod_unit = df.loc[df.Element == "Production", "Unit"].values[0]

print(df[df.Element == "Production"])

quit()