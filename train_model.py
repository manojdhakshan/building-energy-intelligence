"""
STEP 1 - Train the AI model.

Run once:  python train_model.py
It reads the CBECS 2018 dataset from the "data" folder, trains a Random Forest
model and saves it as "model.joblib" for the app to use.
"""
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from building_logic import BUILDING_TYPES, FEATURES, KBTU_TO_KWH, make_features

# The 12 columns used from the full CBECS file (data/cbecs2018_final_public.csv, 1,249 columns)
DATA_FILE = "data/cbecs2018_selected_columns.csv"

# --- 1. Load only the columns we need ---------------------------------------
cols = ["PBA", "PUBCLIM", "SQFT", "BLDSHP", "GLSSPC", "NFLOOR", "FLCEILHT",
        "HEATP", "COOLP", "CWUSED", "MFHTBTU", "MFCLBTU"]
df = pd.read_csv(DATA_FILE, usecols=cols)
print(f"Buildings in dataset: {len(df)}")

# --- 2. Clean the data --------------------------------------------------------
# Building activity codes: merge the two mall types into one "shopping centre" group
df["PBA"] = df["PBA"].replace({24: 23})
df = df[df["PBA"].isin({code for _, code in BUILDING_TYPES})]   # keep the types offered in the app
df = df[df["PUBCLIM"] != 7]                    # climate withheld for confidentiality
df = df[df["FLCEILHT"] != 995]                 # ceiling height "more than 50 ft"
df = df[df["CWUSED"] != 1]                     # district chilled water is not in the cooling totals
df = df.dropna(subset=["MFHTBTU", "MFCLBTU"])
df["NFLOOR"] = df["NFLOOR"].replace({994: 12, 995: 15})   # "10-14" and "15+" floors
df[["HEATP", "COOLP"]] = df[["HEATP", "COOLP"]].fillna(0)  # blank = not heated / not cooled

# Targets: yearly heating / cooling energy per sq. ft., converted from kBtu to kWh
df["heat_kwh_sqft"] = df["MFHTBTU"] * KBTU_TO_KWH / df["SQFT"]
df["cool_kwh_sqft"] = df["MFCLBTU"] * KBTU_TO_KWH / df["SQFT"]
print(f"Buildings used for training after cleaning: {len(df)}")

X = make_features(df.rename(columns={
    "PBA": "type_code", "PUBCLIM": "climate", "SQFT": "area", "BLDSHP": "shape",
    "GLSSPC": "glass_band", "NFLOOR": "floors", "FLCEILHT": "ceiling",
    "HEATP": "heated_pct", "COOLP": "cooled_pct"}))

# --- 3. Train and test one model per target -----------------------------------
models, accuracy = {}, {}
for target, label in [("heat_kwh_sqft", "Heating"), ("cool_kwh_sqft", "Cooling")]:
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy[label] = {"r2": round(r2_score(y_test, pred), 2),
                       "mae": round(mean_absolute_error(y_test, pred), 2)}
    print(f"{label}: R2 = {accuracy[label]['r2']}, average error = {accuracy[label]['mae']} kWh per sq. ft. per year")
    model.fit(X, y)                                 # final model uses all the data
    models[label] = model

# --- 4. Benchmarks: typical (median) energy use of similar buildings ---------
# "Similar" = same type, same climate, and similar share heated (for heating) or cooled (for cooling)
df["heat_grp"] = (df["HEATP"] >= 50).astype(int)
df["cool_grp"] = (df["COOLP"] >= 50).astype(int)
benchmarks = {}
for kind, target, grp in [("heat", "heat_kwh_sqft", "heat_grp"), ("cool", "cool_kwh_sqft", "cool_grp")]:
    detailed = df.groupby(["PBA", "PUBCLIM", grp])[target].agg(["median", "size"])
    by_climate = df.groupby(["PUBCLIM", grp])[target].median()
    benchmarks[kind] = {
        "detailed": {f"{p}_{c}_{g}": [round(m, 3), int(n)] for (p, c, g), (m, n) in detailed.iterrows()},
        "climate": {f"{c}_{g}": round(m, 3) for (c, g), m in by_climate.items()},
    }

joblib.dump({"models": models, "features": FEATURES}, "model.joblib", compress=3)   # compressed: under GitHub's 25 MB upload limit
with open("model_info.json", "w") as f:
    json.dump({"training_buildings": int(len(df)), "accuracy": accuracy, "benchmarks": benchmarks}, f, indent=1)
print("Saved model.joblib and model_info.json")
