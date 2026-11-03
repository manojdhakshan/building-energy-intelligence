"""
STEP 1 - Train the AI model.

Run once:  python train_model.py
Reads the UCI Energy Efficiency dataset (data/ENB2012_data.xlsx), trains one
Random Forest model for Heating Load and one for Cooling Load, and saves them as
"model.joblib" for the app to use. Reference values for the app are saved in
"model_info.json".
"""
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from building_logic import FEATURES, TARGETS

DATA_FILE = "data/ENB2012_data.xlsx"

# --- 1. Load the data (768 simulated buildings, 8 inputs, 2 targets) ---------
df = pd.read_excel(DATA_FILE).dropna(how="all").dropna(axis=1, how="all")
X = df[FEATURES]                  # X1 ... X8, in this exact order
print(f"Buildings in dataset: {len(df)}")

# --- 2. Same split as the project report: 80% train / 20% test, random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, df[TARGETS], test_size=0.2, random_state=42)

# --- 3. One Random Forest per target (no scaling needed for Random Forest) ---
models, accuracy = {}, {}
for target in TARGETS:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train[target])
    pred = model.predict(X_test)
    accuracy[target] = {"r2": round(r2_score(y_test[target], pred), 4),
                        "mae": round(mean_absolute_error(y_test[target], pred), 2)}
    print(f"{target}: test R2 = {accuracy[target]['r2']}, average error = {accuracy[target]['mae']} kWh/m2")
    models[target] = model         # the model evaluated above is the model the app uses

# --- 4. Reference values from the dataset for the app -----------------------
# The 12 building forms: Relative Compactness (X1) fixes X2-X5 in this dataset.
forms = (df.groupby("X1")[["X2", "X3", "X4", "X5"]].first().reset_index()
           .sort_values("X1", ascending=False).to_dict(orient="records"))
info = {
    "dataset_buildings": int(len(df)),
    "test_buildings": int(len(X_test)),
    "accuracy": accuracy,
    "building_forms": forms,
    # Smallest and largest value of each input in the dataset: inputs must stay inside these
    "ranges": {k: [float(df[k].min()), float(df[k].max())] for k in FEATURES},
    # Sorted loads of all 768 buildings: used to rank a prediction against the dataset
    "all_loads": {t: sorted(df[t].round(2).tolist()) for t in TARGETS},
    "quartiles": {t: [round(df[t].quantile(q), 2) for q in (0.25, 0.5, 0.75)] for t in TARGETS},
}

joblib.dump({"models": models, "features": FEATURES}, "model.joblib", compress=3)
with open("model_info.json", "w") as f:
    json.dump(info, f, indent=1)
print("Saved model.joblib and model_info.json")
