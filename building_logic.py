"""
Shared logic used by both the training script and the app:
the 8 model features, dropdown options, input checks, prediction,
comparison with the dataset and design suggestions.

Feature meanings and codes follow the dataset's source paper:
Tsanas, A. & Xifara, A. (2012). Accurate quantitative estimation of energy performance
of residential buildings using statistical machine learning tools. Energy and Buildings, 49, 560-567.
"""
import bisect

import pandas as pd

# Model features, in the exact order used for training
FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
TARGETS = ["Y1", "Y2"]

FEATURE_NAMES = {
    "X1": "Relative Compactness",
    "X2": "Surface Area (m²)",
    "X3": "Wall Area (m²)",
    "X4": "Roof Area (m²)",
    "X5": "Overall Height (m)",
    "X6": "Orientation",
    "X7": "Glazing Area (share of floor area)",
    "X8": "Glazing Area Distribution",
}
TARGET_NAMES = {"Y1": "Heating Load", "Y2": "Cooling Load"}
UNIT = "kWh/m²"

ORIENTATIONS = {2: "North", 3: "East", 4: "South", 5: "West"}
GLAZING_AREAS = {0.0: "0% (no windows)", 0.10: "10%", 0.25: "25%", 0.40: "40%"}
GLAZING_DISTRIBUTIONS = {
    0: "0 – No glazing",
    1: "1 – Uniform (25% on each side)",
    2: "2 – Mostly North (55% north, 15% on other sides)",
    3: "3 – Mostly East (55% east, 15% on other sides)",
    4: "4 – Mostly South (55% south, 15% on other sides)",
    5: "5 – Mostly West (55% west, 15% on other sides)",
}


NUMERIC_FEATURES = ["X1", "X2", "X3", "X4", "X5", "X7"]


def check_inputs(row, info):
    """Return a list of problems (empty = valid). Values must be filled in and lie inside the dataset's range,
    because a Random Forest cannot predict reliably outside the values it was trained on."""
    problems = []
    for k in FEATURES:
        if row.get(k) is None:
            problems.append(f"Please enter {k} · {FEATURE_NAMES[k]}.")
    if problems:
        return problems
    for k in NUMERIC_FEATURES:
        lo, hi = info["ranges"][k]
        if not lo <= row[k] <= hi:
            problems.append(f"{k} · {FEATURE_NAMES[k]} must be between {lo:g} and {hi:g} (the range in the dataset).")
    if row["X6"] not in ORIENTATIONS:
        problems.append("Orientation must be North, East, South or West.")
    if row["X8"] not in GLAZING_DISTRIBUTIONS:
        problems.append("Glazing distribution must be a value from 0 to 5.")
    if (row["X7"] == 0) != (row["X8"] == 0):
        problems.append("Glazing distribution must be '0 – No glazing' when glazing area is 0, "
                        "and between 1 and 5 when there are windows.")
    return problems


def reliability_notes(row, info):
    """Plain-language notes on how close the entered building is to the simulated buildings in the dataset."""
    notes = []
    surface = row["X3"] + 2 * row["X4"]        # walls + roof + floor; holds for every building in the dataset
    if abs(row["X2"] - surface) > 0.01 * surface:
        notes.append(f"Surface area is usually wall area + roof area + floor area. With your values that is "
                     f"{surface:.1f} m² (wall + 2 × roof), but you entered {row['X2']:.1f} m². Please check the areas.")
    if row["X5"] not in (3.5, 7.0):
        notes.append("The dataset only contains buildings 3.5 m or 7.0 m high, so the model treats heights below "
                     "5.25 m like 3.5 m and heights above 5.25 m like 7.0 m.")
    if row["X7"] not in GLAZING_AREAS:
        notes.append("The dataset only contains glazing areas of 0, 0.10, 0.25 and 0.40, so values in between are "
                     "estimated from the nearest of these levels.")
    if not any(all(abs(row[k] - f[k]) < 1e-6 for k in ("X1", "X2", "X3", "X4", "X5")) for f in info["building_forms"]):
        notes.append("This geometry is not one of the 12 simulated building forms in the dataset, so the result is "
                     "an estimate based on the most similar simulated buildings and is less certain.")
    return notes


def predict_rows(bundle, rows):
    """Predict Heating and Cooling Load (kWh/m²) for a list of buildings."""
    X = pd.DataFrame(rows)[bundle["features"]]          # same names and order as training
    return [{"Y1": float(h), "Y2": float(c)}
            for h, c in zip(bundle["models"]["Y1"].predict(X), bundle["models"]["Y2"].predict(X))]


def rank(info, target, value):
    """Percentage of the 768 dataset buildings with a lower load than this value."""
    loads = info["all_loads"][target]
    return 100 * bisect.bisect_left(loads, value) / len(loads)


def level(info, target, value):
    """Compare with the dataset quartiles: lowest 25% / middle 50% / highest 25%."""
    q1, _, q3 = info["quartiles"][target]
    if value < q1:
        return "Low compared with the dataset", "low"
    if value > q3:
        return "High compared with the dataset", "high"
    return "Moderate compared with the dataset", "mid"


def suggestions(bundle, info, row, result):
    """Design suggestions tested with the same model: change one design choice at a time
    (glazing area, glazing distribution, orientation) and keep the changes that lower heating + cooling load."""
    base = result["Y1"] + result["Y2"]
    options = []

    for glaze in GLAZING_AREAS:
        if 0 < glaze < row["X7"] - 1e-9:   # never suggest removing all windows (daylight is needed)
            options.append(("Glazing area", f"Reduce glazing area from {row['X7']:.2f} to {glaze:.2f} of floor area",
                            {**row, "X7": glaze}))
    if row["X7"] > 0:
        for dist in range(1, 6):
            if dist != row["X8"]:
                options.append(("Glazing distribution", f"Place windows as: {GLAZING_DISTRIBUTIONS[dist][4:]}",
                                {**row, "X8": dist}))
    for o, name in ORIENTATIONS.items():
        if o != row["X6"]:
            options.append(("Orientation", f"Orient the building to face {name}", {**row, "X6": o}))

    preds = predict_rows(bundle, [o[2] for o in options])
    best = {}
    for (group, text, _), p in zip(options, preds):
        saving = base - (p["Y1"] + p["Y2"])
        if saving > 0.05 and (group not in best or saving > best[group][1]):
            best[group] = (text, saving, p)
    tips = [(group, text, saving, 100 * saving / base, p) for group, (text, saving, p) in best.items()]
    return sorted(tips, key=lambda t: -t[2])
