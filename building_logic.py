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


def form_label(f):
    return (f"Compactness {f['X1']:.2f} – height {f['X5']:.1f} m, "
            f"surface {f['X2']:.1f} m², wall {f['X3']:.1f} m², roof {f['X4']:.2f} m²")


def make_row(form, orientation, glazing_area, distribution):
    """One building as the 8 model features, in training order."""
    return {"X1": form["X1"], "X2": form["X2"], "X3": form["X3"], "X4": form["X4"], "X5": form["X5"],
            "X6": orientation, "X7": glazing_area, "X8": distribution}


def check_inputs(row, info):
    """Return a list of problems (empty = valid). Only value combinations found in the dataset are allowed."""
    problems = []
    if not any(all(abs(row[k] - f[k]) < 1e-9 for k in ("X1", "X2", "X3", "X4", "X5")) for f in info["building_forms"]):
        problems.append("Relative compactness, surface, wall and roof area and height must match one of the 12 building forms.")
    if row["X6"] not in ORIENTATIONS:
        problems.append("Orientation must be North, East, South or West.")
    if row["X7"] not in GLAZING_AREAS:
        problems.append("Glazing area must be 0%, 10%, 25% or 40%.")
    if row["X8"] not in GLAZING_DISTRIBUTIONS:
        problems.append("Glazing distribution must be a value from 0 to 5.")
    if (row["X7"] == 0) != (row["X8"] == 0):
        problems.append("Glazing distribution must be '0 – No glazing' when glazing area is 0%, "
                        "and between 1 and 5 when there are windows.")
    return problems


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
    and keep the changes that lower the total of heating + cooling load."""
    base = result["Y1"] + result["Y2"]
    options = []

    for glaze in GLAZING_AREAS:
        if 0 < glaze < row["X7"]:          # never suggest removing all windows (daylight is needed)
            options.append(("Glazing area", f"Reduce glazing area from {GLAZING_AREAS[row['X7']]} to {GLAZING_AREAS[glaze]}",
                            {**row, "X7": glaze}))
    if row["X7"] > 0:
        for dist in range(1, 6):
            if dist != row["X8"]:
                options.append(("Glazing distribution", f"Place windows as: {GLAZING_DISTRIBUTIONS[dist][4:]}",
                                {**row, "X8": dist}))
    for o, name in ORIENTATIONS.items():
        if o != row["X6"]:
            options.append(("Orientation", f"Orient the building to face {name}", {**row, "X6": o}))
    for f in info["building_forms"]:
        if f["X1"] != row["X1"]:
            options.append(("Building form", f"Use the building form with compactness {f['X1']:.2f} "
                            f"(height {f['X5']:.1f} m)", make_row(f, row["X6"], row["X7"], row["X8"])))

    preds = predict_rows(bundle, [o[2] for o in options])
    best = {}
    for (group, text, _), p in zip(options, preds):
        saving = base - (p["Y1"] + p["Y2"])
        if saving > 0.05 and (group not in best or saving > best[group][1]):
            best[group] = (text, saving, p)
    tips = [(group, text, saving, 100 * saving / base, p) for group, (text, saving, p) in best.items()]
    return sorted(tips, key=lambda t: -t[2])
