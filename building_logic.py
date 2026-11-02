"""
Shared logic used by both the training script and the app:
dropdown options, conversion of client inputs into model inputs,
result interpretation and design suggestions.
"""
import math

import numpy as np
import pandas as pd

KBTU_TO_KWH = 0.29307107

# (label shown to user, CBECS building activity code)
BUILDING_TYPES = [
    ("Institutional – School / college / university", 14),
    ("Institutional – Hospital (inpatient)", 16),
    ("Institutional – Clinic / outpatient health care", 8),
    ("Institutional – Public order & safety (police, fire station)", 7),
    ("Institutional – Religious building", 12),
    ("Institutional – Public assembly (auditorium, library, community hall)", 13),
    ("Commercial – Office", 2),
    ("Commercial – Retail shop / showroom", 25),
    ("Commercial – Shopping centre / mall", 23),
    ("Commercial – Grocery / food store", 6),
    ("Commercial – Restaurant / food service", 15),
    ("Commercial – Warehouse / storage (not refrigerated)", 5),
    ("Commercial – Service (repair shop, salon, dry cleaner)", 26),
    ("Residential-type – Hotel / hostel / dormitory", 18),
    ("Residential-type – Nursing home / assisted living", 17),
]

CLIMATES = [
    ("Hot or very hot (most of India, Middle East, South-East Asia)", 5),
    ("Warm (e.g. Bengaluru-like mild-warm, southern USA)", 4),
    ("Mixed mild (e.g. hill towns at moderate altitude, central USA)", 3),
    ("Cool (e.g. northern Europe, northern USA)", 2),
    ("Cold or very cold (e.g. Himalayan regions, Canada)", 1),
]

SHAPES = [
    ("Square", 1),
    ("Wide rectangle", 2),
    ("Narrow rectangle", 3),
    ("Rectangle or square with an interior courtyard", 4),
    ("H-shaped", 5),
    ("U-shaped", 6),
    ("E-shaped", 7),
    ("T-shaped", 8),
    ("L-shaped", 9),
    ("Cross (+) shaped", 10),
    ("Other shape", 11),
]

CONDITIONED_SHARE = [
    ("None (0%)", 0),
    ("Small part (about 25%)", 25),
    ("About half (50%)", 50),
    ("Most of it (about 75%)", 75),
    ("All of it (100%)", 100),
]

ORIENTATIONS = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]

TYPE_CODES = [code for _, code in BUILDING_TYPES]
SHAPE_CODES = [code for _, code in SHAPES]
FEATURES = (["area_log", "floors", "ceiling", "glass_band", "climate", "heated_pct", "cooled_pct"]
            + [f"type_{c}" for c in TYPE_CODES] + [f"shape_{c}" for c in SHAPE_CODES])


def make_features(df):
    """Turn a table with columns type_code, climate, area, shape, glass_band, floors, ceiling,
    heated_pct, cooled_pct into the exact columns the model was trained on."""
    X = pd.DataFrame({
        "area_log": np.log(df["area"].astype(float)),
        "floors": df["floors"].astype(float),
        "ceiling": df["ceiling"].astype(float),
        "glass_band": df["glass_band"].astype(float),
        "climate": df["climate"].astype(float),
        "heated_pct": df["heated_pct"].astype(float),
        "cooled_pct": df["cooled_pct"].astype(float),
    }, index=df.index)
    for c in TYPE_CODES:
        X[f"type_{c}"] = (df["type_code"] == c).astype(int)
    for c in SHAPE_CODES:
        X[f"shape_{c}"] = (df["shape"] == c).astype(int)
    return X[FEATURES]


def estimate_wall_area(area, floors, height):
    """Approximate outer wall area, assuming a square footprint."""
    footprint = area / floors
    perimeter = 4 * math.sqrt(footprint)
    return perimeter * height


def glass_percent_to_band(pct):
    """Convert window-to-wall % into the dataset's glass categories (1-6)."""
    for band, upper in [(1, 1), (2, 10), (3, 25), (4, 50), (5, 75)]:
        if pct <= upper:
            return band
    return 6


def prepare_inputs(b):
    """b = dict of the client's inputs. Returns (model feature row, derived values)."""
    wall = estimate_wall_area(b["area"], b["floors"], b["height"])
    glass_pct = min(100.0, 100 * b["glazing"] / wall) if wall > 0 else 0.0
    ceiling = max(7.0, min(50.0, b["height"] / b["floors"]))
    row = pd.DataFrame([{
        "type_code": b["type_code"], "climate": b["climate"], "area": b["area"],
        "shape": b["shape"], "glass_band": glass_percent_to_band(glass_pct),
        "floors": min(b["floors"], 15), "ceiling": ceiling,
        "heated_pct": b["heated_pct"], "cooled_pct": b["cooled_pct"],
    }])
    return make_features(row), {"wall_area": wall, "glass_pct": glass_pct, "floor_height": ceiling}


def predict(bundle, b):
    """Predicted yearly heating / cooling energy in kWh per sq. ft. and in total."""
    X, derived = prepare_inputs(b)
    heat = max(float(bundle["models"]["Heating"].predict(X)[0]), 0.0)
    cool = max(float(bundle["models"]["Cooling"].predict(X)[0]), 0.0)
    return {"heat": heat, "cool": cool, "heat_total": heat * b["area"], "cool_total": cool * b["area"], **derived}


def compare_level(value, typical):
    """Compare a prediction with the typical (median) value of similar buildings."""
    if typical <= 0.05:      # comparable buildings use almost none
        if value <= 0.1:
            return "Very low – same as similar buildings", "low"
        return "Higher than similar buildings", "high"
    ratio = value / typical
    if ratio < 0.85:
        return "Lower than similar buildings", "low"
    if ratio > 1.15:
        return "Higher than similar buildings", "high"
    return "About the same as similar buildings", "mid"


def get_benchmark(info, kind, type_code, climate, share_pct):
    """Typical energy use (kWh per sq. ft.) of similar buildings in the dataset.
    kind = "heat" or "cool"; share_pct = % of the building heated or cooled."""
    grp = int(share_pct >= 50)
    bench = info["benchmarks"][kind]
    found = bench["detailed"].get(f"{type_code}_{climate}_{grp}")
    if found and found[1] >= 15:         # enough similar buildings in the dataset
        return found[0], "buildings of the same type in the same climate"
    return bench["climate"][f"{climate}_{grp}"], "buildings in the same climate"


def suggestions(b, result, heat_level, cool_level):
    """Plain-language design suggestions based on inputs and predicted results."""
    tips = []
    hot_climate = b["climate"] >= 4
    cold_climate = b["climate"] <= 2
    glass = result["glass_pct"]
    orient = b["orientation"]

    # Glazing
    if glass > 40:
        tips.append(("Glazing", f"About {glass:.0f}% of the outer walls is glass, which is high. "
                     "Large glass areas let in a lot of heat in summer and lose heat in winter. "
                     "Consider reducing glass area, or using double-glazed / low-e (heat-reflecting) glass."))
    elif glass < 5:
        tips.append(("Glazing", f"Only about {glass:.0f}% of the outer walls is glass. "
                     "This is good for energy, but check there is enough daylight inside "
                     "so electric lights are not needed during the day."))
    else:
        tips.append(("Glazing", f"Glass covers about {glass:.0f}% of the outer walls, which is moderate. "
                     "Use shading (chajjas, louvres, overhangs) above windows to block direct sun."))

    # Orientation (not in the dataset – rule-of-thumb guidance)
    if hot_climate and orient in ("East", "West", "South-West", "North-West"):
        tips.append(("Orientation", f"The building faces {orient}. In hot climates, east and especially west "
                     "faces receive strong low-angle sun. Keep large windows away from these sides, "
                     "or protect them with vertical fins, trees or deep shading."))
    elif hot_climate:
        tips.append(("Orientation", f"Facing {orient} is generally favourable in hot climates. "
                     "North and south windows are easier to shade than east and west windows."))
    elif cold_climate and orient in ("South", "South-East", "South-West"):
        tips.append(("Orientation", f"Facing {orient} helps capture winter sun in a cold climate – "
                     "a good choice. Keep most windows on this side."))
    else:
        tips.append(("Orientation", "Place the main windows where they can be easily shaded in summer, "
                     "and avoid large unshaded glass on the west side."))

    # Shape / geometry
    if b["shape"] in (5, 6, 7, 8, 9, 10, 11):
        tips.append(("Building shape", "The building has a spread-out shape with more outer wall. "
                     "More outer wall means more surface for heat to pass through. "
                     "Good insulation and wall shading become more important."))
    elif b["shape"] == 3:
        tips.append(("Building shape", "A narrow rectangle gives good daylight and cross-ventilation. "
                     "Keep the long sides facing north and south where possible."))
    else:
        tips.append(("Building shape", "A compact shape has less outer wall for its size, which helps "
                     "reduce heat gain and heat loss."))

    # Courtyard
    if b["courtyard"] > 0:
        ratio = 100 * b["courtyard"] / (b["area"] / b["floors"])
        text = (f"The courtyard is about {ratio:.0f}% of the ground floor area. "
                "A shaded courtyard with plants or water can cool surrounding rooms and bring in daylight. ")
        if hot_climate:
            text += "In hot climates keep it partly shaded and use it for cross-ventilation."
        elif cold_climate:
            text += "In cold climates, an open courtyard adds outer wall area, so insulate the walls facing it."
        if b["shape"] != 4:
            text += (" (Tip: you entered a courtyard area – select 'Rectangle or square with an interior "
                     "courtyard' as the shape so the model accounts for it.)")
        tips.append(("Courtyard", text))
    elif hot_climate:
        tips.append(("Courtyard", "There is no courtyard. In hot climates, a small shaded courtyard can "
                     "improve natural ventilation and daylight."))

    # Results-based tips
    if cool_level == "high":
        tips.append(("Cooling", "Predicted cooling need is higher than similar buildings. "
                     "Consider a light-coloured or insulated roof, external shading, and efficient air-conditioning."))
    if heat_level == "high":
        tips.append(("Heating", "Predicted heating need is higher than similar buildings. "
                     "Consider better wall and roof insulation and sealing air leaks around windows and doors."))
    if b["height"] / b["floors"] > 14:
        tips.append(("Height", "Floors are quite tall. Tall spaces have more air to heat or cool – "
                     "consider ceiling fans or false ceilings in rooms that do not need the height."))
    return tips
