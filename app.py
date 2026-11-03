"""
STEP 2 - Run the app:  streamlit run app.py
"""
import json
import os
import subprocess
import sys

import joblib
import pandas as pd
import streamlit as st

from building_logic import (FEATURE_NAMES, FEATURES, GLAZING_AREAS, GLAZING_DISTRIBUTIONS, ORIENTATIONS, UNIT,
                            check_inputs, form_label, level, make_row, predict_rows, rank, suggestions)

st.set_page_config(page_title="Building Energy Intelligence", page_icon="🏢", layout="centered")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 900px;}
.hero {padding: 2.2rem 2rem; border-radius: 16px; color: white;
       background: linear-gradient(135deg, #0f5132 0%, #198754 60%, #20c997 100%);}
.hero h1 {color: white; margin: 0 0 .4rem 0; font-size: 2.3rem;}
.hero p {font-size: 1.1rem; opacity: .95; margin: 0;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 1.1rem 1.2rem; height: 100%;}
.card h4 {margin: 0 0 .3rem 0;}
.big {font-size: 2rem; font-weight: 700; margin: .2rem 0;}
.tag {display: inline-block; padding: .15rem .6rem; border-radius: 999px; font-size: .85rem; font-weight: 600;}
.low {background: #d1e7dd; color: #0f5132;}
.mid {background: #fff3cd; color: #664d03;}
.high {background: #f8d7da; color: #842029;}
</style>
""", unsafe_allow_html=True)


def read_model():
    """The saved model and its info, or None if missing or not trained on the 8 ENB2012 features."""
    try:
        bundle = joblib.load("model.joblib")
        with open("model_info.json") as f:
            info = json.load(f)
    except Exception:
        return None
    if bundle.get("features") != FEATURES or "building_forms" not in info:
        return None                       # e.g. files left from an older version of the app
    return bundle, info


@st.cache_resource(show_spinner="Loading the AI model (the first start also trains it)...")
def load_model():
    saved = read_model()
    if saved is None:
        subprocess.run([sys.executable, "train_model.py"], check=True)   # e.g. first start online
        saved = read_model()
    return saved


BUNDLE, INFO = load_model()
FORMS = INFO["building_forms"]
ACC = INFO["accuracy"]

PAGES = ["Home", "Building Details", "Prediction Results", "Compare Buildings"]
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Keep entered values when the user moves between pages
for k in list(st.session_state.keys()):
    if k.startswith(("main_", "cmpA_", "cmpB_")):
        st.session_state[k] = st.session_state[k]


def go(page):
    st.session_state.page = page


with st.sidebar:
    st.markdown("### 🏢 Building Energy Intelligence")
    for p in PAGES:
        st.button(p, key=f"nav_{p}", on_click=go, args=(p,), use_container_width=True,
                  type="primary" if st.session_state.page == p else "secondary")
    st.caption(f"Predictions come from a Random Forest model trained on the UCI Energy Efficiency dataset "
               f"(ENB2012) of {INFO['dataset_buildings']} simulated residential buildings.")


# ---------------------------------------------------------------- input form
def building_form(prefix, defaults=None):
    """Input fields for the 8 model features of one building. Returns the feature row."""
    start = {"form": 3, "orient": 3, "glazing": 0.25, "dist": 1, **(defaults or {})}
    for name, value in start.items():          # starting values, set only once
        st.session_state.setdefault(f"{prefix}_{name}", value)

    st.markdown("**Building form** – in this dataset the relative compactness fixes the surface, wall and roof "
                "area and the height (12 building forms, all with the same volume).")
    idx = st.selectbox(f"X1 · {FEATURE_NAMES['X1']}", range(len(FORMS)), key=f"{prefix}_form",
                       format_func=lambda i: form_label(FORMS[i]))
    f = FORMS[idx]
    c1, c2, c3, c4 = st.columns(4)
    c1.number_input(f"X2 · {FEATURE_NAMES['X2']}", value=f["X2"], disabled=True, key=f"show_{prefix}_x2_{idx}")
    c2.number_input(f"X3 · {FEATURE_NAMES['X3']}", value=f["X3"], disabled=True, key=f"show_{prefix}_x3_{idx}")
    c3.number_input(f"X4 · {FEATURE_NAMES['X4']}", value=f["X4"], disabled=True, key=f"show_{prefix}_x4_{idx}")
    c4.number_input(f"X5 · {FEATURE_NAMES['X5']}", value=f["X5"], disabled=True, key=f"show_{prefix}_x5_{idx}")

    st.markdown("**Orientation and windows**")
    c1, _ = st.columns(2)
    orient = c1.selectbox(f"X6 · {FEATURE_NAMES['X6']}", list(ORIENTATIONS), key=f"{prefix}_orient",
                          format_func=lambda o: f"{ORIENTATIONS[o]} (code {o})")
    c1, c2 = st.columns(2)
    with c1:
        glazing = st.selectbox(f"X7 · {FEATURE_NAMES['X7']}", list(GLAZING_AREAS), key=f"{prefix}_glazing",
                               format_func=lambda g: GLAZING_AREAS[g])
    # Dataset rule: no windows <-> distribution 0; with windows -> distribution 1 to 5
    dist_options = [0] if glazing == 0 else [1, 2, 3, 4, 5]
    if st.session_state[f"{prefix}_dist"] not in dist_options:
        st.session_state[f"{prefix}_dist"] = dist_options[0]
    with c2:
        dist = st.selectbox(f"X8 · {FEATURE_NAMES['X8']}", dist_options, key=f"{prefix}_dist",
                            format_func=lambda d: GLAZING_DISTRIBUTIONS[d])
    return make_row(f, orient, glazing, dist)


def analyse(row):
    r = predict_rows(BUNDLE, [row])[0]
    for t in ("Y1", "Y2"):
        r[f"{t}_text"], r[f"{t}_level"] = level(INFO, t, r[t])
        r[f"{t}_rank"] = rank(INFO, t, r[t])
    return r


def overall_message(r):
    levels = {r["Y1_level"], r["Y2_level"]}
    if "high" in levels:
        return "high", ("This building configuration indicates **relatively higher energy requirements** – "
                        "it is among the highest 25% of the dataset for at least one load. "
                        "The design suggestions below show which changes lower the predicted loads.")
    if levels == {"low"}:
        return "low", ("This building configuration indicates **relatively lower energy requirements** – "
                       "both loads are among the lowest 25% of the dataset.")
    return "mid", ("This building configuration indicates **moderate energy requirements** – "
                   "close to the middle of the dataset.")


def result_card(title, icon, target, r):
    median = INFO["quartiles"][target][1]
    st.markdown(f"""<div class="card"><h4>{icon} {title}</h4>
        <div class="big">{r[target]:.2f} <span style="font-size:1rem;font-weight:400">{UNIT}</span></div>
        <div style="margin:.5rem 0"><span class="tag {r[target + '_level']}">{r[target + '_text']}</span></div>
        <div>Higher than {r[target + '_rank']:.0f}% of the {INFO['dataset_buildings']} dataset buildings</div>
        <div style="font-size:.9rem;opacity:.8">Dataset median: {median:.2f} {UNIT}</div>
        </div>""", unsafe_allow_html=True)


def feature_table(row):
    shown = {FEATURE_NAMES[k]: row[k] for k in FEATURES}
    shown[FEATURE_NAMES["X6"]] = f"{ORIENTATIONS[row['X6']]} ({row['X6']})"
    shown[FEATURE_NAMES["X7"]] = f"{row['X7']:.2f}"
    shown[FEATURE_NAMES["X8"]] = GLAZING_DISTRIBUTIONS[row["X8"]]
    return pd.DataFrame({"Feature": [f"{k} · {FEATURE_NAMES[k]}" for k in FEATURES],
                         "Value": [str(v) for v in shown.values()]})


# ---------------------------------------------------------------- pages
page = st.session_state.page

if page == "Home":
    st.markdown("""<div class="hero"><h1>Building Energy Intelligence</h1>
        <p>AI-based prediction of building heating and cooling loads to support energy cost management
        and sustainable design decisions.</p></div>""", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card"><h4>1. Enter details</h4>Building form, orientation, glazing area and glazing distribution.</div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><h4>2. AI analyses</h4>A Random Forest model trained on the UCI Energy Efficiency dataset.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><h4>3. Get results</h4>Heating and cooling load, a clear rating and model-tested design suggestions.</div>', unsafe_allow_html=True)
    st.write("")
    st.button("Start Prediction  →", type="primary", on_click=go, args=("Building Details",), use_container_width=True)
    st.caption(f"The model was trained on the UCI Energy Efficiency dataset (ENB2012): {INFO['dataset_buildings']} "
               "simulated residential building configurations. Results are heating and cooling loads in kWh/m², "
               "not electricity bills or costs. Lower predicted loads point to designs with lower energy demand, "
               "which is how the tool supports energy cost management decisions.")

elif page == "Building Details":
    st.title("Building Details")
    st.write("Choose the building's design. The app uses exactly the 8 features the model was trained on (X1–X8).")
    row = building_form("main")
    if st.button("Predict Energy Performance", type="primary", use_container_width=True):
        problems = check_inputs(row, INFO)
        if problems:
            for p in problems:
                st.error(p)
        else:
            st.session_state.building = row
            st.session_state.result = analyse(row)
            go("Prediction Results")
            st.rerun()

elif page == "Prediction Results":
    st.title("Prediction Results")
    if "result" not in st.session_state:
        st.info("No prediction yet. Please enter your building details first.")
        st.button("Go to Building Details", type="primary", on_click=go, args=("Building Details",))
    else:
        row, r = st.session_state.building, st.session_state.result
        st.caption(f"Compactness {row['X1']:.2f} · height {row['X5']:.1f} m · faces {ORIENTATIONS[row['X6']]} · "
                   f"glazing {GLAZING_AREAS[row['X7']]} · distribution {GLAZING_DISTRIBUTIONS[row['X8']]}")
        c1, c2 = st.columns(2)
        with c1:
            result_card("Heating load", "🔥", "Y1", r)
        with c2:
            result_card("Cooling load", "❄️", "Y2", r)

        st.subheader("Energy performance interpretation")
        lvl, msg = overall_message(r)
        {"high": st.warning, "low": st.success, "mid": st.info}[lvl](msg)

        st.subheader("Suggestions to improve energy performance")
        tips = suggestions(BUNDLE, INFO, row, r)
        if tips:
            st.write("Each suggestion changes **one** design choice and is tested with the same AI model:")
            for group, text, saving, pct, p in tips:
                st.markdown(f"**{group}:** {text} → predicted heating {p['Y1']:.2f} and cooling {p['Y2']:.2f} {UNIT} "
                            f"(combined load about **{pct:.0f}% lower**).")
        else:
            st.success("No single change of building form, orientation, glazing area or distribution lowers the "
                       "predicted loads further – this is already one of the best configurations in the dataset.")

        with st.expander("Input features used by the model"):
            st.table(feature_table(row))

        with st.expander("About this prediction"):
            st.write(f"""
- **Dataset:** UCI Energy Efficiency dataset (ENB2012, Tsanas & Xifara, 2012) – **{INFO['dataset_buildings']} simulated**
  residential building configurations (12 building forms × 4 orientations × 16 glazing options), not measured buildings.
- **Model:** Random Forest Regression, one model for Heating Load (Y1) and one for Cooling Load (Y2),
  trained on 80% of the data (random_state = 42) and tested on the other {INFO['test_buildings']} buildings.
- **Accuracy on the test buildings:** Heating R² = {ACC['Y1']['r2']}, Cooling R² = {ACC['Y2']['r2']}
  (average error {ACC['Y1']['mae']} and {ACC['Y2']['mae']} {UNIT}).
- **Units:** {UNIT} – the heating and cooling load per square metre of floor area, as reported in the dataset's source paper.
- **"Low / Moderate / High"** compares the prediction with the {INFO['dataset_buildings']} dataset buildings
  (lowest 25%, middle 50%, highest 25%).
- The app does **not** calculate electricity bills or money savings. For energy cost management, a lower
  predicted load indicates a design that needs less heating and cooling energy.
""")
        c1, c2 = st.columns(2)
        c1.button("← Edit details", on_click=go, args=("Building Details",), use_container_width=True)
        c2.button("Compare with another design →", on_click=go, args=("Compare Buildings",), use_container_width=True)

elif page == "Compare Buildings":
    st.title("Compare Buildings")
    st.write("Enter two building designs to see which has the lower predicted heating and cooling load.")
    tab_a, tab_b = st.tabs(["Building A", "Building B"])
    with tab_a:
        a = building_form("cmpA")
    with tab_b:
        b = building_form("cmpB", {"form": 8, "orient": 4, "glazing": 0.10, "dist": 4})
    if st.button("Compare Energy Performance", type="primary", use_container_width=True):
        problems = [f"Building A: {p}" for p in check_inputs(a, INFO)] + [f"Building B: {p}" for p in check_inputs(b, INFO)]
        if problems:
            for p in problems:
                st.error(p)
        else:
            ra, rb = predict_rows(BUNDLE, [a, b])        # same model, same calculation for both
            st.session_state.compare = (ra, rb)
    if "compare" in st.session_state:
        ra, rb = st.session_state.compare
        st.subheader("Results")
        c1, c2 = st.columns(2)
        for col, name, r in [(c1, "Building A", ra), (c2, "Building B", rb)]:
            with col:
                st.markdown(f"#### {name}")
                st.metric(f"🔥 Heating load ({UNIT})", f"{r['Y1']:.2f}")
                st.metric(f"❄️ Cooling load ({UNIT})", f"{r['Y2']:.2f}")
                st.metric(f"Heating + cooling ({UNIT})", f"{r['Y1'] + r['Y2']:.2f}")
        ia, ib = ra["Y1"] + ra["Y2"], rb["Y1"] + rb["Y2"]
        if abs(ia - ib) < 0.05 * max(ia, ib):
            st.info("Both designs have **similar** predicted heating and cooling loads (less than 5% apart).")
        else:
            better, worse = ("A", "B") if ia < ib else ("B", "A")
            diff = 100 * abs(ia - ib) / max(ia, ib)
            st.success(f"**Building {better}** has about **{diff:.0f}% lower** combined heating and cooling load "
                       f"than Building {worse}.")
