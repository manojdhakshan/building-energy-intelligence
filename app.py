"""
STEP 2 - Run the app:  streamlit run app.py
"""
import json
import os
import subprocess
import sys

import joblib
import streamlit as st

from building_logic import (BUILDING_TYPES, CLIMATES, CONDITIONED_SHARE, ORIENTATIONS, SHAPES,
                            compare_level, get_benchmark, predict, suggestions)

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


@st.cache_resource(show_spinner="Training the AI model for the first time (about a minute)...")
def load_model():
    if not (os.path.exists("model.joblib") and os.path.exists("model_info.json")):
        subprocess.run([sys.executable, "train_model.py"], check=True)   # e.g. first start online
    with open("model_info.json") as f:
        return joblib.load("model.joblib"), json.load(f)


BUNDLE, INFO = load_model()

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
    st.caption("Predictions come from a Random Forest model trained on the U.S. CBECS 2018 survey "
               f"of {INFO['training_buildings']:,} real buildings.")


# ---------------------------------------------------------------- input form
def building_form(prefix, defaults=None):
    """Input fields for one building. Returns a dict of values."""
    labels = lambda opts: [o[0] for o in opts]
    code_of = lambda opts, label: dict(opts)[label]
    start = {"type": BUILDING_TYPES[0][0], "area": 2000, "floors": 1, "height": 12.0, "orient": "East",
             "climate": CLIMATES[0][0], "shape": SHAPES[3][0], "glazing": 100, "courtyard": 300,
             "cooled": CONDITIONED_SHARE[4][0], "heated": CONDITIONED_SHARE[0][0], **(defaults or {})}
    for name, value in start.items():          # starting values, set only once
        st.session_state.setdefault(f"{prefix}_{name}", value)

    c1, c2 = st.columns(2)
    with c1:
        btype = st.selectbox("Building type", labels(BUILDING_TYPES), key=f"{prefix}_type")
        area = st.number_input("Total floor area (sq. ft.)", min_value=1001, max_value=2_000_000,
                               step=100, key=f"{prefix}_area",
                               help="Total area of all floors added together. The dataset covers buildings above 1,000 sq. ft.")
        floors = st.number_input("Number of floors", min_value=1, max_value=15, step=1, key=f"{prefix}_floors")
        height = st.number_input("Total building height (ft.)", min_value=7.0, max_value=300.0,
                                 step=1.0, key=f"{prefix}_height")
        orientation = st.selectbox("Main facade faces", ORIENTATIONS, key=f"{prefix}_orient")
    with c2:
        climate = st.selectbox("Climate of the location", labels(CLIMATES), key=f"{prefix}_climate")
        shape = st.selectbox("Building shape (ground-floor plan)", labels(SHAPES), key=f"{prefix}_shape")
        glazing = st.number_input("Glazing area – total window & glass door area (sq. ft.)", min_value=0,
                                  max_value=1_000_000, step=10, key=f"{prefix}_glazing")
        courtyard = st.number_input("Courtyard area (sq. ft.)", min_value=0, max_value=1_000_000,
                                    step=10, key=f"{prefix}_courtyard", help="Enter 0 if there is no courtyard.")
        cooled = st.selectbox("How much of the building is air-conditioned?", labels(CONDITIONED_SHARE),
                              key=f"{prefix}_cooled")
        heated = st.selectbox("How much of the building is heated in winter?", labels(CONDITIONED_SHARE),
                              key=f"{prefix}_heated")

    return {"type_label": btype, "type_code": code_of(BUILDING_TYPES, btype), "area": float(area),
            "floors": int(floors), "height": float(height), "orientation": orientation,
            "climate_label": climate, "climate": code_of(CLIMATES, climate),
            "shape_label": shape, "shape": code_of(SHAPES, shape),
            "glazing": float(glazing), "courtyard": float(courtyard),
            "cooled_pct": code_of(CONDITIONED_SHARE, cooled), "heated_pct": code_of(CONDITIONED_SHARE, heated)}


def check_inputs(b):
    """Return a list of problems with the inputs (empty list = OK)."""
    problems = []
    footprint = b["area"] / b["floors"]
    if b["height"] / b["floors"] < 7:
        problems.append("The building height is too low for that number of floors (less than 7 ft per floor).")
    if b["courtyard"] >= footprint:
        problems.append("The courtyard area must be smaller than the ground-floor area "
                        f"(about {footprint:,.0f} sq. ft.).")
    return problems


def analyse(b):
    r = predict(BUNDLE, b)
    r["typ_heat"], peer_h = get_benchmark(INFO, "heat", b["type_code"], b["climate"], b["heated_pct"])
    r["typ_cool"], peer_c = get_benchmark(INFO, "cool", b["type_code"], b["climate"], b["cooled_pct"])
    r["peer"] = peer_h if peer_h == peer_c else "buildings in the same climate"
    r["heat_text"], r["heat_level"] = compare_level(r["heat"], r["typ_heat"])
    r["cool_text"], r["cool_level"] = compare_level(r["cool"], r["typ_cool"])
    return r


def overall_message(r):
    levels = {r["heat_level"], r["cool_level"]}
    if "high" in levels:
        return "high", ("This building configuration indicates **relatively higher energy requirements** "
                        f"than typical {r['peer']}. The design suggestions below can help reduce them.")
    if levels == {"low"} or levels == {"low", "mid"}:
        return "low", ("This building configuration indicates **relatively lower energy requirements** "
                       f"than typical {r['peer']}. A good starting point – the suggestions below can improve it further.")
    return "mid", (f"This building configuration indicates **energy requirements similar to** typical {r['peer']}.")


def result_card(title, icon, value, total, level, text, typical):
    st.markdown(f"""<div class="card"><h4>{icon} {title}</h4>
        <div class="big">{value:.2f} <span style="font-size:1rem;font-weight:400">kWh per sq. ft. per year</span></div>
        <div>≈ {total:,.0f} kWh per year for the whole building</div>
        <div style="margin:.5rem 0"><span class="tag {level}">{text}</span></div>
        <div style="font-size:.9rem;opacity:.8">Typical for similar buildings: {typical:.2f} kWh per sq. ft.</div>
        </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- pages
page = st.session_state.page

if page == "Home":
    st.markdown("""<div class="hero"><h1>Building Energy Intelligence</h1>
        <p>AI-based prediction of building heating and cooling energy for better energy cost management
        and sustainable design decisions.</p></div>""", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card"><h4>1. Enter details</h4>Area, height, glazing, shape, courtyard and building type.</div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><h4>2. AI analyses</h4>A Random Forest model trained on real building survey data.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><h4>3. Get results</h4>Heating and cooling energy, a clear rating and design suggestions.</div>', unsafe_allow_html=True)
    st.write("")
    st.button("Start Prediction  →", type="primary", on_click=go, args=("Building Details",), use_container_width=True)
    st.caption("Results show predicted energy use (kWh), not electricity bills. They are estimates to support "
               "early design decisions and comparison between options.")

elif page == "Building Details":
    st.title("Building Details")
    st.write("Enter the basic details of the building. All fields can be changed later.")
    b = building_form("main")
    if st.button("Predict Energy Performance", type="primary", use_container_width=True):
        problems = check_inputs(b)
        if problems:
            for p in problems:
                st.error(p)
        else:
            st.session_state.building = b
            st.session_state.result = analyse(b)
            go("Prediction Results")
            st.rerun()

elif page == "Prediction Results":
    st.title("Prediction Results")
    if "result" not in st.session_state:
        st.info("No prediction yet. Please enter your building details first.")
        st.button("Go to Building Details", type="primary", on_click=go, args=("Building Details",))
    else:
        b, r = st.session_state.building, st.session_state.result
        st.caption(f"{b['type_label']} · {b['area']:,.0f} sq. ft. · {b['floors']} floor(s) · "
                   f"{b['height']:.0f} ft high · faces {b['orientation']} · {b['shape_label']}")
        c1, c2 = st.columns(2)
        with c1:
            result_card("Heating energy", "🔥", r["heat"], r["heat_total"], r["heat_level"], r["heat_text"], r["typ_heat"])
        with c2:
            result_card("Cooling energy", "❄️", r["cool"], r["cool_total"], r["cool_level"], r["cool_text"], r["typ_cool"])

        st.subheader("Energy performance interpretation")
        level, msg = overall_message(r)
        {"high": st.warning, "low": st.success, "mid": st.info}[level](msg)
        st.write(f"Window glass covers about **{r['glass_pct']:.0f}%** of the outer walls "
                 "(estimated from your glazing area and building size).")

        st.subheader("Suggestions to improve energy performance")
        for title, text in suggestions(b, r, r["heat_level"], r["cool_level"]):
            st.markdown(f"**{title}:** {text}")

        with st.expander("About this prediction"):
            acc = INFO["accuracy"]
            st.write(f"""
- The model was trained on **{INFO['training_buildings']:,} real non-residential and residential-type buildings**
  from the U.S. Commercial Buildings Energy Consumption Survey (CBECS 2018).
- Model inputs: building type, floor area, floors, floor height, glass percentage, building shape (incl. courtyard),
  climate, and share of the building heated / air-conditioned.
- **Orientation and courtyard size are not in the dataset**. They are used only for the design suggestions.
- Accuracy on buildings the model had not seen: cooling R² = {acc['Cooling']['r2']}, heating R² = {acc['Heating']['r2']}.
  Cooling predictions are more reliable than heating predictions, because heating also depends on the
  fuel and equipment used, which are not asked here.
- Results are energy estimates (kWh), not electricity bills. Use them to compare design options.
""")
        c1, c2 = st.columns(2)
        c1.button("← Edit details", on_click=go, args=("Building Details",), use_container_width=True)
        c2.button("Compare with another design →", on_click=go, args=("Compare Buildings",), use_container_width=True)

elif page == "Compare Buildings":
    st.title("Compare Buildings")
    st.write("Enter two building designs to see which needs less heating and cooling energy.")
    tab_a, tab_b = st.tabs(["Building A", "Building B"])
    with tab_a:
        a = building_form("cmpA")
    with tab_b:
        bb = building_form("cmpB", {"glazing": 400, "orient": "West", "courtyard": 0, "shape": SHAPES[0][0]})
    if st.button("Compare Energy Performance", type="primary", use_container_width=True):
        problems = [f"Building A: {p}" for p in check_inputs(a)] + [f"Building B: {p}" for p in check_inputs(bb)]
        if problems:
            for p in problems:
                st.error(p)
        else:
            st.session_state.compare = (a, analyse(a), bb, analyse(bb))
    if "compare" in st.session_state:
        a, ra, bb, rb = st.session_state.compare
        st.subheader("Results")
        c1, c2 = st.columns(2)
        for col, name, r in [(c1, "Building A", ra), (c2, "Building B", rb)]:
            with col:
                st.markdown(f"#### {name}")
                st.metric("🔥 Heating (kWh per sq. ft. per year)", f"{r['heat']:.2f}")
                st.metric("❄️ Cooling (kWh per sq. ft. per year)", f"{r['cool']:.2f}")
                st.metric("Total heating + cooling (kWh per year)", f"{r['heat_total'] + r['cool_total']:,.0f}")
        ia, ib = ra["heat"] + ra["cool"], rb["heat"] + rb["cool"]
        if abs(ia - ib) < 0.05 * max(ia, ib, 0.01):
            st.info("Both designs have **similar** predicted heating and cooling energy per sq. ft.")
        else:
            better, worse = ("A", "B") if ia < ib else ("B", "A")
            diff = 100 * abs(ia - ib) / max(ia, ib)
            st.success(f"**Building {better}** needs about **{diff:.0f}% less** heating and cooling energy "
                       f"per sq. ft. than Building {worse}.")
