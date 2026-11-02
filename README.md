# Building Energy Intelligence

AI-Based Building Energy Performance Prediction for Energy Cost Management and Sustainable Decision-Making.

## How to run the app

**Easiest:** double-click **`Run App.bat`**. The app opens in your browser. Keep the black window open while you use it, and close it to stop the app.

**Or, from a terminal in this folder:**

```
python -m streamlit run app.py
```

Then open http://localhost:8501 in your browser.

If you ever replace the dataset, retrain the model first:

```
python train_model.py
```

(On a new computer, install the tools once with `pip install -r requirements.txt`.)

## What is in this folder

| File | What it does |
|---|---|
| `data/cbecs2018_final_public.csv` | The full dataset (U.S. EIA CBECS 2018 survey, 6,436 real buildings, 1,249 columns). Not uploaded to GitHub because of its size |
| `data/cbecs2018_selected_columns.csv` | The same 6,436 buildings with only the 12 columns the model uses. Training reads this file |
| `data/2018microdata_codebook.xlsx` | Explains every column in the dataset |
| `train_model.py` | Cleans the data, trains the Random Forest model, prints its accuracy |
| `model.joblib`, `model_info.json` | The trained model and typical values of similar buildings (created by `train_model.py`; the app creates them automatically if they are missing, e.g. on its first start online) |
| `building_logic.py` | Turns client inputs into model inputs; interpretation and design suggestions |
| `app.py` | The web application (4 pages) |

## Dataset and method (for your report)

- **Source:** U.S. Energy Information Administration, Commercial Buildings Energy Consumption Survey (CBECS) 2018, public microdata: https://www.eia.gov/consumption/commercial/data/2018/
- **Training data:** 5,044 buildings after cleaning. Removed: building types not offered in the app (vacant, laboratory, refrigerated warehouse, other), buildings whose climate was withheld, ceiling height over 50 ft, and buildings using district chilled water (that cooling is not included in the energy totals).
- **Targets:** yearly heating energy and cooling energy per sq. ft. (all fuels, from CBECS end-use estimates, converted from thousand Btu to kWh).
- **Algorithm:** Random Forest Regression (300 trees), one model for heating and one for cooling.
- **Accuracy on 20% test data:** Cooling R² = 0.52 (average error about 1.0 kWh/sq. ft./yr). Heating R² = 0.16 (average error about 4.3 kWh/sq. ft./yr). Heating is harder to predict because it depends strongly on the heating fuel and equipment, which the app does not ask for.

### How the client inputs map to the dataset

| Client input | Used by the model? | How |
|---|---|---|
| Building type | ✅ | CBECS building activity (15 types under Institutional / Commercial / Residential-type) |
| Total floor area | ✅ | CBECS square footage |
| Number of floors | ✅ | CBECS number of floors |
| Building height | ✅ | Height ÷ floors → floor height (CBECS floor-to-ceiling height) |
| Glazing area | ✅ | Converted to % of outer wall that is glass (wall area estimated assuming a square plan), then to the CBECS glass band |
| Building shape (incl. courtyard) | ✅ | CBECS building shape, which includes "Rectangle or square with an interior courtyard" |
| Climate | ✅ | CBECS ASHRAE climate zone group |
| Share heated / air-conditioned | ✅ | CBECS percent heated / percent cooled |
| Orientation | ❌ Not in dataset | Used only for design suggestions |
| Courtyard area | ❌ Not in dataset | Used only for design suggestions (the courtyard shape *is* a model input) |

### Limitations

- The data comes from U.S. buildings. Climate is matched by climate zone, but construction practices differ between countries.
- Pure residential houses and apartments are not in CBECS. "Residential-type" means hotels, hostels, dormitories and nursing homes.
- The results are predicted **energy use in kWh**, not electricity bills or costs.
- "Higher / lower than similar buildings" compares the prediction with the median of dataset buildings of the same type and climate, with a similar share heated or cooled.
