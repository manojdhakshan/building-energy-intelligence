# Building Energy Intelligence

AI-Based Building Energy Performance Prediction for Energy Cost Management and Sustainable Decision-Making.

## How to run the app

**Easiest:** double-click **`Run App.bat`**. The app opens in your browser. Keep the black window open while you use it, and close it to stop the app.

**Or, from a terminal in this folder:**

```
python -m streamlit run app.py
```

Then open http://localhost:8501 in your browser.

To retrain the model (it is also trained automatically if `model.joblib` is missing):

```
python train_model.py
```

(On a new computer, install the tools once with `pip install -r requirements.txt`.)

## What is in this folder

| File | What it does |
|---|---|
| `data/ENB2012_data.xlsx` | UCI Energy Efficiency dataset: 768 simulated buildings, 8 inputs (X1–X8), 2 targets (Y1, Y2) |
| `train_model.py` | 80/20 split (random_state = 42), trains one Random Forest for Heating Load and one for Cooling Load, prints test R² |
| `model.joblib`, `model_info.json` | The trained models, and dataset reference values (12 building forms, load quartiles) used by the app |
| `building_logic.py` | Feature list and order, dropdown options, input checks, prediction, dataset comparison, model-tested suggestions |
| `app.py` | The web application (4 pages) |

## Dataset

- **Source:** UCI Machine Learning Repository, Energy Efficiency dataset (ENB2012): https://archive.ics.uci.edu/dataset/242/energy+efficiency
- **Paper:** Tsanas, A. & Xifara, A. (2012). Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools. *Energy and Buildings*, 49, 560–567.
- **768 simulated** residential building configurations (12 building forms × 4 orientations × 16 glazing options). They are simulated, not measured buildings.

| Feature | Name | Values in the dataset |
|---|---|---|
| X1 | Relative Compactness | 12 values, 0.62–0.98 |
| X2 | Surface Area (m²) | 514.5–808.5 |
| X3 | Wall Area (m²) | 245.0–416.5 |
| X4 | Roof Area (m²) | 110.25–220.5 |
| X5 | Overall Height (m) | 3.5 or 7.0 |
| X6 | Orientation | 2 = North, 3 = East, 4 = South, 5 = West |
| X7 | Glazing Area (share of floor area) | 0, 0.10, 0.25, 0.40 |
| X8 | Glazing Area Distribution | 0 = none, 1 = uniform, 2 = north, 3 = east, 4 = south, 5 = west |
| Y1 | Heating Load | kWh/m² |
| Y2 | Cooling Load | kWh/m² |

**How the inputs work in the app:** in this dataset each Relative Compactness value (X1) always goes with exactly one combination of Surface Area, Wall Area, Roof Area and Overall Height (X2–X5); there are 12 such building forms. The user chooses one of these 12 forms, and X2–X5 are filled in from the dataset (shown read-only). Orientation, glazing area and glazing distribution are chosen from the dataset's own values. A glazing area of 0% always has distribution 0, and the app enforces this. No conversion formulas are used, so all 8 values the model receives are exactly as in the dataset.

**Units:** the UCI page does not list units. The source paper reports heating and cooling load in kWh/m², and the app displays kWh/m². The app does not convert to kWh per sq. ft. or to totals, and does not calculate electricity bills or costs.

## Model results (80% train / 20% test, random_state = 42)

| Model | Heating Load R² | Cooling Load R² |
|---|---|---|
| Linear Regression | 0.9122 | 0.8932 |
| Decision Tree | 0.9963 | 0.9561 |
| Random Forest (project report) | 0.9977 | 0.9673 |
| Random Forest (model used in this app: 100 trees, random_state = 42) | 0.9977 | 0.9684 |

Linear Regression, Decision Tree and the Random Forest heating result are reproduced exactly by this code. The app's Random Forest cooling R² is 0.9684, against 0.9673 in the report, a 0.0011 difference. This is most likely due to a different number of trees or scikit-learn version. Random Forest is the best model in both cases.

## How the app's results are calculated

- **Heating and cooling load:** the Random Forest prediction for the 8 inputs, in kWh/m².
- **Low / Moderate / High:** compares the prediction with the 768 dataset buildings: below the 25th percentile = Low, above the 75th percentile = High, otherwise Moderate. "Higher than X% of the dataset buildings" is the share of the 768 buildings with a lower load. The dataset median is shown as a reference (Heating 18.95, Cooling 22.08 kWh/m²).
- **Design suggestions:** the app changes one design choice at a time (glazing area, glazing distribution, orientation, building form), predicts again with the same model, and shows the changes that lower heating + cooling load. It never suggests removing all windows.
- **Compare Buildings:** both designs go through the same model and the same calculation.

**Energy cost management:** the app predicts building loads, not bills. Managers can use the predicted loads to compare design options and choose those that need less heating and cooling energy, which lowers expected energy use and cost. Actual costs depend on tariffs, equipment efficiency and usage, which are outside the dataset.
