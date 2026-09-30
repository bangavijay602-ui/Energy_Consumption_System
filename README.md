# Energy Consumption Forecasting Using Time-Series Machine Learning Models

A modular, production-ready time-series machine learning training and inference pipeline designed to ingest heterogeneous energy consumption datasets, automatically discover schemas and sampling cadence, validate and engineer leakage-free dynamic features, benchmark candidate models (Naive Baseline, Random Forest, XGBoost, SARIMA), select and retrain the optimal model, and perform true recursive multi-step future forecasting.

---

## Architecture & End-to-End Workflow

```
                    Raw User CSV Upload
                            ↓
               [ intelligence/profiler.py ]
               Dataset Profiling & Telemetry
                            ↓
            [ intelligence/schema_detector.py ]
        Timestamp, Target & Weather Role Discovery
                            ↓
             [ preprocessing/normalizer.py ]
             Canonical Schema Standardization
                            ↓
              [ intelligence/validator.py ]
            Data Validation & Anomaly Checks
                            ↓
         [ intelligence/frequency_detector.py ]
          Cadence & Frequency-Aware Lag Rules
                            ↓
             [ preprocessing/features.py ]
         Calendar, Cyclic, Lag & Rolling Features
                  (Zero Lookahead Bias)
                            ↓
         Chronological Split (70% / 15% / 15%)
         [Train Set]   [Validation Set]   [Test Set]
                            ↓
               Candidate Model Competition
         ┌────────────┬─────────────┬────────────┐
         ↓            ↓             ↓            ↓
    [ Baseline ] [ RandomForest ] [ XGBoost ] [ SARIMA ]
         │            │             │            │
         └────────────┴──────┬──────┴────────────┘
                             ↓
                 [ forecasting/evaluator.py ]
                 MAE, RMSE, Safe-MAPE Leaderboard
                             ↓
              [ forecasting/model_selector.py ]
               Validation Metric Model Selection
                             ↓
                Retrain on Historical Data
                    (Train + Validation)
                             ↓
                   Held-Out Test Scoring
                             ↓
             [ forecasting/future_forecaster.py ]
          Recursive Step-by-Step Future Generation
             t+1 ──► t+2 ──► t+3 ──► ... ──► t+H
                             ↓
           Serialized Artifacts (.joblib & .json)
                             ↓
                  API / Frontend Dashboard
```

---

## Directory Structure

```
energy_forecasting/
│
├── intelligence/
│   ├── __init__.py
│   ├── profiler.py              # In-depth dataset profiling (rows, cols, nulls, dtypes, summary stats)
│   ├── schema_detector.py       # Heuristic & parseability scoring for timestamp, target, and weather features
│   ├── frequency_detector.py    # Auto-detection of sampling frequency (15m, 30m, hourly, daily, weekly, irregular)
│   └── validator.py             # Validation checks (ordering, nulls, negative values, constant series, spikes)
│
├── preprocessing/
│   ├── __init__.py
│   ├── normalizer.py            # Canonical schema mapping (timestamp, consumption, temp, hum, etc.)
│   └── features.py              # Frequency-aware calendar, cyclic, lag, and rolling feature engineering
│
├── forecasting/
│   ├── __init__.py
│   ├── baseline.py              # Naive & Seasonal Naive baseline forecaster
│   ├── random_forest.py         # Tuned Random Forest time-series forecaster
│   ├── xgboost_model.py         # XGBoost time-series forecaster
│   ├── sarima.py                # SARIMA state-space forecaster
│   ├── evaluator.py             # Metrics calculation (MAE, RMSE, epsilon-safe MAPE, comparison table)
│   ├── model_selector.py        # Validation-based selection, retraining on history, test evaluation
│   └── future_forecaster.py     # Recursive multi-step future forecasting engine
│
├── data/
│   ├── sample_energy.csv        # Standard hourly energy dataset
│   ├── dataset1.csv             # DateTime, Energy_Usage (Hourly)
│   ├── dataset2.csv             # timestamp, load (30-minute)
│   ├── dataset3.csv             # reading_time, power_consumption, temperature, humidity (15-minute)
│   ├── dataset4.csv             # date, electricity_demand (Daily)
│   ├── dataset5.csv             # time, value, temp (Hourly)
│   └── invalid.csv              # Name, Age, Marks, City (Rejection test case)
│
├── models/                      # Serialized models (.joblib) and metadata JSON files
├── tests/
│   ├── __init__.py
│   ├── test_intelligence.py     # Unit tests for intelligence layer
│   ├── test_preprocessing.py    # Unit tests for normalization and feature engineering
│   ├── test_forecasting.py      # Unit tests for models, evaluator, and recursive forecaster
│   └── test_pipeline.py         # End-to-end integration tests across all sample datasets
│
├── generate_data.py             # Synthetic dataset generator for test cases
├── pipeline.py                  # End-to-end EnergyForecastingPipeline class
├── main.py                      # CLI entrypoint and demonstration runner
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## Key Features & Engineering Highlights

### 1. Robust Dataset Intelligence
- **No Hardcoded Column Assumptions**: Automatically scans column names and tests datetime parseability using regex patterns and pandas conversion rates.
- **Ambiguity Guard**: If multiple columns qualify with near-identical confidence scores, the pipeline raises an `AmbiguousSchemaError` with top candidates rather than silently guessing.
- **Frequency Intelligence**: Dynamically detects sampling intervals (`15-minute`, `30-minute`, `hourly`, `daily`, `weekly`, or `irregular`) and configures lag and rolling windows tailored to that cadence.
- **Safe Anomaly Handling**: Flags high demand spikes without silently dropping them, preserving critical grid peak events.

### 2. Zero-Leakage Feature Engineering
- **Strictly Shifted Rolling Statistics**: All rolling aggregations (`rolling_mean`, `rolling_std`, `rolling_min`, `rolling_max`) are calculated on `.shift(1)` to ensure that the current step's target value is never exposed during feature creation.
- **Sinusoidal Cyclic Encodings**: Hours, days of the week, and months are transformed into continuous `sin`/`cos` cyclic coordinates.
- **Optional Environmental Variables**: Automatically integrates weather variables (`temperature`, `humidity`, `wind_speed`, `pressure`, `holiday`) if present.

### 3. Chronological Train / Val / Test Split
- Time-series splits are strictly sequential (70% Train, 15% Validation, 15% Test). Random cross-validation or shuffling is never used.

### 4. Objective Model Selection
- Benchmarks candidates against a mandatory Naive Baseline:
  1. **Naive & Seasonal Baseline**
  2. **Random Forest Regressor**
  3. **Extreme Gradient Boosting (XGBoost)**
  4. **Seasonal ARIMA (SARIMA)**
- Selects the winning model based on measured validation MAE, then retrains that model on all combined historical data ($Train + Validation$) before evaluating on the held-out Test set.

### 5. Genuine Multi-Step Recursive Future Forecasting
- Multi-step forecasts ($t+1, t+2, \dots, t+H$) are generated recursively.
- For each step $k$, dynamic lag and rolling features are re-calculated using prior predictions and historical observations without relying on test set labels or future ground truth.

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.14)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage Instructions

### 1. Run Complete Automated Demo
Run the pipeline across all 5 distinct test datasets and verify rejection of invalid CSVs:
```bash
python main.py --demo
```

### 2. Run on a Custom CSV File
```bash
python main.py --data "data/dataset3.csv" --horizon 24 --output-dir "models"
```

### 3. Run Pytest Test Suite
```bash
pytest -v
```

---

## Programmatic Python Usage

```python
from pipeline import EnergyForecastingPipeline

# Instantiate the pipeline
pipeline = EnergyForecastingPipeline(models_dir="models", default_horizon=24)

# Execute on CSV file or pandas DataFrame
result = pipeline.run(
    data_source="data/dataset1.csv",
    horizon=24,
    model_tag="building_meter_01",
)

# Access structured results
print("Selected Model:", result.selected_model)
print("Test MAE:", result.test_metrics["mae"])
print("Test RMSE:", result.test_metrics["rmse"])
print("Test MAPE:", result.test_metrics["mape"])
print("Forecast Points:", len(result.forecast))
print("Model Path:", result.model_path)
```

---

## FastAPI Backend Integration Example

Easily expose the pipeline via a REST endpoint:

```python
from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io
from pipeline import EnergyForecastingPipeline

app = FastAPI(title="Energy Forecasting API")
pipeline = EnergyForecastingPipeline()

@app.post("/forecast")
async def upload_and_forecast(file: UploadFile = File(...), horizon: int = 24):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    output = pipeline.run(data_source=df, horizon=horizon)
    return output.to_dict()
```
