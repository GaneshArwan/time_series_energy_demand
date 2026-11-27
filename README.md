# Time Series Forecasting: PJM East Electricity Demand

Predicting next-day (24-hour-ahead) electricity load across the PJM East (`PJME`) transmission zone and quantifying what drives forecast errors across seasons, days of the week, and peak hours.

## Dataset & Cleaning
- **Source:** [PJM Hourly Energy Consumption (Kaggle)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) (`PJME_hourly.csv`).
- **Span:** `2002-01-01 01:00:00` to `2018-08-03 00:00:00` (`145,366` raw hourly records).
- **Preprocessing (`src/data.py`):** Sorted chronologically, merged `4` Daylight Saving Time (DST) fall-back duplicate hours by mean, reindexed to a continuous `1h` grid, and linearly interpolated `30` missing/spring-forward hours (`145,392` clean hourly records).
- **Load Range:** Min `14,544.0 MW` | Median `31,420.0 MW` | Mean `32,078.93 MW` | Max `62,009.0 MW`.

## Validation & Leakage Controls
- **Holdout Split:** Chronological split — training set `136,632` hours (`2002-01-01` to `2017-08-03 00:00`) and held-out test set `8,760` hours (`2017-08-03 01:00` to `2018-08-03 00:00`, final 365 days).
- **Cross-Validation:** 5-fold expanding-window `TimeSeriesSplit` across `136,296` training feature rows.
- **Causal Feature Boundary (`src/features.py`):** For 24-hour-ahead prediction, all autoregressive lags (`lag_24`, `lag_48`, `lag_168`, `lag_336`) and rolling statistics (`roll_mean_24`, `roll_std_24`, `roll_mean_168`, `roll_std_168`) are computed on `y.shift(24)`. No intraday values closer than `t - 24h` enter the feature matrix.
- **Runtime Guardrails (`src/guardrails.py`):** Pydantic v2 validation for forecast requests (`ForecastRequest`) and load records (`EnergyRecord`), safe `data/` path containment, and Babel formatting (`en`, `id`).

## Model Benchmark (8,760-Hour Held-Out Test Set)
| Model | MAE (MW) | RMSE (MW) | MAPE (%) |
|-------|----------|-----------|----------|
| **LightGBM** | **1,881.49** | **2,558.01** | **5.91%** |
| Prophet | 3,241.42 | 4,227.92 | 10.23% |
| Seasonal Naive (`t-168h`) | 3,581.81 | 4,852.33 | 11.22% |
| ARIMA `(2, 1, 2)` + train hour profile | 5,680.02 | 6,645.42 | 19.59% |

- **LightGBM 5-Fold CV:** Mean `MAE 1,658.78 MW` (std `67.24`), `RMSE 2,331.00 MW` (std `78.95`), `MAPE 5.02%` (std `0.31%`).
- **Top Feature Contributions (LightGBM split count):** `roll_mean_168` (`13.69%`), `dayofyear` (`10.57%`), `roll_std_168` (`10.42%`), `roll_mean_24` (`9.81%`), `doy_sin` (`9.00%`).

## Error Analysis (LightGBM Test Errors)
- **Peak-Hour Sensitivity:** Lowest hourly error occurs at `10:00` (`MAE 1,537.68 MW`, `MAPE 4.75%`); highest error occurs during late-afternoon ramp and peak hours `17:00` (`MAE 2,397.52 MW`, `MAPE 6.82%`) and `18:00` (`MAE 2,398.09 MW`, `MAPE 6.69%`).
- **Seasonal Extremes:** Winter heating and summer cooling months drive the largest errors — January (`MAE 2,633.97 MW`, `MAPE 7.53%`), July (`MAE 2,597.23 MW`, `MAPE 7.16%`), and June (`MAE 2,564.87 MW`, `MAPE 8.00%`) — compared with mild shoulder months such as November (`MAE 1,119.99 MW`, `MAPE 3.98%`) and April (`MAE 1,176.05 MW`, `MAPE 4.28%`).
- **Day-Type & Outliers:** Weekday `MAE 1,863.39 MW` (`MAPE 5.72%`, `n=6,264`) vs. weekend `MAE 1,926.92 MW` (`MAPE 6.39%`, `n=2,496`). The largest single-day error occurred on `2017-12-31` (`MAE 6,733.72 MW`, `MAPE 17.15%`, actual daily mean `39,016.00 MW` vs. predicted `32,282.28 MW`), followed by `2018-06-03` (`MAE 5,828.88 MW`, `MAPE 22.02%`) and `2018-07-07` (`MAE 5,383.62 MW`, `MAPE 19.06%`).

## Notebooks (Executable Locally & in Google Colab)
- [01_eda.ipynb](notebooks/01_eda.ipynb) — DST cleaning audit, Augmented Dickey-Fuller stationarity check, hourly/monthly profiles, and STL decomposition ([Open in Colab](https://colab.research.google.com/github/GaneshArwan/time_series_energy_demand/blob/main/notebooks/01_eda.ipynb))
- [02_modeling.ipynb](notebooks/02_modeling.ipynb) — Causal feature construction, 4-model benchmark, and residual/error breakdown ([Open in Colab](https://colab.research.google.com/github/GaneshArwan/time_series_energy_demand/blob/main/notebooks/02_modeling.ipynb))

## Interactive Demo & Project Layout
```text
├── app.py                   # Streamlit explorer (Forecast, STL, Benchmark, Error Analysis)
├── artifacts/               # Precomputed summary, metrics, STL, and test predictions (< 1 MB)
├── data/                    # Local CSV workspace (gitignored except .gitkeep)
├── notebooks/               # Executed EDA and modeling notebooks
├── src/
│   ├── data.py              # Safe CSV ingestion and DST hourly grid normalization
│   ├── features.py          # Strictly causal lag, rolling, and calendar features
│   ├── guardrails.py        # Pydantic v2 schemas, path containment, Babel i18n
│   └── train.py             # Chronological split, CV, 4-model evaluation, STL, artifact export
└── tests/
    ├── test_guardrails.py   # Validation, path safety, and locale formatting tests
    └── test_pipeline.py     # DST cleaning, leakage/causality, metric, and pipeline tests
```

## Setup & Local Execution
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run unit tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Retrain models and regenerate artifacts (requires data/PJME_hourly.csv)
.\.venv\Scripts\python.exe -m src.train

# Launch interactive dashboard
.\.venv\Scripts\streamlit.exe run app.py
```

## Methodology & Limitations
Electricity demand in `PJME` is strongly driven by ambient temperature (heating degree days in winter and cooling degree days in summer) and abrupt weather transitions. Because the Kaggle `PJME_hourly.csv` dataset contains load timestamps without weather covariates, univariate models rely entirely on recent lagged demand and calendar cycles. When weather shifts sharply over 24 hours—or during holiday transitions such as New Year's Eve (`2017-12-31`)—lag-based forecasts underestimate or overshoot sudden load spikes. Adding historical and forecasted temperature features, regional holiday calendars, and probabilistic quantile intervals would directly address these peak-error days.
