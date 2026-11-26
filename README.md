# Time Series Forecasting: Energy Demand

## Problem Statement
How accurately can next-day electricity demand be predicted, and what drives forecast errors?

## Dataset
- **Source:** PJM Hourly Energy Consumption (Kaggle) — hourly electricity demand across PJM Interconnection regional transmission zones over 10+ years.
- **Location:** Place local CSV files under `data/` (gitignored except `data/.gitkeep`).

## Skills Demonstrated
Time series decomposition, ARIMA/Prophet, feature engineering with lag variables, forecast evaluation

## Deliverables
- Time series decomposition showing trend, seasonality, and residuals
- Comparison of ARIMA, Prophet, and gradient boosting approaches
- Feature engineering with lag features, rolling averages, and calendar effects
- Forecast accuracy evaluation using MAE, RMSE, and MAPE with error analysis

## Planned Milestones
1. **Data Ingestion & EDA** (`notebooks/`) — Inspect regional hourly series, timezone/DST shifts, and missing timestamps.
2. **Decomposition & Feature Engineering** — Extract trend/seasonal components alongside lag features, rolling statistics, and calendar indicators.
3. **Forecasting & Evaluation** — Benchmark ARIMA, Prophet, and gradient boosting models using MAE, RMSE, and MAPE.

## Current Structure
- `data/` — Local dataset directory (ignored by git)
- `notebooks/` — Exploratory and modeling notebooks workspace
- `src/__init__.py` — Package exports
- `src/guardrails.py` — Pydantic v2 request/record schemas, safe `data/` path resolution, and Babel i18n helpers (`en`, `id`)
- `tests/test_guardrails.py` — Unit tests for input validation, path traversal checks, and localization

## Setup & Verification
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
