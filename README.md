# Time Series Forecasting: Energy Demand

## Problem Statement
How accurately can next-day electricity demand be predicted, and what drives forecast errors?

## Dataset
- **Source:** PJM Hourly Energy Consumption (Kaggle) — hourly energy consumption data across multiple US regions over 10+ years.
- **Size:** [Rows x features]
- **Description:** [What the data represents and any notable quality issues]

## Skills Demonstrated
Time series decomposition, ARIMA/Prophet, feature engineering with lag variables, forecast evaluation

## Deliverables
- Time series decomposition showing trend, seasonality, and residuals
- Comparison of ARIMA, Prophet, and gradient boosting approaches
- Feature engineering with lag features, rolling averages, and calendar effects
- Forecast accuracy evaluation using MAE, RMSE, and MAPE with honest error analysis

## Approach
1. **Data Cleaning & EDA** - [Summary of preprocessing steps]
2. **Feature Engineering** - [Key transformations and rationale]
3. **Modeling** - [Algorithms used and why]
4. **Evaluation** - [Metrics, cross-validation strategy]

## Key Results
| Metric | Score |
|--------|-------|
| MAE | TBD |
| RMSE | TBD |
| MAPE | TBD |

## Key Findings
1. [Most important finding with specific number]
2. [Second finding]
3. [Business implication or recommendation]

## How to Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
streamlit run app.py  # for demo
```

## Files
- `notebooks/01_eda.ipynb` - Exploratory analysis
- `notebooks/02_modeling.ipynb` - Model training and evaluation
- `src/` - Production-ready scripts
- `app.py` - Streamlit demo (if applicable)

## Methodology & Limitations
[2-3 paragraphs on approach, assumptions, and what the model does NOT capture]
