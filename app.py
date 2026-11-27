"""Interactive Streamlit dashboard for PJM East hourly energy demand forecasting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from src.guardrails import ForecastRequest, format_energy_mw, get_message

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

MODEL_COLUMNS: dict[str, str] = {
    "LightGBM": "lgbm_mw",
    "Prophet": "prophet_mw",
    "Seasonal Naive (t-168h)": "naive_mw",
    "ARIMA": "arima_mw",
}

UI_COPY: dict[str, dict[str, str]] = {
    "en": {
        "title": "PJM East Hourly Electricity Demand Forecasting",
        "subtitle": (
            "Next-day (24h-ahead) demand forecasting across 16 years of PJM East "
            "hourly load data (2002-2018) with strict chronological validation."
        ),
        "controls": "Forecast Controls",
        "locale": "Display Locale",
        "region": "PJM Region",
        "horizon": "Forecast Horizon (hours)",
        "tab_forecast": "Forecast Explorer",
        "tab_stl": "STL Decomposition",
        "tab_models": "Model Benchmark",
        "tab_errors": "Error Analysis",
        "date_range": "Test Window Date Range",
        "models": "Overlay Models",
        "best_model": "Best Model",
        "test_mae": "Window MAE",
        "test_rmse": "Window RMSE",
        "test_mape": "Window MAPE",
        "cv_summary": "5-Fold Expanding-Window TimeSeriesSplit CV (LightGBM)",
        "feat_imp": "LightGBM Feature Importance (%)",
        "hourly_err": "MAE by Hour of Day (MW)",
        "monthly_err": "MAE by Month (MW)",
        "top_days": "Top 10 Highest-MAE Dates",
    },
    "id": {
        "title": "Peramalan Permintaan Listrik Per Jam PJM East",
        "subtitle": (
            "Peramalan beban listrik 24 jam ke depan pada 16 tahun data per jam "
            "PJM East (2002-2018) dengan validasi kronologis ketat."
        ),
        "controls": "Pengaturan Peramalan",
        "locale": "Bahasa & Format",
        "region": "Zona PJM",
        "horizon": "Horizon Peramalan (jam)",
        "tab_forecast": "Eksplorasi Peramalan",
        "tab_stl": "Dekomposisi STL",
        "tab_models": "Perbandingan Model",
        "tab_errors": "Analisis Error",
        "date_range": "Rentang Tanggal Uji",
        "models": "Model Pembanding",
        "best_model": "Model Terbaik",
        "test_mae": "MAE Rentang Uji",
        "test_rmse": "RMSE Rentang Uji",
        "test_mape": "MAPE Rentang Uji",
        "cv_summary": "Validasi Silang Expanding-Window 5-Fold (LightGBM)",
        "feat_imp": "Tingkat Kepentingan Fitur LightGBM (%)",
        "hourly_err": "MAE per Jam dalam Sehari (MW)",
        "monthly_err": "MAE per Bulan (MW)",
        "top_days": "10 Hari dengan MAE Tertinggi",
    },
}


def _safe_artifact(filename: str) -> Path:
    base = ARTIFACTS_DIR.resolve()
    target = (base / filename).resolve()
    if target == base or not target.is_relative_to(base) or not target.is_file():
        raise FileNotFoundError(f"Missing artifact: {filename}")
    return target


@st.cache_data
def load_artifacts() -> dict[str, Any]:
    """Load precomputed forecasting artifacts safely."""
    summary = json.loads(_safe_artifact("dataset_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads(_safe_artifact("metrics.json").read_text(encoding="utf-8"))
    errors = json.loads(_safe_artifact("error_analysis.json").read_text(encoding="utf-8"))
    feat_imp = json.loads(_safe_artifact("feature_importance.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(_safe_artifact("test_predictions.csv"), parse_dates=["ds"])
    stl = pd.read_csv(_safe_artifact("decomposition_daily.csv"), parse_dates=["ds"])
    return {
        "summary": summary,
        "metrics": metrics,
        "errors": errors,
        "feat_imp": feat_imp,
        "preds": preds,
        "stl": stl,
    }


def _compute_metrics(actual: pd.Series, pred: pd.Series) -> tuple[float, float, float]:
    diff = actual.to_numpy(dtype=float) - pred.to_numpy(dtype=float)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    y_true = actual.to_numpy(dtype=float)
    mask = y_true > 0
    mape = float(np.mean(np.abs(diff[mask]) / y_true[mask]) * 100.0) if mask.any() else 0.0
    return mae, rmse, mape


def main() -> None:
    """Render the Streamlit forecasting application."""
    st.set_page_config(page_title="PJM Energy Demand Forecasting", layout="wide")
    data = load_artifacts()
    summary = data["summary"]
    metrics = data["metrics"]
    preds = data["preds"]
    stl = data["stl"]
    errors = data["errors"]
    feat_imp = data["feat_imp"]

    with st.sidebar:
        locale = st.selectbox("Locale / Bahasa", options=["en", "id"], index=0)
        copy = UI_COPY[locale]
        st.header(copy["controls"])
        region_input = st.text_input(copy["region"], value="PJME", max_chars=16)
        horizon_input = st.number_input(copy["horizon"], min_value=1, max_value=168, value=24)

        try:
            req = ForecastRequest(region=region_input, horizon_hours=int(horizon_input), locale=locale)
            st.success(f"{get_message('forecast_ready', locale)} ({req.region}, {req.horizon_hours}h)")
        except ValidationError as exc:
            st.error(f"{get_message('invalid_region', locale)}: {exc.errors()[0]['msg']}")
            st.stop()

        st.divider()
        st.caption(
            f"PJME clean series: {summary['clean_rows']:,} hourly rows "
            f"({summary['start_date'][:10]} to {summary['end_date'][:10]})"
        )
        st.caption(
            f"Min {format_energy_mw(summary['min_mw'], locale)} | "
            f"Median {format_energy_mw(summary['median_mw'], locale)} | "
            f"Max {format_energy_mw(summary['max_mw'], locale)}"
        )

    st.title(copy["title"])
    st.caption(copy["subtitle"])

    tabs = st.tabs([copy["tab_forecast"], copy["tab_stl"], copy["tab_models"], copy["tab_errors"]])

    with tabs[0]:
        min_d = preds["ds"].min().date()
        max_d = preds["ds"].max().date()
        default_start = max(min_d, (preds["ds"].max() - pd.Timedelta(days=30)).date())
        start_d, end_d = st.slider(
            copy["date_range"],
            min_value=min_d,
            max_value=max_d,
            value=(default_start, max_d),
        )
        selected_models = st.multiselect(
            copy["models"],
            options=list(MODEL_COLUMNS.keys()),
            default=["LightGBM", "Prophet"],
        )
        window = preds[(preds["ds"].dt.date >= start_d) & (preds["ds"].dt.date <= end_d)]

        primary_col = MODEL_COLUMNS[selected_models[0]] if selected_models else "lgbm_mw"
        primary_name = selected_models[0] if selected_models else "LightGBM"
        mae, rmse, mape = _compute_metrics(window["actual_mw"], window[primary_col])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(copy["best_model"], primary_name)
        c2.metric(copy["test_mae"], format_energy_mw(mae, req.locale))
        c3.metric(copy["test_rmse"], format_energy_mw(rmse, req.locale))
        c4.metric(copy["test_mape"], f"{mape:.2f}%")

        chart_cols = ["actual_mw"] + [MODEL_COLUMNS[m] for m in selected_models]
        chart_df = window.set_index("ds")[chart_cols].rename(
            columns={"actual_mw": "Actual", **{v: k for k, v in MODEL_COLUMNS.items()}}
        )
        st.line_chart(chart_df, height=400)

    with tabs[1]:
        stl_df = stl.set_index("ds")[["observed", "trend", "seasonal", "residual"]]
        st.subheader("Daily Mean Demand & Trend (MW)")
        st.line_chart(stl_df[["observed", "trend"]], height=280)
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Annual Seasonal Component (MW)")
            st.line_chart(stl_df[["seasonal"]], height=240)
        with col_b:
            st.subheader("Residual Component (MW)")
            st.line_chart(stl_df[["residual"]], height=240)

    with tabs[2]:
        model_rows = [
            {"Model": name, "MAE (MW)": vals["mae"], "RMSE (MW)": vals["rmse"], "MAPE (%)": vals["mape"]}
            for name, vals in metrics["models"].items()
        ]
        model_df = pd.DataFrame(model_rows).sort_values("MAE (MW)")
        st.dataframe(model_df, width="stretch", hide_index=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader(copy["cv_summary"])
            cv_df = pd.DataFrame(metrics["lightgbm_cv"]["folds"]).rename(
                columns={"fold": "Fold", "mae": "MAE", "rmse": "RMSE", "mape": "MAPE (%)"}
            )
            st.dataframe(cv_df[["Fold", "train_rows", "val_rows", "MAE", "RMSE", "MAPE (%)"]], hide_index=True)
        with col_r:
            st.subheader(copy["feat_imp"])
            imp_df = pd.DataFrame(feat_imp["features"]).set_index("feature")[["importance_pct"]]
            st.bar_chart(imp_df, height=300)

    with tabs[3]:
        c_hr, c_mo = st.columns(2)
        with c_hr:
            st.subheader(copy["hourly_err"])
            hr_df = pd.DataFrame(errors["by_hour"]).set_index("hour")[["mae"]]
            st.bar_chart(hr_df, height=280)
        with c_mo:
            st.subheader(copy["monthly_err"])
            mo_df = pd.DataFrame(errors["by_month"]).set_index("month")[["mae"]]
            st.bar_chart(mo_df, height=280)

        st.subheader(copy["top_days"])
        top_df = pd.DataFrame(errors["top_10_highest_mae_days"])
        st.dataframe(top_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
