"""Leak-free chronological training, evaluation, and artifact export pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL

from src.data import load_pjm_series
from src.features import build_causal_features
from src.guardrails import ForecastRequest

_DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
_TEST_HOURS_DEFAULT = 365 * 24

# Suppress verbose stan/prophet info logs during training runs.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE (%) guarding against zero targets."""
    actual = np.asarray(y_true, dtype="float64")
    pred = np.asarray(y_pred, dtype="float64")
    if actual.shape != pred.shape or actual.size == 0:
        raise ValueError("y_true and y_pred must be non-empty arrays of identical shape.")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(pred)):
        raise ValueError("y_true and y_pred must contain only finite values.")

    positive_mask = actual > 0.0
    if not np.any(positive_mask):
        raise ValueError("MAPE requires at least one strictly positive actual value.")

    mae = float(mean_absolute_error(actual, pred))
    rmse = float(root_mean_squared_error(actual, pred))
    mape = float(
        np.mean(np.abs(actual[positive_mask] - pred[positive_mask]) / actual[positive_mask])
        * 100.0
    )
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
    }


def predict_seasonal_naive(df: pd.DataFrame, test_hours: int) -> np.ndarray:
    """Predict test window using strictly causal weekly seasonal naive (t - 168h)."""
    if len(df) < test_hours + 168:
        raise ValueError("Series is too short for 168-hour seasonal naive baseline.")
    shifted = df["y"].shift(168).iloc[-test_hours:]
    return shifted.to_numpy(dtype="float64")


def predict_arima_hourly(
    train_df: pd.DataFrame,
    test_ds: pd.Series,
    order: tuple[int, int, int] = (2, 1, 2),
) -> np.ndarray:
    """Forecast hourly demand via daily ARIMA plus train-only hour-of-day profile."""
    indexed = train_df.set_index("ds")["y"]
    day_counts = indexed.groupby(indexed.index.floor("D")).size()
    full_days = day_counts[day_counts == 24].index
    daily_all = indexed.resample("1D").mean().dropna()
    daily_train = daily_all.loc[daily_all.index.isin(full_days)] if len(full_days) >= 30 else daily_all

    last_train_day = daily_train.index.max()
    test_dates = pd.to_datetime(test_ds).dt.floor("D")
    max_test_day = test_dates.max()
    steps = max(1, int((max_test_day - last_train_day).days))

    fitted = ARIMA(daily_train, order=order).fit()
    fc_daily = fitted.forecast(steps=steps)
    fc_index = pd.date_range(last_train_day + pd.Timedelta(days=1), periods=steps, freq="D")
    fc_series = pd.Series(fc_daily.to_numpy(dtype="float64"), index=fc_index)

    # Fallback only if test shares the final partial training day on tiny datasets.
    if last_train_day in set(test_dates) and last_train_day not in fc_series.index:
        fc_series.loc[last_train_day] = float(daily_train.iloc[-1])

    train_mean = float(indexed.mean())
    if train_mean > 0.0:
        hour_profile = indexed.groupby(indexed.index.hour).mean() / train_mean
        daily_mapped = test_dates.map(fc_series).to_numpy(dtype="float64")
        hour_mapped = pd.to_datetime(test_ds).dt.hour.map(hour_profile).to_numpy(dtype="float64")
        preds = daily_mapped * hour_mapped
    else:
        hour_offset = indexed.groupby(indexed.index.hour).mean() - train_mean
        daily_mapped = test_dates.map(fc_series).to_numpy(dtype="float64")
        hour_mapped = pd.to_datetime(test_ds).dt.hour.map(hour_offset).to_numpy(dtype="float64")
        preds = daily_mapped + hour_mapped

    if np.isnan(preds).any():
        raise ValueError("ARIMA predictions contain NaN values due to date alignment.")
    return preds


def predict_prophet_hourly(train_df: pd.DataFrame, test_ds: pd.Series) -> np.ndarray:
    """Fit Prophet with daily/weekly/yearly seasonality on train and predict test."""
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    model.fit(train_df[["ds", "y"]])
    forecast = model.predict(pd.DataFrame({"ds": pd.to_datetime(test_ds)}))
    return forecast["yhat"].to_numpy(dtype="float64")


def _build_lgbm_regressor() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample_for_bin=50000,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def evaluate_and_fit_lightgbm(
    featured_df: pd.DataFrame,
    feature_cols: list[str],
    test_start: pd.Timestamp,
    n_splits: int = 5,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Run expanding-window TimeSeriesSplit CV on train, then fit full train and predict test."""
    train_feat = featured_df.loc[featured_df["ds"] < test_start].reset_index(drop=True)
    test_feat = featured_df.loc[featured_df["ds"] >= test_start].reset_index(drop=True)
    if train_feat.empty or test_feat.empty:
        raise ValueError("Featured train or test split is empty.")

    x_train = train_feat[feature_cols]
    y_train = train_feat["y"].to_numpy(dtype="float64")
    x_test = test_feat[feature_cols]

    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds: list[dict[str, Any]] = []
    for fold_idx, (tr_idx, val_idx) in enumerate(splitter.split(x_train), start=1):
        cv_model = _build_lgbm_regressor()
        cv_model.fit(x_train.iloc[tr_idx], y_train[tr_idx])
        val_pred = cv_model.predict(x_train.iloc[val_idx])
        fold_metrics = compute_metrics(y_train[val_idx], val_pred)
        folds.append(
            {
                "fold": fold_idx,
                "train_rows": int(len(tr_idx)),
                "val_rows": int(len(val_idx)),
                **fold_metrics,
            }
        )

    cv_summary = {
        "n_splits": n_splits,
        "train_feature_rows": int(len(train_feat)),
        "folds": folds,
        "mean": {
            metric: round(float(np.mean([f[metric] for f in folds])), 2)
            for metric in ("mae", "rmse", "mape")
        },
        "std": {
            metric: round(float(np.std([f[metric] for f in folds])), 2)
            for metric in ("mae", "rmse", "mape")
        },
    }

    final_model = _build_lgbm_regressor()
    final_model.fit(x_train, y_train)
    test_pred = np.asarray(final_model.predict(x_test), dtype="float64")

    raw_importances = np.asarray(final_model.feature_importances_, dtype="float64")
    total_imp = float(raw_importances.sum())
    importance_records = [
        {
            "feature": feat,
            "importance": int(imp),
            "importance_pct": round((float(imp) / total_imp * 100.0) if total_imp > 0.0 else 0.0, 2),
        }
        for feat, imp in zip(feature_cols, raw_importances, strict=True)
    ]
    importance_records.sort(key=lambda item: item["importance"], reverse=True)
    return test_pred, cv_summary, importance_records


def compute_stl_decomposition(clean_df: pd.DataFrame, period: int = 365) -> pd.DataFrame:
    """Compute robust STL decomposition on daily mean demand."""
    daily_series = clean_df.set_index("ds")["y"].resample("1D").mean().dropna()
    stl_result = STL(daily_series, period=period, robust=True).fit()
    decomp = pd.DataFrame(
        {
            "ds": daily_series.index.strftime("%Y-%m-%d"),
            "observed": np.round(stl_result.observed.to_numpy(dtype="float64"), 2),
            "trend": np.round(stl_result.trend.to_numpy(dtype="float64"), 2),
            "seasonal": np.round(stl_result.seasonal.to_numpy(dtype="float64"), 2),
            "residual": np.round(stl_result.resid.to_numpy(dtype="float64"), 2),
        }
    )
    return decomp


def compute_error_analysis(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    """Compute hourly, monthly, day-type, and worst-day error breakdown."""
    frame = pd.DataFrame(
        {
            "ds": pd.to_datetime(test_df["ds"]),
            "actual": test_df["y"].to_numpy(dtype="float64"),
            "pred": np.asarray(y_pred, dtype="float64"),
        }
    )
    frame["abs_err"] = np.abs(frame["actual"] - frame["pred"])
    frame["pct_err"] = np.where(
        frame["actual"] > 0.0,
        frame["abs_err"] / frame["actual"] * 100.0,
        np.nan,
    )
    frame["hour"] = frame["ds"].dt.hour
    frame["month"] = frame["ds"].dt.month
    frame["is_weekend"] = frame["ds"].dt.day_of_week >= 5
    frame["date"] = frame["ds"].dt.strftime("%Y-%m-%d")

    by_hour = [
        {
            "hour": int(hour),
            "mae": round(float(grp["abs_err"].mean()), 2),
            "mape": round(float(grp["pct_err"].mean()), 2),
        }
        for hour, grp in frame.groupby("hour", sort=True)
    ]

    by_month = [
        {
            "month": int(month),
            "mae": round(float(grp["abs_err"].mean()), 2),
            "mape": round(float(grp["pct_err"].mean()), 2),
        }
        for month, grp in frame.groupby("month", sort=True)
    ]

    day_type_summary: dict[str, dict[str, Any]] = {}
    for is_wknd, label in ((False, "weekday"), (True, "weekend")):
        subset = frame.loc[frame["is_weekend"] == is_wknd]
        day_type_summary[label] = {
            "mae": round(float(subset["abs_err"].mean()), 2),
            "mape": round(float(subset["pct_err"].mean()), 2),
            "count": int(len(subset)),
        }

    daily_errors = (
        frame.groupby("date", as_index=False)
        .agg(
            mae=("abs_err", "mean"),
            mape=("pct_err", "mean"),
            actual_mean_mw=("actual", "mean"),
            pred_mean_mw=("pred", "mean"),
        )
        .sort_values("mae", ascending=False)
        .head(10)
    )
    top_10_days = [
        {
            "date": str(row.date),
            "mae": round(float(row.mae), 2),
            "mape": round(float(row.mape), 2),
            "actual_mean_mw": round(float(row.actual_mean_mw), 2),
            "pred_mean_mw": round(float(row.pred_mean_mw), 2),
        }
        for row in daily_errors.itertuples(index=False)
    ]

    return {
        "best_model": model_name,
        "by_hour": by_hour,
        "by_month": by_month,
        "by_day_type": day_type_summary,
        "top_10_highest_mae_days": top_10_days,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_pipeline(
    filename: str = "PJME_hourly.csv",
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    test_hours: int = _TEST_HOURS_DEFAULT,
    horizon_hours: int = 24,
    stl_period: int = 365,
    n_cv_splits: int = 5,
) -> dict[str, Any]:
    """Execute end-to-end leak-free training and export compact artifacts."""
    clean_df, summary = load_pjm_series(filename=filename, data_dir=data_dir)
    request = ForecastRequest(region=summary["region"], horizon_hours=horizon_hours)

    if len(clean_df) <= test_hours + 336:
        raise ValueError("Series does not have enough observations for train/test split.")

    train_df = clean_df.iloc[:-test_hours].reset_index(drop=True)
    test_df = clean_df.iloc[-test_hours:].reset_index(drop=True)
    test_start = pd.Timestamp(test_df["ds"].iloc[0])

    featured_df, feature_cols = build_causal_features(clean_df, horizon_hours=request.horizon_hours)
    y_test = test_df["y"].to_numpy(dtype="float64")

    naive_pred = predict_seasonal_naive(clean_df, test_hours=test_hours)
    arima_pred = predict_arima_hourly(train_df, test_df["ds"])
    prophet_pred = predict_prophet_hourly(train_df, test_df["ds"])
    lgbm_pred, lgbm_cv, feature_importance = evaluate_and_fit_lightgbm(
        featured_df=featured_df,
        feature_cols=feature_cols,
        test_start=test_start,
        n_splits=n_cv_splits,
    )

    predictions = {
        "Seasonal Naive (t-168h)": naive_pred,
        "ARIMA": arima_pred,
        "Prophet": prophet_pred,
        "LightGBM": lgbm_pred,
    }
    model_metrics = {name: compute_metrics(y_test, pred) for name, pred in predictions.items()}
    best_model = min(model_metrics, key=lambda name: model_metrics[name]["mae"])

    metrics_payload: dict[str, Any] = {
        "split": {
            "horizon_hours": request.horizon_hours,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_start": train_df["ds"].iloc[0].isoformat(),
            "train_end": train_df["ds"].iloc[-1].isoformat(),
            "test_start": test_df["ds"].iloc[0].isoformat(),
            "test_end": test_df["ds"].iloc[-1].isoformat(),
        },
        "models": model_metrics,
        "best_model": best_model,
        "lightgbm_cv": lgbm_cv,
    }

    decomp_df = compute_stl_decomposition(clean_df, period=stl_period)
    error_analysis = compute_error_analysis(test_df, predictions[best_model], model_name=best_model)
    test_predictions_df = pd.DataFrame(
        {
            "ds": test_df["ds"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "actual_mw": np.round(y_test, 2),
            "naive_mw": np.round(naive_pred, 2),
            "arima_mw": np.round(arima_pred, 2),
            "prophet_mw": np.round(prophet_pred, 2),
            "lgbm_mw": np.round(lgbm_pred, 2),
        }
    )

    artifacts_dir = (output_dir if output_dir is not None else _DEFAULT_ARTIFACTS_DIR).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _write_json(artifacts_dir / "dataset_summary.json", summary)
    _write_json(artifacts_dir / "metrics.json", metrics_payload)
    decomp_df.to_csv(artifacts_dir / "decomposition_daily.csv", index=False)
    test_predictions_df.to_csv(artifacts_dir / "test_predictions.csv", index=False)
    _write_json(artifacts_dir / "error_analysis.json", error_analysis)
    _write_json(
        artifacts_dir / "feature_importance.json",
        {"model": "LightGBM", "features": feature_importance},
    )
    return metrics_payload


def main() -> None:
    """CLI entrypoint for training and artifact generation."""
    metrics = run_pipeline()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
