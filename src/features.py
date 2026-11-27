"""Strictly causal feature engineering for hourly energy demand forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

_CANDIDATE_LAGS = (24, 48, 168, 336)


def build_causal_features(
    df: pd.DataFrame,
    horizon_hours: int = 24,
) -> tuple[pd.DataFrame, list[str]]:
    """Build leak-free calendar, lag, and rolling features for h-step forecasting.

    All lag and rolling features are restricted to target values at least
    ``horizon_hours`` steps before the prediction timestamp ``ds``.

    Args:
        df: Clean hourly DataFrame containing ``ds`` and ``y`` columns.
        horizon_hours: Forecast horizon in hours (must be >= 1).

    Returns:
        Tuple of ``(featured_df, feature_cols)`` with initial NaN rows dropped.

    Raises:
        ValueError: If ``horizon_hours < 1`` or required columns are missing.
    """
    if isinstance(horizon_hours, bool) or not isinstance(horizon_hours, int) or horizon_hours < 1:
        raise ValueError("horizon_hours must be an integer >= 1.")
    if not {"ds", "y"}.issubset(df.columns):
        raise ValueError("Input DataFrame must contain 'ds' and 'y' columns.")

    work = df[["ds", "y"]].copy()
    work["ds"] = pd.to_datetime(work["ds"], errors="raise")
    work["y"] = pd.to_numeric(work["y"], errors="raise").astype("float64")
    work = work.sort_values("ds").reset_index(drop=True)

    hour = work["ds"].dt.hour.astype("int32")
    dayofweek = work["ds"].dt.day_of_week.astype("int32")
    month = work["ds"].dt.month.astype("int32")
    dayofyear = work["ds"].dt.day_of_year.astype("int32")

    work["hour"] = hour
    work["dayofweek"] = dayofweek
    work["month"] = month
    work["dayofyear"] = dayofyear
    work["is_weekend"] = (dayofweek >= 5).astype("int8")
    work["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    work["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    work["doy_sin"] = np.sin(2.0 * np.pi * dayofyear / 365.25)
    work["doy_cos"] = np.cos(2.0 * np.pi * dayofyear / 365.25)

    feature_cols: list[str] = [
        "hour",
        "dayofweek",
        "month",
        "dayofyear",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]

    for lag in _CANDIDATE_LAGS:
        if lag >= horizon_hours:
            col_name = f"lag_{lag}"
            work[col_name] = work["y"].shift(lag)
            feature_cols.append(col_name)

    shifted_y = work["y"].shift(horizon_hours)
    work["roll_mean_24"] = shifted_y.rolling(window=24, min_periods=24).mean()
    work["roll_std_24"] = shifted_y.rolling(window=24, min_periods=24).std()
    work["roll_mean_168"] = shifted_y.rolling(window=168, min_periods=168).mean()
    work["roll_std_168"] = shifted_y.rolling(window=168, min_periods=168).std()
    feature_cols.extend(["roll_mean_24", "roll_std_24", "roll_mean_168", "roll_std_168"])

    featured_df = work.dropna(subset=[*feature_cols, "y"]).reset_index(drop=True)
    return featured_df, feature_cols
