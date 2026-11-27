"""Load, clean, and validate hourly PJM electricity demand series."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.guardrails import EnergyRecord, resolve_safe_data_path

_CANDIDATE_TIME_COLS = ("Datetime", "ds")


def _extract_region_and_target_col(columns: list[str]) -> tuple[str, str]:
    """Infer region code and demand column name from CSV headers."""
    mw_cols = [col for col in columns if col.endswith("_MW")]
    if len(mw_cols) == 1:
        col = mw_cols[0]
        return col[:-3].strip().upper(), col
    if "y" in columns:
        return "PJME", "y"
    raise ValueError("CSV must contain a single '*_MW' demand column or 'y'.")


def _validate_sample_records(df: pd.DataFrame, region: str) -> None:
    """Validate head/tail sample rows through EnergyRecord schema."""
    if len(df) <= 100:
        sample = df
    else:
        sample = pd.concat([df.iloc[:50], df.iloc[-50:]], ignore_index=True)

    for row in sample.itertuples(index=False):
        EnergyRecord(
            timestamp=row.ds.to_pydatetime(),
            region=region,
            demand_mw=float(row.y),
        )


def load_pjm_series(
    filename: str = "PJME_hourly.csv",
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, deduplicate, reindex, and validate an hourly PJM demand CSV.

    Args:
        filename: Relative CSV filename inside ``data_dir``.
        data_dir: Optional directory override; defaults to project ``data/``.

    Returns:
        Tuple of ``(clean_df, summary)`` where ``clean_df`` has timezone-naive
        ``ds`` (``datetime64[ns]``) and ``y`` (``float64``) on a strict 1h grid.

    Raises:
        FileNotFoundError: If the resolved CSV file does not exist.
        ValueError: On unsafe paths, missing required columns, or invalid data.
    """
    resolved_path = resolve_safe_data_path(filename, data_dir=data_dir)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Data file not found: {filename}")

    raw_df = pd.read_csv(resolved_path)
    raw_df.columns = [str(col).strip() for col in raw_df.columns]
    if raw_df.empty:
        raise ValueError("Input CSV contains no rows.")

    time_col = next((c for c in _CANDIDATE_TIME_COLS if c in raw_df.columns), None)
    if time_col is None:
        raise ValueError("CSV must contain a 'Datetime' or 'ds' timestamp column.")

    region, target_col = _extract_region_and_target_col(list(raw_df.columns))

    parsed_ts = pd.to_datetime(raw_df[time_col], errors="raise")
    if getattr(parsed_ts.dt, "tz", None) is not None:
        parsed_ts = parsed_ts.dt.tz_convert(None)

    demand_values = pd.to_numeric(raw_df[target_col], errors="raise").astype("float64")
    raw_rows = int(len(raw_df))

    staged = pd.DataFrame({"ds": parsed_ts, "y": demand_values}).sort_values("ds")
    deduped = staged.groupby("ds", sort=True)["y"].mean()
    duplicates_resolved = int(raw_rows - len(deduped))

    hourly_index = pd.date_range(
        start=deduped.index.min(),
        end=deduped.index.max(),
        freq="1h",
    )
    reindexed = deduped.reindex(hourly_index)
    gaps_interpolated = int(reindexed.isna().sum())
    interpolated = reindexed.interpolate(method="linear")

    clean_df = pd.DataFrame(
        {
            "ds": hourly_index.astype("datetime64[ns]"),
            "y": interpolated.to_numpy(dtype="float64"),
        }
    )
    if clean_df["y"].isna().any():
        raise ValueError("Cleaned demand series still contains NaN values.")

    _validate_sample_records(clean_df, region=region)

    summary: dict[str, Any] = {
        "region": region,
        "raw_rows": raw_rows,
        "clean_rows": int(len(clean_df)),
        "duplicates_resolved": duplicates_resolved,
        "gaps_interpolated": gaps_interpolated,
        "start_date": clean_df["ds"].iloc[0].isoformat(),
        "end_date": clean_df["ds"].iloc[-1].isoformat(),
        "min_mw": round(float(clean_df["y"].min()), 2),
        "median_mw": round(float(clean_df["y"].median()), 2),
        "mean_mw": round(float(clean_df["y"].mean()), 2),
        "max_mw": round(float(clean_df["y"].max()), 2),
    }
    return clean_df, summary
