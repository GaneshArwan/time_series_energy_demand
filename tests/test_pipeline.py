"""Unit and integration tests for data, features, and training pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_pjm_series
from src.features import build_causal_features
from src.train import compute_metrics, run_pipeline


class TestLoadPjmSeries(unittest.TestCase):
    """Tests for load_pjm_series cleaning and validation."""

    def test_dst_duplicate_mean_and_gap_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            csv_path = data_dir / "PJME_hourly.csv"
            rows = [
                ("2025-03-09 03:00:00", 32000.0),
                ("2025-03-09 00:00:00", 29000.0),
                ("2025-03-09 01:00:00", 30000.0),
                # 02:00:00 missing (spring-forward gap -> interpolated 31000.0)
                ("2025-03-09 04:00:00", 33000.0),
                ("2025-03-09 04:00:00", 35000.0),  # duplicate -> mean 34000.0
            ]
            raw = pd.DataFrame(rows, columns=["Datetime", "PJME_MW"])
            raw.to_csv(csv_path, index=False)

            clean_df, summary = load_pjm_series("PJME_hourly.csv", data_dir=data_dir)

            self.assertEqual(list(clean_df.columns), ["ds", "y"])
            self.assertEqual(str(clean_df["ds"].dtype), "datetime64[ns]")
            self.assertIsNone(clean_df["ds"].dt.tz)
            self.assertEqual(str(clean_df["y"].dtype), "float64")
            self.assertEqual(len(clean_df), 5)

            gap_val = clean_df.loc[clean_df["ds"] == pd.Timestamp("2025-03-09 02:00:00"), "y"].iloc[0]
            dup_val = clean_df.loc[clean_df["ds"] == pd.Timestamp("2025-03-09 04:00:00"), "y"].iloc[0]
            self.assertAlmostEqual(float(gap_val), 31000.0)
            self.assertAlmostEqual(float(dup_val), 34000.0)

            self.assertEqual(summary["region"], "PJME")
            self.assertEqual(summary["raw_rows"], 5)
            self.assertEqual(summary["clean_rows"], 5)
            self.assertEqual(summary["duplicates_resolved"], 1)
            self.assertEqual(summary["gaps_interpolated"], 1)
            self.assertEqual(summary["min_mw"], 29000.0)
            self.assertEqual(summary["max_mw"], 34000.0)

    def test_rejects_unsafe_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_pjm_series("../secret.csv", data_dir=Path(tmp))


class TestCausalFeatures(unittest.TestCase):
    """Tests for strict chronological causality in build_causal_features."""

    def _make_series(self, n_hours: int = 500) -> pd.DataFrame:
        ds = pd.date_range("2025-01-01", periods=n_hours, freq="1h")
        y = 30000.0 + np.arange(n_hours, dtype="float64") * 5.0
        return pd.DataFrame({"ds": ds, "y": y})

    def test_strict_causality_under_future_target_mutation(self) -> None:
        horizon = 24
        base_df = self._make_series(500)
        feat_orig, feature_cols = build_causal_features(base_df, horizon_hours=horizon)

        target_idx = 420
        target_ts = base_df.loc[target_idx, "ds"]

        mutated_df = base_df.copy()
        # Mutate y at t and t-1..t-(horizon-1); none of these may affect row t.
        for offset in range(horizon):
            mutated_df.loc[target_idx - offset, "y"] += 9999.0

        feat_mutated, _ = build_causal_features(mutated_df, horizon_hours=horizon)

        row_orig = feat_orig.loc[feat_orig["ds"] == target_ts, feature_cols].iloc[0]
        row_mut = feat_mutated.loc[feat_mutated["ds"] == target_ts, feature_cols].iloc[0]
        pd.testing.assert_series_equal(row_orig, row_mut)

        # Mutating y at t - horizon MUST change causal features at t.
        past_mutated = base_df.copy()
        past_mutated.loc[target_idx - horizon, "y"] += 5000.0
        feat_past, _ = build_causal_features(past_mutated, horizon_hours=horizon)
        row_past = feat_past.loc[feat_past["ds"] == target_ts, feature_cols].iloc[0]
        self.assertFalse(row_orig.equals(row_past))

    def test_excludes_lags_smaller_than_horizon_and_rejects_bad_horizon(self) -> None:
        base_df = self._make_series(500)
        _, cols_48 = build_causal_features(base_df, horizon_hours=48)
        self.assertNotIn("lag_24", cols_48)
        self.assertIn("lag_48", cols_48)
        self.assertIn("lag_168", cols_48)
        self.assertIn("lag_336", cols_48)

        with self.assertRaises(ValueError):
            build_causal_features(base_df, horizon_hours=0)


class TestComputeMetrics(unittest.TestCase):
    """Tests for MAE, RMSE, and MAPE metric calculations."""

    def test_exact_metrics_and_zero_guard(self) -> None:
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 180.0, 330.0])
        metrics = compute_metrics(y_true, y_pred)

        # abs errors: 10, 20, 30 -> MAE = 20.0
        # sq errors: 100, 400, 900 -> mean=1400/3 -> RMSE = 21.60
        # pct errors: 10%, 10%, 10% -> MAPE = 10.0
        self.assertAlmostEqual(metrics["mae"], 20.0)
        self.assertAlmostEqual(metrics["rmse"], 21.60)
        self.assertAlmostEqual(metrics["mape"], 10.0)

        guarded = compute_metrics(np.array([0.0, 100.0]), np.array([10.0, 110.0]))
        self.assertAlmostEqual(guarded["mape"], 10.0)

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            compute_metrics(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        with self.assertRaises(ValueError):
            compute_metrics(np.array([10.0]), np.array([10.0, 20.0]))


class TestPipelineArtifacts(unittest.TestCase):
    """End-to-end synthetic pipeline run validating exported artifact schemas."""

    def test_run_pipeline_writes_valid_artifacts(self) -> None:
        rng = np.random.default_rng(42)
        n_hours = 45 * 24  # 1,080 hours
        ds = pd.date_range("2025-01-01 00:00:00", periods=n_hours, freq="1h")
        hours = np.arange(n_hours, dtype="float64")
        y = (
            30000.0
            + 4000.0 * np.sin(2.0 * np.pi * hours / 24.0)
            + 1500.0 * np.sin(2.0 * np.pi * hours / 168.0)
            + rng.normal(0.0, 250.0, size=n_hours)
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            out_dir = root / "artifacts"
            data_dir.mkdir()
            pd.DataFrame({"Datetime": ds, "PJME_MW": y}).to_csv(
                data_dir / "PJME_hourly.csv", index=False
            )

            metrics = run_pipeline(
                filename="PJME_hourly.csv",
                data_dir=data_dir,
                output_dir=out_dir,
                test_hours=7 * 24,
                horizon_hours=24,
                stl_period=7,
                n_cv_splits=3,
            )

            expected_files = (
                "dataset_summary.json",
                "metrics.json",
                "decomposition_daily.csv",
                "test_predictions.csv",
                "error_analysis.json",
                "feature_importance.json",
            )
            for fname in expected_files:
                self.assertTrue((out_dir / fname).is_file(), f"Missing artifact: {fname}")

            summary = json.loads((out_dir / "dataset_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["region"], "PJME")
            self.assertEqual(summary["clean_rows"], n_hours)

            loaded_metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(set(loaded_metrics["models"].keys()), {
                "Seasonal Naive (t-168h)",
                "ARIMA",
                "Prophet",
                "LightGBM",
            })
            self.assertIn(loaded_metrics["best_model"], loaded_metrics["models"])
            self.assertEqual(loaded_metrics["lightgbm_cv"]["n_splits"], 3)
            self.assertEqual(metrics["best_model"], loaded_metrics["best_model"])

            preds_df = pd.read_csv(out_dir / "test_predictions.csv")
            self.assertEqual(
                list(preds_df.columns),
                ["ds", "actual_mw", "naive_mw", "arima_mw", "prophet_mw", "lgbm_mw"],
            )
            self.assertEqual(len(preds_df), 7 * 24)
            self.assertFalse(preds_df.isna().any().any())

            decomp_df = pd.read_csv(out_dir / "decomposition_daily.csv")
            self.assertEqual(
                list(decomp_df.columns),
                ["ds", "observed", "trend", "seasonal", "residual"],
            )
            self.assertFalse(decomp_df.isna().any().any())

            err = json.loads((out_dir / "error_analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(len(err["by_hour"]), 24)
            self.assertIn("weekday", err["by_day_type"])
            self.assertIn("weekend", err["by_day_type"])
            self.assertTrue(1 <= len(err["top_10_highest_mae_days"]) <= 10)

            imp = json.loads((out_dir / "feature_importance.json").read_text(encoding="utf-8"))
            self.assertEqual(imp["model"], "LightGBM")
            self.assertGreater(len(imp["features"]), 10)


if __name__ == "__main__":
    unittest.main()
