"""Unit tests for src.guardrails."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from src.guardrails import (
    EnergyRecord,
    ForecastRequest,
    format_energy_mw,
    get_message,
    resolve_safe_data_path,
)


class TestForecastRequest(unittest.TestCase):
    """Tests for ForecastRequest validation."""

    def test_valid_normalizes_region_and_defaults(self) -> None:
        req = ForecastRequest(region=" pjm_east ")
        self.assertEqual(req.region, "PJM_EAST")
        self.assertEqual(req.horizon_hours, 24)
        self.assertEqual(req.locale, "en")

    def test_valid_explicit_fields(self) -> None:
        req = ForecastRequest(region="aep1", horizon_hours=168, locale="id")
        self.assertEqual(req.region, "AEP1")
        self.assertEqual(req.horizon_hours, 168)
        self.assertEqual(req.locale, "id")

    def test_invalid_region_rejected(self) -> None:
        for bad_region in ("", "A", "A" * 17, "PJM-EAST", "PJM EAST"):
            with self.subTest(bad_region=bad_region):
                with self.assertRaises(ValidationError):
                    ForecastRequest(region=bad_region)

    def test_invalid_horizon_rejected(self) -> None:
        for bad_horizon in (0, -5, 169):
            with self.subTest(bad_horizon=bad_horizon):
                with self.assertRaises(ValidationError):
                    ForecastRequest(region="PJME", horizon_hours=bad_horizon)

    def test_invalid_locale_and_extra_forbid(self) -> None:
        with self.assertRaises(ValidationError):
            ForecastRequest(region="PJME", locale="fr")  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            ForecastRequest(region="PJME", unknown="x")  # type: ignore[call-arg]


class TestEnergyRecord(unittest.TestCase):
    """Tests for EnergyRecord validation."""

    def test_valid_record(self) -> None:
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        rec = EnergyRecord(timestamp=ts, region=" pjme ", demand_mw=31250.5)
        self.assertEqual(rec.timestamp, ts)
        self.assertEqual(rec.region, "PJME")
        self.assertEqual(rec.demand_mw, 31250.5)

    def test_zero_demand_allowed(self) -> None:
        ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        rec = EnergyRecord(timestamp=ts, region="PJME", demand_mw=0.0)
        self.assertEqual(rec.demand_mw, 0.0)

    def test_invalid_region_rejected(self) -> None:
        ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        for bad_region in ("", "   ", "A", "A" * 17, "PJM-EAST"):
            with self.subTest(bad_region=bad_region):
                with self.assertRaises(ValidationError):
                    EnergyRecord(timestamp=ts, region=bad_region, demand_mw=10.0)

    def test_invalid_demand_rejected(self) -> None:
        ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        for bad_mw in (-0.1, float("nan"), float("inf"), float("-inf"), True, False):
            with self.subTest(bad_mw=bad_mw):
                with self.assertRaises(ValidationError):
                    EnergyRecord(timestamp=ts, region="PJME", demand_mw=bad_mw)


class TestResolveSafeDataPath(unittest.TestCase):
    """Tests for resolve_safe_data_path."""

    def test_valid_relative_and_default_dir(self) -> None:
        resolved = resolve_safe_data_path("pjm/PJME_hourly.csv")
        expected_base = Path(__file__).resolve().parent.parent / "data"
        self.assertEqual(resolved, (expected_base / "pjm" / "PJME_hourly.csv").resolve())

    def test_rejects_traversal_null_and_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            abs_target = str((base / "file.csv").resolve())
            bad_names = (
                "",
                "   ",
                ".",
                "..",
                "../out.csv",
                "a/../../out.csv",
                "bad\x00.csv",
                abs_target,
            )
            for bad_name in bad_names:
                with self.subTest(bad_name=bad_name):
                    with self.assertRaises(ValueError):
                        resolve_safe_data_path(bad_name, data_dir=base)

    def test_rejects_directory_and_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            subdir = base / "nested"
            subdir.mkdir()
            with self.assertRaises(ValueError):
                resolve_safe_data_path("nested", data_dir=base)

            outside = base.parent / "outside.csv"
            real_resolve = Path.resolve

            def fake_resolve(path_obj: Path) -> Path:
                if path_obj.name == "link.csv":
                    return outside
                return real_resolve(path_obj)

            with patch.object(Path, "resolve", fake_resolve):
                with self.assertRaises(ValueError):
                    resolve_safe_data_path("link.csv", data_dir=base)

    def test_rejects_oversized_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sample = base / "sample.csv"
            sample.write_bytes(b"0123456789")
            self.assertEqual(resolve_safe_data_path("sample.csv", data_dir=base, max_bytes=10), sample.resolve())
            with self.assertRaises(ValueError):
                resolve_safe_data_path("sample.csv", data_dir=base, max_bytes=5)


class TestBabelI18n(unittest.TestCase):
    """Tests for Babel formatting and localized messages."""

    def test_format_energy_mw(self) -> None:
        self.assertEqual(format_energy_mw(1234.5, "en"), "1,234.5 MW")
        self.assertEqual(format_energy_mw(1234.5, "id"), "1.234,5 MW")

    def test_format_energy_mw_rejects_bad_inputs(self) -> None:
        for bad_val in (float("nan"), float("inf"), True, False):
            with self.subTest(bad_val=bad_val):
                with self.assertRaises(ValueError):
                    format_energy_mw(bad_val, "en")
        with self.assertRaises(ValueError):
            format_energy_mw(10.0, "es")

    def test_get_message(self) -> None:
        for key in ("invalid_region", "path_outside_data", "forecast_ready"):
            en_msg = get_message(key, "en")
            id_msg = get_message(key, "id")
            self.assertTrue(en_msg)
            self.assertTrue(id_msg)
            self.assertNotEqual(en_msg, id_msg)
        with self.assertRaises(ValueError):
            get_message("forecast_ready", "es")
        with self.assertRaises(KeyError):
            get_message("missing_key", "en")


if __name__ == "__main__":
    unittest.main()
