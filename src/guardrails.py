"""Runtime validation, path safety, and i18n guardrails."""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from babel.numbers import format_decimal
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
)

LocaleCode = Literal["en", "id"]
_REGION_RE = re.compile(r"^[A-Z0-9_]{2,16}$")
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "invalid_region": "Region must be 2-16 alphanumeric/underscore chars.",
        "path_outside_data": "Data path must stay within the data directory.",
        "forecast_ready": "Forecast request validated.",
    },
    "id": {
        "invalid_region": "Wilayah harus 2-16 karakter alfanumerik/garis bawah.",
        "path_outside_data": "Jalur data harus berada di dalam direktori data.",
        "forecast_ready": "Permintaan prakiraan telah divalidasi.",
    },
}


def _validate_region(value: str) -> str:
    normalized = value.strip().upper()
    if not _REGION_RE.fullmatch(normalized):
        raise ValueError(_MESSAGES["en"]["invalid_region"])
    return normalized


def _reject_bool(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid numeric demand.")
    return value


RegionCode = Annotated[str, AfterValidator(_validate_region)]


class ForecastRequest(BaseModel):
    """Validated parameters for a demand forecast request."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    region: RegionCode
    horizon_hours: int = Field(default=24, ge=1, le=168)
    locale: LocaleCode = "en"


class EnergyRecord(BaseModel):
    """Validated hourly electricity demand observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    timestamp: datetime
    region: RegionCode
    demand_mw: Annotated[
        float,
        BeforeValidator(_reject_bool),
        Field(ge=0.0, allow_inf_nan=False),
    ]


def resolve_safe_data_path(
    filename: str, data_dir: Path | None = None, max_bytes: int = 500_000_000
) -> Path:
    """Resolve a relative filename safely inside ``data_dir``.

    Args:
        filename: Relative file path within ``data_dir``.
        data_dir: Base directory override; defaults to project ``data/``.
        max_bytes: Maximum allowed size in bytes for existing files.

    Returns:
        Resolved ``Path`` inside ``data_dir``.

    Raises:
        ValueError: On empty, null-byte, absolute, traversing, dir, or oversized paths.
    """
    if not filename or "\x00" in filename or not filename.strip():
        raise ValueError("Filename must be non-empty without null bytes.")
    candidate = Path(filename.strip())
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise ValueError(_MESSAGES["en"]["path_outside_data"])
    base = (data_dir if data_dir is not None else _DEFAULT_DATA_DIR).resolve()
    resolved = (base / candidate).resolve()
    if resolved == base or not resolved.is_relative_to(base):
        raise ValueError(_MESSAGES["en"]["path_outside_data"])
    if resolved.is_dir():
        raise ValueError("Data path must reference a file, not a directory.")
    if resolved.is_file() and resolved.stat().st_size > max_bytes:
        raise ValueError(f"File exceeds {max_bytes} byte limit.")
    return resolved


def format_energy_mw(value: float, locale: str = "en") -> str:
    """Format a megawatt value using Babel locale conventions."""
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("Energy value must be a finite number.")
    if locale not in _MESSAGES:
        raise ValueError(f"Unsupported locale: {locale}")
    return f"{format_decimal(value, locale=locale)} MW"


def get_message(key: str, locale: str = "en") -> str:
    """Return a localized guardrail status or error message."""
    if locale not in _MESSAGES:
        raise ValueError(f"Unsupported locale: {locale}")
    if key not in _MESSAGES[locale]:
        raise KeyError(f"Unknown message key: {key}")
    return _MESSAGES[locale][key]
