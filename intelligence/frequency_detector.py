"""Frequency detection module for determining time-series sampling cadence."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class FrequencyInfo:
    """Telemetry describing time-series cadence and lag configuration."""
    name: str                       # e.g., '15-minute', '30-minute', 'hourly', 'daily', 'weekly', 'irregular'
    pandas_alias: str               # e.g., '15min', '30min', '1h', '1D', '1W'
    median_delta_seconds: float
    periods_per_day: int
    periods_per_week: int
    is_regular: bool
    irregular_gap_percentage: float
    recommended_lag_indices: Dict[str, int]
    resampling_recommended: bool

    def to_dict(self) -> Dict[str, object]:
        """Convert frequency info to dictionary."""
        return {
            "name": self.name,
            "pandas_alias": self.pandas_alias,
            "median_delta_seconds": self.median_delta_seconds,
            "periods_per_day": self.periods_per_day,
            "periods_per_week": self.periods_per_week,
            "is_regular": self.is_regular,
            "irregular_gap_percentage": self.irregular_gap_percentage,
            "recommended_lag_indices": self.recommended_lag_indices,
            "resampling_recommended": self.resampling_recommended,
        }


class FrequencyDetector:
    """Inspects timestamp sequence to detect time step and construct frequency-aware configs."""

    # Standard frequency tolerances (in seconds)
    FREQ_DEFINITIONS = [
        # (name, alias, target_seconds, tolerance_seconds, per_day, per_week)
        ("15-minute", "15min", 900, 60, 96, 672),
        ("30-minute", "30min", 1800, 120, 48, 336),
        ("hourly", "1h", 3600, 300, 24, 168),
        ("daily", "1D", 86400, 3600, 1, 7),
        ("weekly", "1W", 604800, 21600, 0, 1),
    ]

    def __init__(self, regular_tolerance_ratio: float = 0.15) -> None:
        """Initialize detector.
        
        Args:
            regular_tolerance_ratio: Maximum fraction of non-standard deltas before flagging irregular.
        """
        self.regular_tolerance_ratio = regular_tolerance_ratio

    def detect_frequency(self, timestamps: pd.Series) -> FrequencyInfo:
        """Analyze a timestamp series and return FrequencyInfo.

        Args:
            timestamps: Monotonically sorted pd.Series or DatetimeIndex of timestamps.

        Returns:
            FrequencyInfo with frequency name, pandas alias, and lag mappings.
        """
        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            ts = pd.to_datetime(timestamps, errors="coerce").dropna()
        else:
            ts = timestamps.dropna()

        if len(ts) < 3:
            # Fallback for ultra-short series
            return FrequencyInfo(
                name="irregular",
                pandas_alias="1h",
                median_delta_seconds=3600.0,
                periods_per_day=24,
                periods_per_week=168,
                is_regular=False,
                irregular_gap_percentage=100.0,
                recommended_lag_indices={"lag_1": 1, "lag_2": 2, "lag_3": 3, "lag_day": 24, "lag_week": 168},
                resampling_recommended=True,
            )

        deltas = ts.diff().dropna()
        delta_seconds = deltas.dt.total_seconds()
        
        # Filter out 0 delta if duplicates slipped past
        valid_deltas = delta_seconds[delta_seconds > 0]
        if len(valid_deltas) == 0:
            median_sec = 3600.0
        else:
            median_sec = float(valid_deltas.median())

        # Match with standard frequencies
        matched_freq = None
        for name, alias, target_sec, tol_sec, per_day, per_week in self.FREQ_DEFINITIONS:
            if abs(median_sec - target_sec) <= tol_sec:
                matched_freq = (name, alias, target_sec, per_day, per_week)
                break

        if matched_freq:
            name, alias, target_sec, per_day, per_week = matched_freq
            # Measure regularity against the target seconds
            deviations = (delta_seconds < target_sec * 0.9) | (delta_seconds > target_sec * 1.1)
            irregular_pct = round(float(deviations.mean()) * 100.0, 2)
            is_reg = irregular_pct <= (self.regular_tolerance_ratio * 100.0)

            # Build frequency-aware lag indices
            if name == "15-minute":
                lags = {
                    "lag_1": 1,
                    "lag_4": 4,      # 1 hour
                    "lag_8": 8,      # 2 hours
                    "lag_day": 96,   # 1 day
                    "lag_week": 672, # 1 week
                }
            elif name == "30-minute":
                lags = {
                    "lag_1": 1,
                    "lag_2": 2,      # 1 hour
                    "lag_4": 4,      # 2 hours
                    "lag_day": 48,   # 1 day
                    "lag_week": 336, # 1 week
                }
            elif name == "hourly":
                lags = {
                    "lag_1": 1,
                    "lag_2": 2,
                    "lag_3": 3,
                    "lag_day": 24,   # 24h
                    "lag_48": 48,    # 48h
                    "lag_week": 168, # 168h
                }
            elif name == "daily":
                lags = {
                    "lag_1": 1,
                    "lag_2": 2,
                    "lag_week": 7,   # 7d
                    "lag_14": 14,
                    "lag_month": 30, # 30d
                }
            elif name == "weekly":
                lags = {
                    "lag_1": 1,
                    "lag_2": 2,
                    "lag_month": 4,
                }
            else:
                lags = {"lag_1": 1, "lag_2": 2, "lag_day": 24, "lag_week": 168}

            return FrequencyInfo(
                name=name,
                pandas_alias=alias,
                median_delta_seconds=median_sec,
                periods_per_day=per_day,
                periods_per_week=per_week,
                is_regular=is_reg,
                irregular_gap_percentage=irregular_pct,
                recommended_lag_indices=lags,
                resampling_recommended=not is_reg,
            )
        else:
            # Irregular cadence
            est_per_day = max(1, int(round(86400 / max(1.0, median_sec))))
            est_per_week = est_per_day * 7
            return FrequencyInfo(
                name="irregular",
                pandas_alias="1h",
                median_delta_seconds=median_sec,
                periods_per_day=est_per_day,
                periods_per_week=est_per_week,
                is_regular=False,
                irregular_gap_percentage=100.0,
                recommended_lag_indices={
                    "lag_1": 1,
                    "lag_2": 2,
                    "lag_day": est_per_day,
                    "lag_week": est_per_week,
                },
                resampling_recommended=True,
            )
