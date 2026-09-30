"""Time-series feature engineering module with frequency adaptation and zero data leakage."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from intelligence.frequency_detector import FrequencyInfo


@dataclass
class FeatureEngineeringConfig:
    """Configuration and metadata for reproducible feature transformations."""
    frequency_name: str
    lag_indices: Dict[str, int]
    rolling_windows: Dict[str, int]
    calendar_features: List[str]
    cyclic_features: List[str]
    weather_features: List[str]
    feature_columns: List[str] = field(default_factory=list)
    warmup_rows: int = 0


class TimeSeriesFeatureEngineer:
    """Generates calendar, cyclic, lag, and rolling features adapted to time-series frequency."""

    def __init__(self, freq_info: FrequencyInfo) -> None:
        """Initialize feature engineer with detected frequency intelligence.

        Args:
            freq_info: FrequencyInfo describing data cadence and recommended lag steps.
        """
        self.freq_info = freq_info
        self.lag_indices = freq_info.recommended_lag_indices
        
        # Calculate adaptive rolling window sizes (e.g. short = 6h or 1/4 day, long = 1 day or 1 week)
        per_day = max(1, freq_info.periods_per_day)
        per_week = max(7, freq_info.periods_per_week)

        if freq_info.name in ["15-minute", "30-minute", "hourly"]:
            short_win = max(3, per_day // 4)     # ~6 hours
            long_win = per_day                  # 1 day
        elif freq_info.name == "daily":
            short_win = 3                       # 3 days
            long_win = 7                        # 1 week
        else:
            short_win = 2
            long_win = max(4, per_day)

        self.rolling_windows = {
            "short": short_win,
            "long": long_win,
        }

        self.calendar_features = [
            "hour",
            "day",
            "day_of_week",
            "month",
            "quarter",
            "year",
            "is_weekend",
        ]

        self.cyclic_features = [
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
        ]

        # Calculate max warmup window required for lags and rolling
        max_lag = max(self.lag_indices.values()) if self.lag_indices else 1
        max_roll = max(self.rolling_windows.values()) if self.rolling_windows else 1
        self.warmup_rows = max(max_lag, max_roll + 1)
        self.config: Optional[FeatureEngineeringConfig] = None

    def _extract_calendar_and_cyclic(self, timestamps: pd.Series) -> pd.DataFrame:
        """Extract calendar and sinusoidal cyclical features from timestamps."""
        ts = pd.to_datetime(timestamps)
        df_cal = pd.DataFrame(index=ts.index)

        # Calendar
        df_cal["hour"] = ts.dt.hour
        df_cal["day"] = ts.dt.day
        df_cal["day_of_week"] = ts.dt.dayofweek
        df_cal["month"] = ts.dt.month
        df_cal["quarter"] = ts.dt.quarter
        df_cal["year"] = ts.dt.year
        df_cal["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

        # Cyclic encodings
        hour_rad = 2 * np.pi * df_cal["hour"] / 24.0
        df_cal["hour_sin"] = np.sin(hour_rad)
        df_cal["hour_cos"] = np.cos(hour_rad)

        weekday_rad = 2 * np.pi * df_cal["day_of_week"] / 7.0
        df_cal["weekday_sin"] = np.sin(weekday_rad)
        df_cal["weekday_cos"] = np.cos(weekday_rad)

        month_rad = 2 * np.pi * (df_cal["month"] - 1) / 12.0
        df_cal["month_sin"] = np.sin(month_rad)
        df_cal["month_cos"] = np.cos(month_rad)

        return df_cal

    def create_features(
        self,
        df: pd.DataFrame,
        drop_warmup: bool = True,
    ) -> Tuple[pd.DataFrame, FeatureEngineeringConfig]:
        """Generate full feature matrix from canonical DataFrame.

        Strictly avoids lookahead data leakage by shifting rolling statistics by 1.

        Args:
            df: Normalized DataFrame containing 'timestamp', 'consumption', and optional weather.
            drop_warmup: Whether to drop initial rows with NaN created by lag/rolling windows.

        Returns:
            Tuple of (Featured DataFrame, FeatureEngineeringConfig).
        """
        feat_df = df.copy()

        # 1. Calendar and Cyclic Features
        cal_cyclic = self._extract_calendar_and_cyclic(feat_df["timestamp"])
        for col in cal_cyclic.columns:
            feat_df[col] = cal_cyclic[col]

        # 2. Lag Features (Frequency-aware)
        for lag_name, lag_steps in self.lag_indices.items():
            feat_df[lag_name] = feat_df["consumption"].shift(lag_steps)

        # 3. Rolling Features (Strictly shifted by 1 to prevent current-step target leakage)
        target_shifted = feat_df["consumption"].shift(1)
        
        short_w = self.rolling_windows["short"]
        long_w = self.rolling_windows["long"]

        feat_df[f"rolling_mean_{short_w}"] = target_shifted.rolling(window=short_w, min_periods=1).mean()
        feat_df[f"rolling_std_{short_w}"] = target_shifted.rolling(window=short_w, min_periods=1).std().fillna(0.0)
        
        feat_df[f"rolling_mean_{long_w}"] = target_shifted.rolling(window=long_w, min_periods=1).mean()
        feat_df[f"rolling_std_{long_w}"] = target_shifted.rolling(window=long_w, min_periods=1).std().fillna(0.0)
        feat_df[f"rolling_min_{long_w}"] = target_shifted.rolling(window=long_w, min_periods=1).min()
        feat_df[f"rolling_max_{long_w}"] = target_shifted.rolling(window=long_w, min_periods=1).max()

        # 4. Optional Environmental / Weather Columns
        optional_cols = ["temperature", "humidity", "wind_speed", "pressure", "holiday"]
        active_weather = [c for c in optional_cols if c in feat_df.columns]

        # Add small lags for active weather features if present
        for w_col in active_weather:
            feat_df[f"{w_col}_lag_1"] = feat_df[w_col].shift(1)

        # 5. Determine Feature Columns
        exclude_cols = {"timestamp", "consumption"}
        feature_cols = [c for c in feat_df.columns if c not in exclude_cols]

        config = FeatureEngineeringConfig(
            frequency_name=self.freq_info.name,
            lag_indices=self.lag_indices,
            rolling_windows=self.rolling_windows,
            calendar_features=self.calendar_features,
            cyclic_features=self.cyclic_features,
            weather_features=active_weather,
            feature_columns=feature_cols,
            warmup_rows=self.warmup_rows,
        )
        self.config = config

        if drop_warmup:
            feat_df = feat_df.iloc[self.warmup_rows:].reset_index(drop=True)

        return feat_df, config

    def generate_single_step_features(
        self,
        history_df: pd.DataFrame,
        next_timestamp: pd.Timestamp,
        future_weather: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Construct a 1-row feature vector for forecasting step t+k using accumulated history.

        Args:
            history_df: Historical records buffer containing ['timestamp', 'consumption', ...].
            next_timestamp: The target future timestamp to predict.
            future_weather: Optional dictionary of future weather forecasts. If None, forward-fills
                            from the latest historical record.

        Returns:
            A 1-row DataFrame containing all feature columns matching self.config.feature_columns.
        """
        if self.config is None:
            raise ValueError("Feature engineer has not been fitted. Call create_features first.")

        row_dict: Dict[str, float] = {}

        # 1. Calendar & Cyclic
        cal_cyclic = self._extract_calendar_and_cyclic(pd.Series([next_timestamp]))
        for col in cal_cyclic.columns:
            row_dict[col] = float(cal_cyclic[col].iloc[0])

        # 2. Lag Features
        consumption_series = history_df["consumption"].values
        n_hist = len(consumption_series)

        for lag_name, lag_steps in self.config.lag_indices.items():
            if n_hist >= lag_steps:
                row_dict[lag_name] = float(consumption_series[-lag_steps])
            else:
                row_dict[lag_name] = float(consumption_series[0])

        # 3. Rolling Features (Computed on past history buffer ending at t-1)
        short_w = self.config.rolling_windows["short"]
        long_w = self.config.rolling_windows["long"]

        short_hist = consumption_series[-short_w:] if n_hist >= short_w else consumption_series
        long_hist = consumption_series[-long_w:] if n_hist >= long_w else consumption_series

        row_dict[f"rolling_mean_{short_w}"] = float(np.mean(short_hist))
        row_dict[f"rolling_std_{short_w}"] = float(np.std(short_hist)) if len(short_hist) > 1 else 0.0

        row_dict[f"rolling_mean_{long_w}"] = float(np.mean(long_hist))
        row_dict[f"rolling_std_{long_w}"] = float(np.std(long_hist)) if len(long_hist) > 1 else 0.0
        row_dict[f"rolling_min_{long_w}"] = float(np.min(long_hist))
        row_dict[f"rolling_max_{long_w}"] = float(np.max(long_hist))

        # 4. Weather Features
        for w_col in self.config.weather_features:
            if future_weather and w_col in future_weather:
                val = float(future_weather[w_col])
            elif w_col in history_df.columns:
                # Forward-fill latest known weather observation
                val = float(history_df[w_col].iloc[-1])
            else:
                val = 0.0
            row_dict[w_col] = val
            
            # Weather lag
            if w_col in history_df.columns:
                row_dict[f"{w_col}_lag_1"] = float(history_df[w_col].iloc[-1])
            else:
                row_dict[f"{w_col}_lag_1"] = val

        # Assemble into DataFrame with exact feature columns order
        feat_row = pd.DataFrame([row_dict])[self.config.feature_columns]
        return feat_row
