"""Genuine recursive multi-step future forecasting engine."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from preprocessing.features import TimeSeriesFeatureEngineer
from intelligence.frequency_detector import FrequencyInfo


@dataclass
class ForecastPoint:
    """A single predicted future timestamp and consumption point."""
    timestamp: str
    predicted_consumption: float
    step_ahead: int


@dataclass
class ForecastResult:
    """Full multi-step future forecast result."""
    model_name: str
    horizon: int
    start_timestamp: str
    end_timestamp: str
    forecasts: List[ForecastPoint]
    forecast_df: pd.DataFrame = field(repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert forecast result to dictionary."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "forecasts": [
                {
                    "timestamp": p.timestamp,
                    "predicted_consumption": round(p.predicted_consumption, 4),
                    "step_ahead": p.step_ahead,
                }
                for p in self.forecasts
            ],
        }


class FutureForecaster:
    """Generates out-of-sample multi-step recursive forecasts with dynamic feature reconstruction."""

    def __init__(
        self,
        model: Any,
        feature_engineer: TimeSeriesFeatureEngineer,
        freq_info: FrequencyInfo,
    ) -> None:
        """Initialize FutureForecaster.

        Args:
            model: Trained forecasting model (e.g. retrained best model).
            feature_engineer: Fitted TimeSeriesFeatureEngineer instance.
            freq_info: Detected FrequencyInfo.
        """
        self.model = model
        self.feature_engineer = feature_engineer
        self.freq_info = freq_info

    def generate_forecast(
        self,
        history_df: pd.DataFrame,
        horizon: int = 24,
        future_weather: Optional[List[Dict[str, float]]] = None,
    ) -> ForecastResult:
        """Execute recursive multi-step forecasting into the true future.

        Args:
            history_df: Historical canonical DataFrame (must contain 'timestamp', 'consumption', ...).
            horizon: Number of future time steps to predict.
            future_weather: Optional list of weather dictionaries for each step 1..horizon.

        Returns:
            ForecastResult with step-by-step predictions and timestamps.
        """
        if len(history_df) == 0:
            raise ValueError("History DataFrame cannot be empty.")

        # Copy history buffer
        buffer_df = history_df.copy().reset_index(drop=True)
        buffer_df["timestamp"] = pd.to_datetime(buffer_df["timestamp"])
        
        last_timestamp = buffer_df["timestamp"].iloc[-1]
        delta = pd.to_timedelta(self.freq_info.median_delta_seconds, unit="s")

        forecast_points: List[ForecastPoint] = []
        forecast_rows: List[Dict[str, Any]] = []

        curr_ts = last_timestamp

        for step in range(1, horizon + 1):
            curr_ts = curr_ts + delta
            
            step_weather = future_weather[step - 1] if (future_weather and step - 1 < len(future_weather)) else None

            # 1. Generate 1-row feature vector dynamically from current history buffer (including prior predictions)
            feat_row = self.feature_engineer.generate_single_step_features(
                history_df=buffer_df,
                next_timestamp=curr_ts,
                future_weather=step_weather,
            )

            # 2. Predict next consumption value
            pred_arr = self.model.predict(feat_row)
            pred_val = float(pred_arr[0])

            # In physical power systems, consumption is rarely arbitrarily negative unless small generation offset
            pred_val_cleaned = max(0.0, pred_val)

            # 3. Append new prediction point to forecast collection
            forecast_points.append(
                ForecastPoint(
                    timestamp=curr_ts.isoformat(),
                    predicted_consumption=pred_val_cleaned,
                    step_ahead=step,
                )
            )

            forecast_rows.append({
                "timestamp": curr_ts,
                "predicted_consumption": pred_val_cleaned,
                "step_ahead": step,
            })

            # 4. Append the predicted observation to buffer_df for the next step's lags and rollings!
            new_buffer_row: Dict[str, Any] = {
                "timestamp": curr_ts,
                "consumption": pred_val_cleaned,
            }

            # Forward-fill or use supplied weather in buffer
            for w_col in self.feature_engineer.config.weather_features:
                if step_weather and w_col in step_weather:
                    new_buffer_row[w_col] = step_weather[w_col]
                elif w_col in buffer_df.columns:
                    new_buffer_row[w_col] = buffer_df[w_col].iloc[-1]
                else:
                    new_buffer_row[w_col] = 0.0

            buffer_df = pd.concat([buffer_df, pd.DataFrame([new_buffer_row])], ignore_index=True)

        forecast_df = pd.DataFrame(forecast_rows)
        model_name = getattr(self.model, "__class__", type(self.model)).__name__

        return ForecastResult(
            model_name=model_name,
            horizon=horizon,
            start_timestamp=forecast_points[0].timestamp if forecast_points else "",
            end_timestamp=forecast_points[-1].timestamp if forecast_points else "",
            forecasts=forecast_points,
            forecast_df=forecast_df,
        )
