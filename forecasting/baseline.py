"""Baseline time-series forecasting model (Naive and Seasonal Persistence)."""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


class BaselineForecaster:
    """Naive and Seasonal-Naive baseline forecaster for benchmark comparison."""

    def __init__(self, strategy: str = "last_value", seasonal_period: int = 1) -> None:
        """Initialize baseline forecaster.

        Args:
            strategy: 'last_value' (naive lag_1 persistence) or 'seasonal' (lag_s persistence).
            seasonal_period: Number of steps for seasonal persistence (e.g. 24 for hourly daily cycle).
        """
        self.strategy = strategy
        self.seasonal_period = max(1, seasonal_period)
        self.last_observed_value: Optional[float] = None
        self.recent_history: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineForecaster":
        """Fit baseline on historical data.

        Args:
            X: Feature matrix (unused by naive baseline, kept for sklearn API consistency).
            y: Target energy consumption series.

        Returns:
            self
        """
        y_arr = np.asarray(y, dtype=np.float64)
        if len(y_arr) == 0:
            raise ValueError("Cannot fit BaselineForecaster on empty target series.")

        self.last_observed_value = float(y_arr[-1])
        # Keep sufficient recent history for seasonal lookup
        self.recent_history = y_arr[-self.seasonal_period:] if len(y_arr) >= self.seasonal_period else y_arr
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict energy consumption for feature rows.

        If X contains a direct lag feature matching the baseline (e.g. 'lag_1' or 'lag_day'),
        uses that feature row-wise; otherwise falls back to last observed value.

        Args:
            X: Feature matrix.

        Returns:
            1D numpy array of predicted values.
        """
        if not self.is_fitted or self.last_observed_value is None:
            raise ValueError("BaselineForecaster must be fitted before predict.")

        n_samples = len(X)
        if n_samples == 0:
            return np.array([], dtype=np.float64)

        if self.strategy == "last_value" and "lag_1" in X.columns:
            return X["lag_1"].values.astype(np.float64)
        elif self.strategy == "seasonal" and "lag_day" in X.columns:
            return X["lag_day"].values.astype(np.float64)
        elif self.strategy == "seasonal" and f"lag_{self.seasonal_period}" in X.columns:
            return X[f"lag_{self.seasonal_period}"].values.astype(np.float64)
        else:
            # Constant repeat of last observed value
            return np.full(shape=(n_samples,), fill_value=self.last_observed_value, dtype=np.float64)

    def get_params(self) -> Dict[str, Any]:
        """Return model hyperparameters."""
        return {
            "strategy": self.strategy,
            "seasonal_period": self.seasonal_period,
        }
