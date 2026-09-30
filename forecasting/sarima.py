"""Seasonal AutoRegressive Integrated Moving Average (SARIMA) model."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    from sklearn.linear_model import Ridge


class SARIMAForecaster:
    """SARIMA state-space forecaster for capturing autoregressive and seasonal dynamics."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 0, 0, 24),
        use_exog: bool = False,
    ) -> None:
        """Initialize SARIMA forecaster.

        Args:
            order: (p, d, q) non-seasonal ARIMA specification.
            seasonal_order: (P, D, Q, s) seasonal ARIMA specification.
            use_exog: Whether to pass feature matrix X as exogenous variables.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_exog = use_exog
        self.fitted_model_ = None
        self.last_training_index_ = 0
        self.is_fitted = False
        self._fallback_ridge = None
        self.last_y_val_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SARIMAForecaster":
        """Fit SARIMA on target series y (and optional exogenous X).

        Args:
            X: Feature matrix DataFrame.
            y: Target energy consumption Series.

        Returns:
            self
        """
        y_vec = y.values if isinstance(y, pd.Series) else np.asarray(y, dtype=np.float64)
        self.last_training_index_ = len(y_vec)
        self.last_y_val_ = float(y_vec[-1]) if len(y_vec) > 0 else 0.0

        if HAS_STATSMODELS:
            exog_data = X.values if (self.use_exog and X is not None and len(X.columns) > 0) else None
            try:
                model = SARIMAX(
                    endog=y_vec,
                    exog=exog_data,
                    order=self.order,
                    seasonal_order=self.seasonal_order if self.seasonal_order[3] > 1 else (0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self.fitted_model_ = model.fit(disp=False, maxiter=50)
            except Exception:
                # Fallback to simple (1,1,0) if complex order fails to converge
                try:
                    fallback_model = SARIMAX(
                        endog=y_vec,
                        order=(1, 1, 0),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    self.fitted_model_ = fallback_model.fit(disp=False, maxiter=30)
                except Exception:
                    # Fallback to ridge AR
                    self._fit_fallback_ridge(X, y_vec)
        else:
            self._fit_fallback_ridge(X, y_vec)

        self.is_fitted = True
        return self

    def _fit_fallback_ridge(self, X: pd.DataFrame, y_vec: np.ndarray) -> None:
        """Fallback linear autoregressive regressor when statsmodels is unavailable."""
        self._fallback_ridge = Ridge(alpha=1.0)
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        self._fallback_ridge.fit(X_mat, y_vec)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict target values for feature matrix X.

        Args:
            X: Feature matrix.

        Returns:
            1D numpy array of predictions.
        """
        if not self.is_fitted:
            raise ValueError("SARIMAForecaster must be fitted before predict.")

        n_steps = len(X)
        if n_steps == 0:
            return np.array([], dtype=np.float64)

        if self.fitted_model_ is not None:
            exog_data = X.values if (self.use_exog and X is not None and len(X.columns) > 0) else None
            try:
                forecast_res = self.fitted_model_.forecast(steps=n_steps, exog=exog_data)
                preds = np.asarray(forecast_res, dtype=np.float64)
                # Ensure no NaNs in forecast
                if np.isnan(preds).any():
                    preds = np.nan_to_num(preds, nan=self.last_y_val_)
                return preds
            except Exception:
                pass

        if self._fallback_ridge is not None:
            X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            return self._fallback_ridge.predict(X_mat).astype(np.float64)

        # Fallback to last value
        return np.full(shape=(n_steps,), fill_value=self.last_y_val_, dtype=np.float64)

    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "use_exog": self.use_exog,
            "backend": "statsmodels_sarimax" if HAS_STATSMODELS and self.fitted_model_ is not None else "ridge_ar_fallback",
        }
