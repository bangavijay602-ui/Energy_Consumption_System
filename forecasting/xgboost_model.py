"""XGBoost time-series forecasting model with gradient boosting support."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import HistGradientBoostingRegressor


class XGBoostForecaster:
    """Extreme Gradient Boosting (XGBoost) forecaster for time-series regression."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        """Initialize XGBoost Forecaster.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate (shrinkage).
            subsample: Subsample ratio of training instances.
            colsample_bytree: Subsample ratio of columns per tree.
            random_state: Random number seed.
            n_jobs: Number of parallel threads.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.feature_names_: List[str] = []
        self.is_fitted = False

        if HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                tree_method="hist",
            )
        else:
            self.model = HistGradientBoostingRegressor(
                max_iter=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostForecaster":
        """Fit model on feature matrix X and target y.

        Args:
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            self
        """
        self.feature_names_ = list(X.columns)
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_vec = y.values if isinstance(y, pd.Series) else np.asarray(y)

        self.model.fit(X_mat, y_vec)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict target energy consumption.

        Args:
            X: Feature matrix DataFrame.

        Returns:
            1D numpy array of predictions.
        """
        if not self.is_fitted:
            raise ValueError("XGBoostForecaster must be fitted before predict.")

        X_mat = X[self.feature_names_].values if isinstance(X, pd.DataFrame) and self.feature_names_ else X
        preds = self.model.predict(X_mat)
        return np.asarray(preds, dtype=np.float64)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return feature importances if supported."""
        if not self.is_fitted:
            return {}
        if hasattr(self.model, "feature_importances_"):
            return {
                feat: round(float(imp), 4)
                for feat, imp in zip(self.feature_names_, self.model.feature_importances_)
            }
        return {}

    def get_params(self) -> Dict[str, Any]:
        """Return model hyperparameters."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "backend": "xgboost" if HAS_XGBOOST else "hist_gradient_boosting",
        }
