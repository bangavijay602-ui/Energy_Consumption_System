"""Random Forest time-series forecasting model."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class RandomForestForecaster:
    """Random Forest regressor tailored for tabular time-series features."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        """Initialize Random Forest Forecaster.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum tree depth to prevent overfitting.
            min_samples_split: Minimum number of samples required to split an internal node.
            min_samples_leaf: Minimum number of samples required at a leaf node.
            random_state: Seed for reproducibility.
            n_jobs: Number of CPU workers.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.feature_names_: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestForecaster":
        """Train Random Forest on feature matrix X and target y.

        Args:
            X: Feature matrix DataFrame.
            y: Target energy consumption Series.

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
        """Predict energy consumption on input features.

        Args:
            X: Feature matrix.

        Returns:
            1D numpy array of predicted values.
        """
        if not self.is_fitted:
            raise ValueError("RandomForestForecaster must be fitted before predict.")

        X_mat = X[self.feature_names_].values if isinstance(X, pd.DataFrame) and self.feature_names_ else X
        preds = self.model.predict(X_mat)
        return np.asarray(preds, dtype=np.float64)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return dictionary of {feature_name: importance_score}."""
        if not self.is_fitted or not hasattr(self.model, "feature_importances_"):
            return {}
        return {
            feat: round(float(imp), 4)
            for feat, imp in zip(self.feature_names_, self.model.feature_importances_)
        }

    def get_params(self) -> Dict[str, Any]:
        """Return model hyperparameters."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
        }
