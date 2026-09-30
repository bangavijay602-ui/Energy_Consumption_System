"""Evaluation metrics and model comparison for time-series forecasting."""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class EvaluationMetrics:
    """Standardized performance metrics for a forecasting model."""
    model_name: str
    mae: float
    rmse: float
    mape: float
    wape: float
    r2: float
    sample_count: int

    def to_dict(self) -> Dict[str, Union[str, float, int]]:
        """Convert metrics to dictionary."""
        return asdict(self)


class ModelEvaluator:
    """Calculates time-series regression metrics with safe zero handling."""

    def __init__(self, epsilon: float = 1e-5) -> None:
        """Initialize evaluator.

        Args:
            epsilon: Minimum denominator threshold to prevent division by zero in MAPE.
        """
        self.epsilon = epsilon

    def evaluate(
        self,
        y_true: Union[pd.Series, np.ndarray, List[float]],
        y_pred: Union[pd.Series, np.ndarray, List[float]],
        model_name: str = "Model",
    ) -> EvaluationMetrics:
        """Compute MAE, RMSE, epsilon-safe MAPE, WAPE, and R2.

        Args:
            y_true: Ground truth target values.
            y_pred: Predicted values from forecaster.
            model_name: Identifier for the evaluated model.

        Returns:
            EvaluationMetrics object containing computed metrics.
        """
        yt = np.asarray(y_true, dtype=np.float64).ravel()
        yp = np.asarray(y_pred, dtype=np.float64).ravel()

        if len(yt) != len(yp):
            raise ValueError(f"Length mismatch: y_true ({len(yt)}) vs y_pred ({len(yp)}).")
        if len(yt) == 0:
            raise ValueError("Cannot evaluate metrics on empty arrays.")

        errors = yt - yp
        abs_errors = np.abs(errors)
        sq_errors = np.square(errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(sq_errors)))

        # Epsilon-safe MAPE
        safe_denom = np.maximum(np.abs(yt), self.epsilon)
        mape = float(np.mean(abs_errors / safe_denom) * 100.0)

        # Weighted Absolute Percentage Error (WAPE = sum(|yt - yp|) / sum(|yt|))
        sum_yt = float(np.sum(np.abs(yt)))
        wape = float((np.sum(abs_errors) / max(sum_yt, self.epsilon)) * 100.0)

        # R-squared
        ss_tot = float(np.sum(np.square(yt - np.mean(yt))))
        ss_res = float(np.sum(sq_errors))
        r2 = float(1.0 - (ss_res / max(ss_tot, self.epsilon))) if ss_tot > 0 else 0.0

        return EvaluationMetrics(
            model_name=model_name,
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            mape=round(mape, 2),
            wape=round(wape, 2),
            r2=round(r2, 4),
            sample_count=len(yt),
        )

    def create_comparison_table(
        self, metrics_list: List[EvaluationMetrics]
    ) -> pd.DataFrame:
        """Create a formatted comparison DataFrame sorted by best MAE.

        Args:
            metrics_list: List of EvaluationMetrics objects.

        Returns:
            pandas DataFrame sorted by MAE ascending.
        """
        rows = []
        for m in metrics_list:
            rows.append({
                "Model": m.model_name,
                "MAE": m.mae,
                "RMSE": m.rmse,
                "MAPE (%)": f"{m.mape:.2f}%",
                "WAPE (%)": f"{m.wape:.2f}%",
                "R2": m.r2,
            })
        df_comp = pd.DataFrame(rows)
        return df_comp.sort_values("MAE").reset_index(drop=True)
