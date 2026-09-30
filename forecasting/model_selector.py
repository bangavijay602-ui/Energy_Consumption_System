"""Model selection and retraining module for time-series forecasting."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .evaluator import ModelEvaluator, EvaluationMetrics
from .baseline import BaselineForecaster
from .random_forest import RandomForestForecaster
from .xgboost_model import XGBoostForecaster
from .sarima import SARIMAForecaster


@dataclass
class ModelSelectionResult:
    """Result of validation-based model selection, history retraining, and test scoring."""
    selected_model_name: str
    selected_model_instance: Any
    validation_metrics: Dict[str, EvaluationMetrics]
    test_metrics: EvaluationMetrics
    comparison_table: pd.DataFrame
    selection_criterion: str = "MAE"


class ModelSelector:
    """Trains candidate models, compares on validation split, and retrains the winner."""

    def __init__(
        self,
        evaluator: Optional[ModelEvaluator] = None,
        selection_metric: str = "mae",
    ) -> None:
        """Initialize ModelSelector.

        Args:
            evaluator: ModelEvaluator instance.
            selection_metric: Metric to optimize on validation set ('mae', 'rmse', 'mape').
        """
        self.evaluator = evaluator or ModelEvaluator()
        self.selection_metric = selection_metric.lower()

    def get_default_candidates(self, seasonal_period: int = 24) -> Dict[str, Any]:
        """Instantiate candidate forecasting models.

        Args:
            seasonal_period: Lag steps corresponding to 1 day / dominant seasonality.

        Returns:
            Dictionary of {model_name: model_instance}.
        """
        return {
            "Naive Baseline": BaselineForecaster(strategy="last_value", seasonal_period=seasonal_period),
            "Random Forest": RandomForestForecaster(n_estimators=100, max_depth=12, random_state=42),
            "XGBoost": XGBoostForecaster(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42),
            "SARIMA": SARIMAForecaster(
                order=(1, 1, 1),
                seasonal_order=(1, 0, 0, min(seasonal_period, 24)),
                use_exog=False,
            ),
        }

    def select_and_retrain(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        candidate_models: Optional[Dict[str, Any]] = None,
        seasonal_period: int = 24,
    ) -> ModelSelectionResult:
        """Train candidates, select top performer on validation set, retrain on train+val, and test.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            X_test: Test features.
            y_test: Test target.
            candidate_models: Dict of models to compete. If None, uses defaults.
            seasonal_period: Seasonal period for baselines and SARIMA.

        Returns:
            ModelSelectionResult with retrained model, validation leaderboard, and final test score.
        """
        models = candidate_models or self.get_default_candidates(seasonal_period=seasonal_period)

        val_metrics: Dict[str, EvaluationMetrics] = {}
        val_predictions: Dict[str, np.ndarray] = {}

        # 1. Train and evaluate on Validation Set
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds_val = model.predict(X_val)
                metrics = self.evaluator.evaluate(y_true=y_val, y_pred=preds_val, model_name=name)
                val_metrics[name] = metrics
                val_predictions[name] = preds_val
            except Exception as e:
                # Log error and assign high penalty
                val_metrics[name] = EvaluationMetrics(
                    model_name=name,
                    mae=float("inf"),
                    rmse=float("inf"),
                    mape=float("inf"),
                    wape=float("inf"),
                    r2=-1.0,
                    sample_count=len(y_val),
                )

        # 2. Select Best Model based on chosen validation metric
        comparison_table = self.evaluator.create_comparison_table(list(val_metrics.values()))
        
        best_name = None
        best_score = float("inf")
        for name, metrics in val_metrics.items():
            metric_val = getattr(metrics, self.selection_metric, metrics.mae)
            if metric_val < best_score:
                best_score = metric_val
                best_name = name

        if best_name is None:
            best_name = "Naive Baseline"

        # 3. Retrain selected model on all available historical data (Train + Validation)
        X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_train_val = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

        retrained_model = self.get_default_candidates(seasonal_period=seasonal_period)[best_name]
        retrained_model.fit(X_train_val, y_train_val)

        # 4. Evaluate retrained model on held-out Test Set
        test_preds = retrained_model.predict(X_test)
        test_metrics = self.evaluator.evaluate(y_true=y_test, y_pred=test_preds, model_name=best_name)

        return ModelSelectionResult(
            selected_model_name=best_name,
            selected_model_instance=retrained_model,
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            comparison_table=comparison_table,
            selection_criterion=self.selection_metric.upper(),
        )
