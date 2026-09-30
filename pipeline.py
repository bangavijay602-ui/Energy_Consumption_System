"""End-to-End Production ML Training & Forecasting Pipeline."""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import pandas as pd
import numpy as np

from intelligence.profiler import DatasetProfiler, DatasetProfile
from intelligence.schema_detector import SchemaDetector, SchemaDetectionResult, DetectedSchema, SchemaDetectionError
from intelligence.frequency_detector import FrequencyDetector, FrequencyInfo
from intelligence.validator import DataValidator, ValidationReport, PipelineValidationError
from preprocessing.normalizer import SchemaNormalizer
from preprocessing.features import TimeSeriesFeatureEngineer, FeatureEngineeringConfig
from forecasting.evaluator import ModelEvaluator, EvaluationMetrics
from forecasting.model_selector import ModelSelector, ModelSelectionResult
from forecasting.future_forecaster import FutureForecaster, ForecastResult


@dataclass
class PipelineOutput:
    """Standardized output structure of the complete ML forecasting pipeline."""
    dataset_profile: Dict[str, Any]
    schema: Dict[str, Any]
    frequency: Dict[str, Any]
    validation: Dict[str, Any]
    features: List[str]
    model_results: Dict[str, Dict[str, Any]]
    selected_model: str
    test_metrics: Dict[str, Any]
    forecast: List[Dict[str, Any]]
    model_path: str
    metadata_path: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert output to standard dictionary."""
        return asdict(self)


class EnergyForecastingPipeline:
    """Production-grade orchestrator for energy consumption time-series forecasting."""

    def __init__(
        self,
        models_dir: str = "models",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        default_horizon: int = 24,
    ) -> None:
        """Initialize pipeline with directory paths and split ratios.

        Args:
            models_dir: Directory where trained model artifacts and metadata will be saved.
            train_ratio: Fraction of data for initial training (default 70%).
            val_ratio: Fraction of data for model validation (default 15%).
            test_ratio: Fraction of data for final out-of-sample testing (default 15%).
            default_horizon: Default number of future time steps to forecast.
        """
        self.models_dir = models_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.default_horizon = default_horizon

        os.makedirs(self.models_dir, exist_ok=True)

        self.profiler = DatasetProfiler()
        self.schema_detector = SchemaDetector()
        self.normalizer = SchemaNormalizer()
        self.validator = DataValidator()
        self.frequency_detector = FrequencyDetector()
        self.evaluator = ModelEvaluator()
        self.model_selector = ModelSelector(evaluator=self.evaluator)

    def _split_chronological(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Chronologically split featured dataset into Train, Validation, and Test sets.

        Strictly prevents lookahead leakage. Never uses random shuffling.

        Args:
            df: Featured DataFrame.
            feature_cols: List of predictor column names.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
        """
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
        df_test = df.iloc[val_end:]

        X_train = df_train[feature_cols].copy().reset_index(drop=True)
        y_train = df_train["consumption"].copy().reset_index(drop=True)

        X_val = df_val[feature_cols].copy().reset_index(drop=True)
        y_val = df_val["consumption"].copy().reset_index(drop=True)

        X_test = df_test[feature_cols].copy().reset_index(drop=True)
        y_test = df_test["consumption"].copy().reset_index(drop=True)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def run(
        self,
        data_source: Union[str, pd.DataFrame],
        horizon: Optional[int] = None,
        override_timestamp: Optional[str] = None,
        override_consumption: Optional[str] = None,
        future_weather: Optional[List[Dict[str, float]]] = None,
        model_tag: str = "energy_model",
    ) -> PipelineOutput:
        """Execute complete end-to-end ML training and forecasting workflow.

        Args:
            data_source: Filepath string to CSV or in-memory pandas DataFrame.
            horizon: Number of future steps to forecast (defaults to self.default_horizon).
            override_timestamp: Optional column name for timestamp if confirming ambiguity.
            override_consumption: Optional column name for target if confirming ambiguity.
            future_weather: Optional future weather forecasts for multi-step prediction.
            model_tag: Prefix name for saved artifact files.

        Returns:
            PipelineOutput containing structured diagnostics, model comparisons, test metrics, and future forecast.
        """
        forecast_horizon = horizon or self.default_horizon

        # 1. Load Data
        if isinstance(data_source, str):
            if not os.path.exists(data_source):
                raise FileNotFoundError(f"Input data file '{data_source}' does not exist.")
            raw_df = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            raw_df = data_source.copy()
        else:
            raise TypeError(f"Unsupported data source type: {type(data_source)}")

        # 2. Dataset Profiling
        profile: DatasetProfile = self.profiler.profile(raw_df)

        # 3. Schema Detection
        schema_res: SchemaDetectionResult = self.schema_detector.detect_schema(
            df=raw_df,
            override_timestamp=override_timestamp,
            override_consumption=override_consumption,
        )
        detected_schema = schema_res.detected_schema

        # 4. Schema Normalization
        normalized_df = self.normalizer.normalize(df=raw_df, schema=detected_schema)

        # 5. Data Validation
        val_report: ValidationReport = self.validator.validate(normalized_df)
        if not val_report.is_valid:
            raise PipelineValidationError(val_report.errors)

        # 6. Frequency Detection
        freq_info: FrequencyInfo = self.frequency_detector.detect_frequency(normalized_df["timestamp"])

        # 7. Time-Series Feature Engineering
        feature_engineer = TimeSeriesFeatureEngineer(freq_info=freq_info)
        featured_df, feat_config = feature_engineer.create_features(normalized_df, drop_warmup=True)

        if len(featured_df) < 30:
            raise PipelineValidationError([
                f"Remaining sample count after warmup ({len(featured_df)}) is too small for chronological splitting."
            ])

        # 8. Chronological Train / Validation / Test Split
        X_train, y_train, X_val, y_val, X_test, y_test = self._split_chronological(
            featured_df, feat_config.feature_columns
        )

        # 9 & 10. Train Candidates, Evaluate & Select Best Model
        seasonal_period = freq_info.periods_per_day if freq_info.periods_per_day > 1 else 7
        selection_res: ModelSelectionResult = self.model_selector.select_and_retrain(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            seasonal_period=seasonal_period,
        )

        best_model = selection_res.selected_model_instance

        # 11. Genuine Future Forecasting (Multi-step recursive dynamic lag engine)
        future_forecaster = FutureForecaster(
            model=best_model,
            feature_engineer=feature_engineer,
            freq_info=freq_info,
        )
        forecast_res: ForecastResult = future_forecaster.generate_forecast(
            history_df=normalized_df,
            horizon=forecast_horizon,
            future_weather=future_weather,
        )

        # 12. Model and Artifact Persistence
        model_filename = f"{model_tag}_{selection_res.selected_model_name.lower().replace(' ', '_')}.joblib"
        meta_filename = f"{model_tag}_metadata.json"
        
        model_path = os.path.join(self.models_dir, model_filename)
        metadata_path = os.path.join(self.models_dir, meta_filename)

        # Save model
        joblib.dump(best_model, model_path)

        # Save metadata and configuration
        metadata_payload = {
            "selected_model": selection_res.selected_model_name,
            "selection_criterion": selection_res.selection_criterion,
            "detected_schema": detected_schema.to_mapping(),
            "frequency_info": freq_info.to_dict(),
            "feature_config": {
                "feature_columns": feat_config.feature_columns,
                "lag_indices": feat_config.lag_indices,
                "rolling_windows": feat_config.rolling_windows,
                "weather_features": feat_config.weather_features,
            },
            "validation_metrics": {
                k: v.to_dict() for k, v in selection_res.validation_metrics.items()
            },
            "test_metrics": selection_res.test_metrics.to_dict(),
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)

        # Format output
        formatted_model_results = {
            k: v.to_dict() for k, v in selection_res.validation_metrics.items()
        }

        return PipelineOutput(
            dataset_profile=profile.to_dict(),
            schema=detected_schema.to_mapping(),
            frequency=freq_info.to_dict(),
            validation=val_report.to_dict(),
            features=feat_config.feature_columns,
            model_results=formatted_model_results,
            selected_model=selection_res.selected_model_name,
            test_metrics=selection_res.test_metrics.to_dict(),
            forecast=forecast_res.to_dict()["forecasts"],
            model_path=os.path.abspath(model_path),
            metadata_path=os.path.abspath(metadata_path),
        )
