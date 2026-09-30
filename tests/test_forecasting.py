"""Unit tests for models, evaluator, model selector, and future forecaster."""

import pytest
import pandas as pd
import numpy as np

from forecasting.baseline import BaselineForecaster
from forecasting.random_forest import RandomForestForecaster
from forecasting.xgboost_model import XGBoostForecaster
from forecasting.sarima import SARIMAForecaster
from forecasting.evaluator import ModelEvaluator
from forecasting.model_selector import ModelSelector
from forecasting.future_forecaster import FutureForecaster
from intelligence.frequency_detector import FrequencyDetector
from preprocessing.features import TimeSeriesFeatureEngineer


@pytest.fixture
def synthetic_split_data():
    dates = pd.date_range("2026-01-01", periods=300, freq="1h")
    hour = dates.hour.to_numpy()
    cons = 30.0 + 10.0 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 1.0, 300)
    df = pd.DataFrame({"timestamp": dates, "consumption": cons})

    freq_det = FrequencyDetector()
    freq_info = freq_det.detect_frequency(pd.Series(dates))

    fe = TimeSeriesFeatureEngineer(freq_info=freq_info)
    feat_df, config = fe.create_features(df, drop_warmup=True)

    n = len(feat_df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train = feat_df.iloc[:train_end][config.feature_columns]
    y_train = feat_df.iloc[:train_end]["consumption"]

    X_val = feat_df.iloc[train_end:val_end][config.feature_columns]
    y_val = feat_df.iloc[train_end:val_end]["consumption"]

    X_test = feat_df.iloc[val_end:][config.feature_columns]
    y_test = feat_df.iloc[val_end:]["consumption"]

    return df, X_train, y_train, X_val, y_val, X_test, y_test, fe, freq_info


def test_models_fit_predict(synthetic_split_data):
    df, X_train, y_train, X_val, y_val, X_test, y_test, fe, freq_info = synthetic_split_data

    # 1. Baseline
    b_model = BaselineForecaster(strategy="last_value", seasonal_period=24)
    b_model.fit(X_train, y_train)
    b_preds = b_model.predict(X_val)
    assert len(b_preds) == len(X_val)

    # 2. Random Forest
    rf_model = RandomForestForecaster(n_estimators=20, max_depth=6)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_val)
    assert len(rf_preds) == len(X_val)
    assert len(rf_model.get_feature_importances()) > 0

    # 3. XGBoost
    xgb_model = XGBoostForecaster(n_estimators=20, max_depth=4)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_val)
    assert len(xgb_preds) == len(X_val)

    # 4. SARIMA
    sarima_model = SARIMAForecaster(order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
    sarima_model.fit(X_train, y_train)
    sarima_preds = sarima_model.predict(X_val)
    assert len(sarima_preds) == len(X_val)


def test_evaluator_and_safe_mape():
    evaluator = ModelEvaluator()
    y_true = np.array([0.0, 10.0, 20.0, 30.0])
    y_pred = np.array([1.0, 11.0, 19.0, 32.0])

    metrics = evaluator.evaluate(y_true, y_pred, model_name="TestModel")
    assert not np.isnan(metrics.mape)
    assert not np.isinf(metrics.mape)
    assert metrics.mae > 0
    assert metrics.rmse > 0


def test_model_selector_and_future_forecaster(synthetic_split_data):
    df, X_train, y_train, X_val, y_val, X_test, y_test, fe, freq_info = synthetic_split_data

    selector = ModelSelector()
    res = selector.select_and_retrain(
        X_train, y_train, X_val, y_val, X_test, y_test, seasonal_period=24
    )

    assert res.selected_model_name in ["Naive Baseline", "Random Forest", "XGBoost", "SARIMA"]
    assert res.test_metrics.mae > 0

    forecaster = FutureForecaster(model=res.selected_model_instance, feature_engineer=fe, freq_info=freq_info)
    forecast_res = forecaster.generate_forecast(history_df=df, horizon=12)

    assert len(forecast_res.forecasts) == 12
    assert len(forecast_res.forecast_df) == 12
    for pt in forecast_res.forecasts:
        assert pt.predicted_consumption >= 0
