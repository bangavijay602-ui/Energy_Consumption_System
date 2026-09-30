"""Unit tests for normalizer and feature engineer."""

import pytest
import pandas as pd
import numpy as np

from intelligence.frequency_detector import FrequencyDetector
from preprocessing.normalizer import SchemaNormalizer
from preprocessing.features import TimeSeriesFeatureEngineer


def test_schema_normalizer():
    normalizer = SchemaNormalizer()
    raw_df = pd.DataFrame({
        "DateTime": ["2026-01-01 02:00:00", "2026-01-01 01:00:00"], # Unsorted
        "Energy_Usage": [35.5, 30.2],
        "Temp": [18.0, 17.5],
    })
    mapping = {
        "timestamp": "DateTime",
        "consumption": "Energy_Usage",
        "temperature": "Temp",
    }
    norm_df = normalizer.normalize(raw_df, schema=mapping)

    assert list(norm_df.columns) == ["timestamp", "consumption", "temperature"]
    # Check sorted
    assert norm_df["timestamp"].iloc[0] < norm_df["timestamp"].iloc[1]
    assert norm_df["consumption"].iloc[0] == 30.2


def test_feature_engineering_no_leakage():
    ts = pd.date_range("2026-01-01", periods=200, freq="1h")
    consumption = np.linspace(10, 100, 200) # strictly increasing
    df = pd.DataFrame({"timestamp": ts, "consumption": consumption})

    freq_det = FrequencyDetector()
    freq_info = freq_det.detect_frequency(pd.Series(ts))

    fe = TimeSeriesFeatureEngineer(freq_info=freq_info)
    feat_df, config = fe.create_features(df, drop_warmup=True)

    # Check that lag_1 at index i equals consumption at index i-1
    assert "lag_1" in feat_df.columns
    # Check rolling mean short is strictly from prior values (shift 1)
    # The rolling mean of previous values must be strictly less than current consumption
    for i in range(len(feat_df)):
        current_cons = feat_df["consumption"].iloc[i]
        rolling_mean = feat_df[f"rolling_mean_{config.rolling_windows['short']}"].iloc[i]
        assert rolling_mean < current_cons, "Rolling feature leaked current consumption value!"


def test_single_step_feature_generation():
    ts = pd.date_range("2026-01-01", periods=100, freq="1h")
    consumption = np.random.uniform(20, 50, 100)
    df = pd.DataFrame({"timestamp": ts, "consumption": consumption})

    freq_det = FrequencyDetector()
    freq_info = freq_det.detect_frequency(pd.Series(ts))

    fe = TimeSeriesFeatureEngineer(freq_info=freq_info)
    _, config = fe.create_features(df, drop_warmup=True)

    next_ts = ts[-1] + pd.Timedelta(hours=1)
    single_feat = fe.generate_single_step_features(history_df=df, next_timestamp=next_ts)

    assert len(single_feat) == 1
    assert list(single_feat.columns) == config.feature_columns
    assert single_feat["lag_1"].iloc[0] == float(consumption[-1])
