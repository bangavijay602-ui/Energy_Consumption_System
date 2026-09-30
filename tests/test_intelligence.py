"""Unit tests for dataset intelligence components."""

import pytest
import pandas as pd
import numpy as np

from intelligence.profiler import DatasetProfiler
from intelligence.schema_detector import SchemaDetector, SchemaDetectionError, AmbiguousSchemaError
from intelligence.frequency_detector import FrequencyDetector
from intelligence.validator import DataValidator


def test_dataset_profiler():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=100, freq="1h"),
        "load": np.random.uniform(10, 50, 100),
        "category": ["A"] * 50 + ["B"] * 50,
    })
    profiler = DatasetProfiler()
    profile = profiler.profile(df)

    assert profile.num_rows == 100
    assert profile.num_columns == 3
    assert "load" in profile.numeric_columns
    assert "category" in profile.categorical_columns
    assert profile.total_missing_cells == 0


def test_schema_detector_variations():
    detector = SchemaDetector()

    # Variation 1: DateTime, Energy_Usage
    df1 = pd.DataFrame({
        "DateTime": pd.date_range("2026-01-01", periods=50, freq="1h"),
        "Energy_Usage": np.random.uniform(10, 50, 50),
    })
    res1 = detector.detect_schema(df1)
    assert res1.detected_schema.timestamp == "DateTime"
    assert res1.detected_schema.consumption == "Energy_Usage"

    # Variation 2: reading_time, power_consumption, temperature, humidity
    df2 = pd.DataFrame({
        "reading_time": pd.date_range("2026-01-01", periods=50, freq="15min"),
        "power_consumption": np.random.uniform(10, 50, 50),
        "temperature": np.random.uniform(15, 25, 50),
        "humidity": np.random.uniform(40, 70, 50),
    })
    res2 = detector.detect_schema(df2)
    assert res2.detected_schema.timestamp == "reading_time"
    assert res2.detected_schema.consumption == "power_consumption"
    assert res2.detected_schema.temperature == "temperature"
    assert res2.detected_schema.humidity == "humidity"


def test_schema_detector_invalid():
    detector = SchemaDetector()
    invalid_df = pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35],
        "Marks": [80, 90, 85],
    })
    with pytest.raises(SchemaDetectionError):
        detector.detect_schema(invalid_df)


def test_frequency_detector():
    detector = FrequencyDetector()

    # 15-min
    ts_15m = pd.date_range("2026-01-01", periods=200, freq="15min")
    f_15m = detector.detect_frequency(pd.Series(ts_15m))
    assert f_15m.name == "15-minute"
    assert f_15m.periods_per_day == 96
    assert f_15m.periods_per_week == 672

    # Hourly
    ts_1h = pd.date_range("2026-01-01", periods=200, freq="1h")
    f_1h = detector.detect_frequency(pd.Series(ts_1h))
    assert f_1h.name == "hourly"
    assert f_1h.periods_per_day == 24
    assert f_1h.periods_per_week == 168

    # Daily
    ts_1d = pd.date_range("2026-01-01", periods=100, freq="1D")
    f_1d = detector.detect_frequency(pd.Series(ts_1d))
    assert f_1d.name == "daily"
    assert f_1d.periods_per_day == 1
    assert f_1d.periods_per_week == 7


def test_validator():
    validator = DataValidator(min_observations=50)

    # Valid DataFrame
    valid_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=100, freq="1h"),
        "consumption": np.random.uniform(20, 100, 100),
    })
    rep = validator.validate(valid_df)
    assert rep.is_valid
    assert len(rep.errors) == 0

    # Constant target (Zero variance)
    const_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=100, freq="1h"),
        "consumption": [50.0] * 100,
    })
    rep_const = validator.validate(const_df)
    assert not rep_const.is_valid
    assert any("constant" in err.lower() for err in rep_const.errors)

    # Insufficient rows
    short_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=20, freq="1h"),
        "consumption": np.random.uniform(20, 100, 20),
    })
    rep_short = validator.validate(short_df)
    assert not rep_short.is_valid
    assert any("insufficient" in err.lower() for err in rep_short.errors)
