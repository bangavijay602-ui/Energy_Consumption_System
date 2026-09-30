"""Integration tests for the complete end-to-end energy forecasting pipeline."""

import os
import pytest
import pandas as pd

from pipeline import EnergyForecastingPipeline, PipelineOutput
from intelligence.schema_detector import SchemaDetectionError
from intelligence.validator import PipelineValidationError


@pytest.fixture
def sample_data_dir():
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    return data_dir


def test_pipeline_dataset1_hourly(sample_data_dir, tmp_path):
    fpath = os.path.join(sample_data_dir, "dataset1.csv")
    pipeline = EnergyForecastingPipeline(models_dir=str(tmp_path), default_horizon=24)
    output = pipeline.run(data_source=fpath, horizon=24, model_tag="test_ds1")

    assert output.schema["timestamp"] == "DateTime"
    assert output.schema["consumption"] == "Energy_Usage"
    assert output.frequency["name"] == "hourly"
    assert output.selected_model in output.model_results
    assert len(output.forecast) == 24
    assert os.path.exists(output.model_path)
    assert os.path.exists(output.metadata_path)


def test_pipeline_dataset2_30min(sample_data_dir, tmp_path):
    fpath = os.path.join(sample_data_dir, "dataset2.csv")
    pipeline = EnergyForecastingPipeline(models_dir=str(tmp_path), default_horizon=12)
    output = pipeline.run(data_source=fpath, horizon=12, model_tag="test_ds2")

    assert output.schema["timestamp"] == "timestamp"
    assert output.schema["consumption"] == "load"
    assert output.frequency["name"] == "30-minute"
    assert len(output.forecast) == 12


def test_pipeline_dataset3_weather_15min(sample_data_dir, tmp_path):
    fpath = os.path.join(sample_data_dir, "dataset3.csv")
    pipeline = EnergyForecastingPipeline(models_dir=str(tmp_path), default_horizon=16)
    output = pipeline.run(data_source=fpath, horizon=16, model_tag="test_ds3")

    assert output.schema["timestamp"] == "reading_time"
    assert output.schema["consumption"] == "power_consumption"
    assert output.schema.get("temperature") == "temperature"
    assert output.schema.get("humidity") == "humidity"
    assert output.frequency["name"] == "15-minute"
    assert len(output.forecast) == 16


def test_pipeline_dataset4_daily(sample_data_dir, tmp_path):
    fpath = os.path.join(sample_data_dir, "dataset4.csv")
    pipeline = EnergyForecastingPipeline(models_dir=str(tmp_path), default_horizon=7)
    output = pipeline.run(data_source=fpath, horizon=7, model_tag="test_ds4")

    assert output.schema["timestamp"] == "date"
    assert output.schema["consumption"] == "electricity_demand"
    assert output.frequency["name"] == "daily"
    assert len(output.forecast) == 7


def test_pipeline_dataset5_hourly_temp(sample_data_dir, tmp_path):
    fpath = os.path.join(sample_data_dir, "dataset5.csv")
    pipeline = EnergyForecastingPipeline(models_dir=str(tmp_path), default_horizon=24)
    output = pipeline.run(data_source=fpath, horizon=24, model_tag="test_ds5")

    assert output.schema["timestamp"] == "time"
    assert output.schema["consumption"] == "value"
    assert output.schema.get("temperature") == "temp"
    assert len(output.forecast) == 24


def test_pipeline_invalid_rejection(sample_data_dir, tmp_path):
    fpath = os.path.join(sample_data_dir, "invalid.csv")
    pipeline = EnergyForecastingPipeline(models_dir=str(tmp_path), default_horizon=24)
    with pytest.raises((SchemaDetectionError, PipelineValidationError)):
        pipeline.run(data_source=fpath)
