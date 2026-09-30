"""Tests for FastAPI endpoints."""

import io
import os
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_api_forecast_endpoint():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset1.csv")
    with open(data_path, "rb") as f:
        csv_bytes = f.read()

    response = client.post(
        "/api/forecast",
        files={"file": ("dataset1.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"horizon": 12},
    )

    assert response.status_code == 200
    data = response.json()
    assert "dataset_profile" in data
    assert "schema" in data
    assert "model_results" in data
    assert "selected_model" in data
    assert len(data["forecast"]) == 12


def test_api_invalid_file_rejection():
    invalid_path = os.path.join(os.path.dirname(__file__), "..", "data", "invalid.csv")
    with open(invalid_path, "rb") as f:
        csv_bytes = f.read()

    response = client.post(
        "/api/forecast",
        files={"file": ("invalid.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"horizon": 24},
    )

    assert response.status_code == 422
