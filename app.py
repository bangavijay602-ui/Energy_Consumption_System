"""FastAPI REST server for the Energy Consumption Forecasting Pipeline."""

import io
import os
from typing import Optional
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pipeline import EnergyForecastingPipeline
from intelligence.schema_detector import SchemaDetectionError, AmbiguousSchemaError
from intelligence.validator import PipelineValidationError

app = FastAPI(
    title="Energy Consumption Forecasting API",
    description="Production-grade time-series machine learning training and inference API.",
    version="1.0.0",
)

# Enable CORS for React/Vue/Angular frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = EnergyForecastingPipeline(models_dir="models", default_horizon=24)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "energy-forecasting-pipeline"}


@app.post("/api/forecast")
async def train_and_forecast(
    file: UploadFile = File(..., description="CSV file containing energy time series"),
    horizon: int = Form(24, description="Number of future time steps to forecast"),
    override_timestamp: Optional[str] = Form(None, description="Optional timestamp column name"),
    override_consumption: Optional[str] = Form(None, description="Optional target column name"),
):
    """Ingest user CSV, profile, normalize, validate, train candidate models, select winner, and forecast."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .csv file.")

    try:
        content = await file.read()
        raw_df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {str(e)}")

    model_tag = os.path.splitext(file.filename)[0]

    try:
        output = pipeline.run(
            data_source=raw_df,
            horizon=horizon,
            override_timestamp=override_timestamp,
            override_consumption=override_consumption,
            model_tag=model_tag,
        )
        return JSONResponse(status_code=200, content=output.to_dict())

    except AmbiguousSchemaError as e:
        return JSONResponse(
            status_code=422,
            content={
                "error_type": "AmbiguousSchemaError",
                "message": str(e),
                "role": e.role,
                "candidates": [{"column": c, "confidence": s} for c, s in e.candidates],
            },
        )
    except (SchemaDetectionError, PipelineValidationError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal pipeline error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
