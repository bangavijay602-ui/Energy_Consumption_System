"""CLI and demonstration runner for the Energy Consumption Forecasting Pipeline."""

import argparse
import json
import os
import sys
from typing import List
import pandas as pd

from pipeline import EnergyForecastingPipeline, PipelineOutput
from intelligence.schema_detector import SchemaDetectionError
from intelligence.validator import PipelineValidationError


def print_banner(title: str) -> None:
    """Print formatted section banner."""
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def run_pipeline_on_file(
    file_path: str,
    horizon: int = 24,
    models_dir: str = "models",
) -> Optional[PipelineOutput]:
    """Execute pipeline on a specific dataset file and display results."""
    print_banner(f"Processing Dataset: {os.path.basename(file_path)}")
    pipeline = EnergyForecastingPipeline(models_dir=models_dir, default_horizon=horizon)

    try:
        output: PipelineOutput = pipeline.run(
            data_source=file_path,
            horizon=horizon,
            model_tag=os.path.splitext(os.path.basename(file_path))[0],
        )

        # 1. Profiling & Intelligence Summary
        print("\n--- 1. DATASET INTELLIGENCE & SCHEMA ---")
        p = output.dataset_profile
        print(f"Rows: {p['num_rows']}, Columns: {p['num_columns']}")
        print(f"Column Names: {p['column_names']}")
        print(f"Missing Cells: {p['total_missing_cells']} ({p['missing_cell_percentage']}%)")
        print(f"Duplicate Rows: {p['duplicate_rows']}")

        print(f"\nDetected Schema Mapping:")
        for canon, raw in output.schema.items():
            print(f"  {canon.ljust(15)} <- '{raw}'")

        print(f"\nDetected Frequency:")
        freq = output.frequency
        print(f"  Cadence: {freq['name']} (pandas alias: {freq['pandas_alias']})")
        print(f"  Periods/Day: {freq['periods_per_day']}, Periods/Week: {freq['periods_per_week']}")
        print(f"  Regularity: {'Regular' if freq['is_regular'] else 'Irregular'}")

        # 2. Validation
        print("\n--- 2. VALIDATION REPORT ---")
        val = output.validation
        print(f"Valid: {val['is_valid']}")
        if val["warnings"]:
            print("Warnings:")
            for w in val["warnings"]:
                print(f"  [WARN] {w}")

        # 3. Model Evaluation Table
        print("\n--- 3. CANDIDATE MODEL VALIDATION LEADERBOARD ---")
        table_rows = []
        for model_name, m in output.model_results.items():
            table_rows.append({
                "Model": model_name,
                "MAE": m["mae"],
                "RMSE": m["rmse"],
                "MAPE (%)": f"{m['mape']:.2f}%",
                "WAPE (%)": f"{m['wape']:.2f}%",
                "R2 Score": m["r2"],
            })
        df_comp = pd.DataFrame(table_rows).sort_values("MAE").reset_index(drop=True)
        print(df_comp.to_string(index=False))

        # 4. Selection & Test Evaluation
        print("\n--- 4. MODEL SELECTION & TEST SET EVALUATION ---")
        print(f"Selected Winner: >>> {output.selected_model} <<< (Lowest Validation MAE)")
        tm = output.test_metrics
        print(f"Held-Out Test Performance:")
        print(f"  MAE:  {tm['mae']:.4f}")
        print(f"  RMSE: {tm['rmse']:.4f}")
        print(f"  MAPE: {tm['mape']:.2f}%")
        print(f"  WAPE: {tm['wape']:.2f}%")
        print(f"  R2:   {tm['r2']:.4f}")

        # 5. Future Forecast
        print(f"\n--- 5. RECURSIVE MULTI-STEP FUTURE FORECAST (Horizon={horizon}) ---")
        forecast_df = pd.DataFrame(output.forecast)
        print(forecast_df.head(10).to_string(index=False))
        if len(forecast_df) > 10:
            print(f"... [{len(forecast_df) - 10} more steps] ...")
            print(forecast_df.tail(3).to_string(index=False))

        # 6. Persistence
        print(f"\n--- 6. ARTIFACT PERSISTENCE ---")
        print(f"Model Artifact:    {output.model_path}")
        print(f"Metadata Artifact: {output.metadata_path}")

        return output

    except (SchemaDetectionError, PipelineValidationError) as e:
        print(f"\n[PIPELINE REJECTED DATASET] Expected error caught:")
        print(f"  Error: {str(e)}")
        return None
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR]: {str(e)}")
        raise e


def run_demo(data_dir: str = "data") -> None:
    """Run pipeline against all 5 valid test formats and 1 invalid format."""
    print_banner("ENERGY CONSUMPTION FORECASTING PIPELINE DEMO")
    print("Testing 5 diverse dataset schemas + 1 invalid non-time-series CSV.\n")

    test_files = [
        "dataset1.csv",  # DateTime, Energy_Usage (Hourly)
        "dataset2.csv",  # timestamp, load (30-min)
        "dataset3.csv",  # reading_time, power_consumption, temperature, humidity (15-min)
        "dataset4.csv",  # date, electricity_demand (Daily)
        "dataset5.csv",  # time, value, temp (Hourly)
        "invalid.csv",   # Name, Age, Marks, City (Should reject gracefully)
    ]

    for fname in test_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            run_pipeline_on_file(file_path=fpath, horizon=24)
        else:
            print(f"Warning: {fpath} not found.")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Energy Consumption Forecasting ML Pipeline"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV file containing energy time-series data.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Number of future time steps to forecast (default: 24).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save models and metadata (default: models).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run comprehensive test suite across all 5 test datasets + 1 invalid dataset.",
    )

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.data:
        run_pipeline_on_file(file_path=args.data, horizon=args.horizon, models_dir=args.output_dir)
    else:
        print("No input dataset provided. Running demo mode across all sample datasets.\n")
        run_demo()


if __name__ == "__main__":
    main()
