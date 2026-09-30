"""Dataset profiling module for inspecting raw uploaded time-series CSVs."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


@dataclass
class ColumnProfile:
    """Statistical profile of a single column."""
    name: str
    dtype: str
    missing_count: int
    missing_percentage: float
    unique_count: int
    is_numeric: bool
    sample_values: List[Any] = field(default_factory=list)
    stats: Optional[Dict[str, float]] = None


@dataclass
class DatasetProfile:
    """Comprehensive profile of an uploaded dataset."""
    num_rows: int
    num_columns: int
    column_names: List[str]
    duplicate_rows: int
    duplicate_row_percentage: float
    total_missing_cells: int
    missing_cell_percentage: float
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_candidate_columns: List[str]
    column_profiles: Dict[str, ColumnProfile]
    memory_usage_mb: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataset profile to a standard Python dictionary."""
        result = asdict(self)
        return result


class DatasetProfiler:
    """Inspects and profiles raw pandas DataFrames before ingestion."""

    def __init__(self, sample_size: int = 5) -> None:
        """Initialize profiler.
        
        Args:
            sample_size: Number of example values to store per column.
        """
        self.sample_size = sample_size

    def profile(self, df: pd.DataFrame) -> DatasetProfile:
        """Profile a DataFrame and return structured telemetry.

        Args:
            df: Raw DataFrame loaded from user CSV.

        Returns:
            DatasetProfile containing complete dataset overview.
        """
        if df.empty:
            raise ValueError("Cannot profile an empty DataFrame.")

        num_rows, num_cols = df.shape
        col_names = list(df.columns)
        dup_count = int(df.duplicated().sum())
        dup_pct = round((dup_count / num_rows) * 100.0, 2) if num_rows > 0 else 0.0

        total_cells = num_rows * num_cols
        total_missing = int(df.isna().sum().sum())
        missing_pct = round((total_missing / total_cells) * 100.0, 2) if total_cells > 0 else 0.0

        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        datetime_candidates: List[str] = []
        col_profiles: Dict[str, ColumnProfile] = {}

        mem_bytes = df.memory_usage(deep=True).sum()
        mem_mb = round(mem_bytes / (1024 * 1024), 3)

        for col in df.columns:
            series = df[col]
            missing_c = int(series.isna().sum())
            missing_p = round((missing_c / num_rows) * 100.0, 2) if num_rows > 0 else 0.0
            unique_c = int(series.nunique(dropna=True))
            is_num = bool(pd.api.types.is_numeric_dtype(series))

            # Store non-null samples safely
            valid_series = series.dropna()
            samples = [
                str(v) if not isinstance(v, (int, float, bool)) or np.isnan(v) else v
                for v in valid_series.head(self.sample_size).tolist()
            ]

            stats: Optional[Dict[str, float]] = None
            if is_num and len(valid_series) > 0:
                numeric_cols.append(str(col))
                try:
                    stats = {
                        "min": float(valid_series.min()),
                        "max": float(valid_series.max()),
                        "mean": round(float(valid_series.mean()), 4),
                        "std": round(float(valid_series.std()), 4) if len(valid_series) > 1 else 0.0,
                        "median": round(float(valid_series.median()), 4),
                        "skew": round(float(valid_series.skew()), 4) if len(valid_series) > 2 else 0.0,
                    }
                except Exception:
                    stats = None
            else:
                categorical_cols.append(str(col))
                # Check quick sample datetime parseability
                sample_text = valid_series.astype(str).head(15)
                if len(sample_text) > 0:
                    try:
                        parsed = pd.to_datetime(sample_text, errors="coerce")
                        if parsed.notna().mean() >= 0.7:
                            datetime_candidates.append(str(col))
                    except Exception:
                        pass

            col_profiles[str(col)] = ColumnProfile(
                name=str(col),
                dtype=str(series.dtype),
                missing_count=missing_c,
                missing_percentage=missing_p,
                unique_count=unique_c,
                is_numeric=is_num,
                sample_values=samples,
                stats=stats,
            )

        return DatasetProfile(
            num_rows=num_rows,
            num_columns=num_cols,
            column_names=[str(c) for c in col_names],
            duplicate_rows=dup_count,
            duplicate_row_percentage=dup_pct,
            total_missing_cells=total_missing,
            missing_cell_percentage=missing_pct,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_candidate_columns=datetime_candidates,
            column_profiles=col_profiles,
            memory_usage_mb=mem_mb,
        )
