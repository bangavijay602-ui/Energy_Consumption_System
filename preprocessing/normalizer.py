"""Schema normalization module to produce clean, canonical time-series DataFrames."""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

from intelligence.schema_detector import DetectedSchema


class SchemaNormalizer:
    """Standardizes heterogeneous raw DataFrames into canonical schema and data types."""

    CANONICAL_COLUMNS = [
        "timestamp",
        "consumption",
        "temperature",
        "humidity",
        "wind_speed",
        "pressure",
        "holiday",
    ]

    def normalize(
        self,
        df: pd.DataFrame,
        schema: Union[DetectedSchema, Dict[str, str]],
    ) -> pd.DataFrame:
        """Transform raw input DataFrame into canonical schema.

        Args:
            df: Raw DataFrame loaded from user CSV.
            schema: DetectedSchema object or mapping of {canonical_name: raw_column_name}.

        Returns:
            Normalized DataFrame indexed by integer with standard column names and types.
        """
        mapping = schema.to_mapping() if isinstance(schema, DetectedSchema) else schema
        
        # Invert mapping to {raw_name: canonical_name}
        rename_dict = {raw_col: canon_col for canon_col, raw_col in mapping.items() if raw_col in df.columns}
        
        # Subset and rename
        subset_df = df[list(rename_dict.keys())].copy()
        norm_df = subset_df.rename(columns=rename_dict)

        if "timestamp" not in norm_df.columns:
            raise KeyError("Canonical column 'timestamp' is missing in mapped schema.")
        if "consumption" not in norm_df.columns:
            raise KeyError("Canonical column 'consumption' is missing in mapped schema.")

        # 1. Standardize Timestamp
        norm_df["timestamp"] = pd.to_datetime(norm_df["timestamp"], errors="coerce")
        # Drop rows where timestamp could not be parsed
        norm_df = norm_df.dropna(subset=["timestamp"])

        # 2. Sort chronologically
        norm_df = norm_df.sort_values("timestamp").reset_index(drop=True)

        # 3. Handle Duplicate Timestamps by taking mean of numeric features
        if norm_df["timestamp"].duplicated().any():
            numeric_cols = [c for c in norm_df.columns if c != "timestamp"]
            norm_df = norm_df.groupby("timestamp", as_index=False)[numeric_cols].mean()

        # 4. Standardize Numeric Types and Impute Small Gaps
        for col in norm_df.columns:
            if col == "timestamp":
                continue
            norm_df[col] = pd.to_numeric(norm_df[col], errors="coerce").astype(np.float64)
            # Impute missing values with forward-fill then backward-fill
            if norm_df[col].isna().any():
                norm_df[col] = norm_df[col].ffill().bfill()

        # Final reset index
        norm_df = norm_df.reset_index(drop=True)
        return norm_df
