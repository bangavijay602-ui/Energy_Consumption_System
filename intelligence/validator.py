"""Validation module for standardized time-series datasets."""

from dataclasses import dataclass, field
from typing import Any, Dict, List
import pandas as pd
import numpy as np


class PipelineValidationError(Exception):
    """Raised when dataset fails critical validation requirements."""
    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__("Dataset validation failed with errors:\n - " + "\n - ".join(errors))


@dataclass
class ValidationReport:
    """Comprehensive validation diagnostics and warnings."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "diagnostics": self.diagnostics,
        }


class DataValidator:
    """Validates normalized canonical datasets for production ML readiness."""

    def __init__(self, min_observations: int = 50, max_missing_ratio: float = 0.20) -> None:
        """Initialize validator.
        
        Args:
            min_observations: Minimum acceptable number of rows.
            max_missing_ratio: Maximum fraction of missing target values before failing.
        """
        self.min_observations = min_observations
        self.max_missing_ratio = max_missing_ratio

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Perform rigorous validation on normalized DataFrame.

        Args:
            df: Normalized DataFrame with canonical columns 'timestamp' and 'consumption'.

        Returns:
            ValidationReport with boolean validity status, errors, warnings, and telemetry.
        """
        errors: List[str] = []
        warnings: List[str] = []
        diag: Dict[str, Any] = {}

        # 1. Column existence
        if "timestamp" not in df.columns:
            errors.append("Required column 'timestamp' is missing.")
        if "consumption" not in df.columns:
            errors.append("Required column 'consumption' is missing.")

        if errors:
            return ValidationReport(is_valid=False, errors=errors, warnings=warnings, diagnostics=diag)

        n_rows = len(df)
        diag["num_rows"] = n_rows

        # 2. Minimum observation threshold
        if n_rows < self.min_observations:
            errors.append(
                f"Insufficient observations: dataset has {n_rows} rows, but at least {self.min_observations} "
                "are required for meaningful train/val/test splits and lag feature creation."
            )

        # 3. Timestamp validation
        ts = df["timestamp"]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            try:
                parsed_ts = pd.to_datetime(ts, errors="coerce")
                null_ts = int(parsed_ts.isna().sum())
                if null_ts > 0:
                    errors.append(f"Timestamp column contains {null_ts} unparseable datetime values.")
            except Exception as e:
                errors.append(f"Failed to parse timestamp column: {str(e)}")
        else:
            parsed_ts = ts
            null_ts = int(parsed_ts.isna().sum())
            if null_ts > 0:
                errors.append(f"Timestamp column contains {null_ts} missing (NaT) values.")

        # Check timestamp ordering
        if not parsed_ts.is_monotonic_increasing:
            warnings.append("Timestamps are not in strict chronological order. Pipeline will sort them.")

        # Check duplicate timestamps
        dup_ts = int(parsed_ts.duplicated().sum())
        diag["duplicate_timestamps"] = dup_ts
        if dup_ts > 0:
            warnings.append(f"Detected {dup_ts} duplicate timestamps. They will be aggregated during normalization.")

        # 4. Target (consumption) validation
        target = df["consumption"]
        if not pd.api.types.is_numeric_dtype(target):
            try:
                num_target = pd.to_numeric(target, errors="coerce")
                non_numeric = int(num_target.isna().sum() - target.isna().sum())
                if non_numeric > 0:
                    errors.append(f"Target column 'consumption' contains {non_numeric} non-numeric string values.")
            except Exception as e:
                errors.append(f"Target column 'consumption' is not numeric: {str(e)}")
                num_target = pd.Series([np.nan] * n_rows)
        else:
            num_target = target

        # Missing values
        missing_target = int(num_target.isna().sum())
        missing_ratio = (missing_target / n_rows) if n_rows > 0 else 1.0
        diag["missing_target_count"] = missing_target
        diag["missing_target_ratio"] = round(missing_ratio, 4)

        if missing_ratio > self.max_missing_ratio:
            errors.append(
                f"High missing target values: {missing_target} rows ({missing_ratio*100:.1f}%) are NaN. "
                f"Maximum allowed is {self.max_missing_ratio*100:.1f}%."
            )
        elif missing_target > 0:
            warnings.append(
                f"Target contains {missing_target} missing values ({missing_ratio*100:.1f}%). "
                "Forward-fill interpolation will be applied."
            )

        # Variance / Constant Series Check
        valid_targets = num_target.dropna()
        if len(valid_targets) > 0:
            target_min = float(valid_targets.min())
            target_max = float(valid_targets.max())
            target_std = float(valid_targets.std()) if len(valid_targets) > 1 else 0.0
            diag["target_min"] = target_min
            diag["target_max"] = target_max
            diag["target_std"] = round(target_std, 4)

            if target_std < 1e-6 or target_min == target_max:
                errors.append(
                    "Target series is constant (zero variance). Forecasting models cannot learn from constant data."
                )

            # Negative values check
            negative_count = int((valid_targets < 0).sum())
            diag["negative_target_count"] = negative_count
            if negative_count > 0:
                warnings.append(
                    f"Target column contains {negative_count} negative consumption values. "
                    "In physical energy systems, negative net consumption typically represents local solar/wind generation or sensor offset."
                )

            # Peak / Outlier Detection (Diagnostics Only - never silently dropped!)
            q25 = float(valid_targets.quantile(0.25))
            q75 = float(valid_targets.quantile(0.75))
            iqr = q75 - q25
            spike_threshold = q75 + (3.0 * iqr)
            spikes = valid_targets[valid_targets > spike_threshold]
            diag["detected_energy_spikes"] = len(spikes)
            if len(spikes) > 0:
                warnings.append(
                    f"Detected {len(spikes)} high energy demand peaks (> {spike_threshold:.2f}). "
                    "These represent valid extreme load events and will be preserved for training."
                )

        is_valid = len(errors) == 0
        return ValidationReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            diagnostics=diag,
        )
