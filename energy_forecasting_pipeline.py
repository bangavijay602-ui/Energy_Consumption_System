"""Single-File Production-Quality Energy Consumption Forecasting Pipeline.

This standalone module contains the complete end-to-end time-series machine learning pipeline:
1. Dataset Profiling & Intelligence (Profiler, Schema Detector, Frequency Detector, Validator)
2. Preprocessing & Normalization (Schema Normalizer, Frequency-aware Feature Engineer with zero lookahead leakage)
3. Forecasting Models (Naive Baseline, Random Forest, XGBoost, SARIMA)
4. Evaluation & Model Selection (MAE, RMSE, safe-MAPE, history retraining, out-of-sample test scoring)
5. Genuine Recursive Multi-step Future Forecaster (Dynamic step-by-step feature reconstruction)
6. Model & Metadata Persistence (joblib & json)
7. CLI & Automated Demo Runner with Synthetic Dataset Generator
"""

import argparse
import io
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

# Optional imports with graceful fallbacks
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# =====================================================================
# 1. EXCEPTIONS & DATA STRUCTURES
# =====================================================================

class SchemaDetectionError(Exception):
    """Raised when schema detection fails or is invalid."""
    pass


class AmbiguousSchemaError(SchemaDetectionError):
    """Raised when multiple columns match a required role with similar confidence."""
    def __init__(self, role: str, candidates: List[Tuple[str, float]]) -> None:
        self.role = role
        self.candidates = candidates
        cand_str = ", ".join([f"'{c}' (score: {s:.2f})" for c, s in candidates])
        super().__init__(
            f"Ambiguous candidates detected for required role '{role}': {cand_str}. "
            "Please confirm or explicitly specify the column mapping."
        )


class PipelineValidationError(Exception):
    """Raised when dataset fails critical validation requirements."""
    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__("Dataset validation failed with errors:\n - " + "\n - ".join(errors))


@dataclass
class ColumnCandidate:
    column_name: str
    confidence: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class DetectedSchema:
    timestamp: str
    consumption: str
    temperature: Optional[str] = None
    humidity: Optional[str] = None
    wind_speed: Optional[str] = None
    pressure: Optional[str] = None
    holiday: Optional[str] = None

    def to_mapping(self) -> Dict[str, str]:
        mapping = {"timestamp": self.timestamp, "consumption": self.consumption}
        if self.temperature: mapping["temperature"] = self.temperature
        if self.humidity: mapping["humidity"] = self.humidity
        if self.wind_speed: mapping["wind_speed"] = self.wind_speed
        if self.pressure: mapping["pressure"] = self.pressure
        if self.holiday: mapping["holiday"] = self.holiday
        return mapping


@dataclass
class DatasetProfile:
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
    column_profiles: Dict[str, Any]
    memory_usage_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrequencyInfo:
    name: str                       # '15-minute', '30-minute', 'hourly', 'daily', 'weekly', 'irregular'
    pandas_alias: str               # '15min', '30min', '1h', '1D', '1W'
    median_delta_seconds: float
    periods_per_day: int
    periods_per_week: int
    is_regular: bool
    irregular_gap_percentage: float
    recommended_lag_indices: Dict[str, int]
    resampling_recommended: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureEngineeringConfig:
    frequency_name: str
    lag_indices: Dict[str, int]
    rolling_windows: Dict[str, int]
    calendar_features: List[str]
    cyclic_features: List[str]
    weather_features: List[str]
    feature_columns: List[str] = field(default_factory=list)
    warmup_rows: int = 0


@dataclass
class EvaluationMetrics:
    model_name: str
    mae: float
    rmse: float
    mape: float
    wape: float
    r2: float
    sample_count: int

    def to_dict(self) -> Dict[str, Union[str, float, int]]:
        return asdict(self)


@dataclass
class ForecastPoint:
    timestamp: str
    predicted_consumption: float
    step_ahead: int


@dataclass
class ForecastResult:
    model_name: str
    horizon: int
    start_timestamp: str
    end_timestamp: str
    forecasts: List[ForecastPoint]
    forecast_df: pd.DataFrame = field(repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "forecasts": [
                {
                    "timestamp": p.timestamp,
                    "predicted_consumption": round(p.predicted_consumption, 4),
                    "step_ahead": p.step_ahead,
                }
                for p in self.forecasts
            ],
        }


@dataclass
class PipelineOutput:
    dataset_profile: Dict[str, Any]
    schema: Dict[str, Any]
    frequency: Dict[str, Any]
    validation: Dict[str, Any]
    features: List[str]
    model_results: Dict[str, Dict[str, Any]]
    selected_model: str
    test_metrics: Dict[str, Any]
    forecast: List[Dict[str, Any]]
    model_path: str
    metadata_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# 2. DATASET INTELLIGENCE LAYER
# =====================================================================

class DatasetProfiler:
    """Profiles raw datasets for row/column counts, types, nulls, duplicates, and stats."""

    def __init__(self, sample_size: int = 5) -> None:
        self.sample_size = sample_size

    def profile(self, df: pd.DataFrame) -> DatasetProfile:
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
        col_profiles: Dict[str, Any] = {}

        mem_bytes = df.memory_usage(deep=True).sum()
        mem_mb = round(mem_bytes / (1024 * 1024), 3)

        for col in df.columns:
            series = df[col]
            missing_c = int(series.isna().sum())
            missing_p = round((missing_c / num_rows) * 100.0, 2) if num_rows > 0 else 0.0
            unique_c = int(series.nunique(dropna=True))
            is_num = bool(pd.api.types.is_numeric_dtype(series))

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
                    }
                except Exception:
                    stats = None
            else:
                categorical_cols.append(str(col))
                sample_text = valid_series.astype(str).head(15)
                if len(sample_text) > 0:
                    try:
                        parsed = pd.to_datetime(sample_text, errors="coerce")
                        if parsed.notna().mean() >= 0.7:
                            datetime_candidates.append(str(col))
                    except Exception:
                        pass

            col_profiles[str(col)] = {
                "name": str(col),
                "dtype": str(series.dtype),
                "missing_count": missing_c,
                "missing_percentage": missing_p,
                "unique_count": unique_c,
                "is_numeric": is_num,
                "sample_values": samples,
                "stats": stats,
            }

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


class SchemaDetector:
    """Detects timestamp, energy target, and environmental variables with ambiguity guards."""

    TIMESTAMP_PATTERNS = [
        (r"^timestamp$", 1.0), (r"^datetime$", 1.0), (r"^date_time$", 1.0),
        (r"^reading_time$", 0.95), (r"^recorded_at$", 0.95), (r"^measurement_time$", 0.95),
        (r"^record_time$", 0.95), (r"^time_stamp$", 0.95), (r"^date$", 0.85), (r"^time$", 0.85),
        (r".*time.*", 0.65), (r".*date.*", 0.65), (r".*period.*", 0.50), (r"^utc_timestamp$", 1.0),
    ]

    TARGET_PATTERNS = [
        (r"^consumption$", 1.0), (r"^energy_usage$", 1.0), (r"^power_consumption$", 1.0),
        (r"^electricity_demand$", 1.0), (r"^electricity_usage$", 1.0), (r"^active_power$", 0.95),
        (r"^energy$", 0.95), (r"^load$", 0.90), (r"^demand$", 0.90), (r"^power$", 0.90),
        (r"^usage$", 0.85), (r"^electricity$", 0.85), (r"^kwh$", 0.85), (r"^kw$", 0.80),
        (r"^mw$", 0.80), (r"^mwh$", 0.80), (r"^value$", 0.60), (r"^target$", 0.60),
        (r".*consumption.*", 0.85), (r".*demand.*", 0.80), (r".*load.*", 0.75),
        (r".*energy.*", 0.75), (r".*power.*", 0.70),
    ]

    WEATHER_PATTERNS = {
        "temperature": [(r"^temperature$", 1.0), (r"^temp$", 0.95), (r"^temp_c$", 0.95), (r"^air_temp$", 0.90), (r"^deg_c$", 0.80), (r".*temp.*", 0.70)],
        "humidity": [(r"^humidity$", 1.0), (r"^hum$", 0.95), (r"^rh$", 0.90), (r"^relative_humidity$", 0.95), (r".*hum.*", 0.70)],
        "wind_speed": [(r"^wind_speed$", 1.0), (r"^windspeed$", 0.95), (r"^wind$", 0.90), (r"^wspd$", 0.85)],
        "pressure": [(r"^pressure$", 1.0), (r"^press$", 0.90), (r"^baro$", 0.85), (r"^barometer$", 0.90), (r"^pres$", 0.80)],
        "holiday": [(r"^holiday$", 1.0), (r"^is_holiday$", 1.0), (r"^public_holiday$", 0.95)],
    }

    def __init__(self, min_confidence: float = 0.40, ambiguity_delta: float = 0.05) -> None:
        self.min_confidence = min_confidence
        self.ambiguity_delta = ambiguity_delta

    def _match_name_patterns(self, col_name: str, patterns: List[Tuple[str, float]]) -> float:
        clean_name = col_name.strip().lower().replace(" ", "_").replace("-", "_")
        best_score = 0.0
        for pat, score in patterns:
            if re.fullmatch(pat, clean_name): return score
            if re.search(pat, clean_name): best_score = max(best_score, score * 0.8)
        return best_score

    def detect_timestamp_candidates(self, df: pd.DataFrame) -> List[ColumnCandidate]:
        candidates: List[ColumnCandidate] = []
        sample_size = min(len(df), 200)

        for col in df.columns:
            series = df[col]
            name_score = self._match_name_patterns(str(col), self.TIMESTAMP_PATTERNS)
            reasons = []
            sample_series = series.dropna().head(sample_size)
            if len(sample_series) == 0: continue

            parse_rate = 0.0
            is_monotonic = False
            try:
                if pd.api.types.is_datetime64_any_dtype(series):
                    parse_rate = 1.0
                    reasons.append("Already datetime dtype")
                elif pd.api.types.is_numeric_dtype(sample_series):
                    if sample_series.min() > 1e8:
                        parsed = pd.to_datetime(sample_series, unit="s", errors="coerce")
                        parse_rate = float(parsed.notna().mean())
                    else:
                        parse_rate = 0.0
                else:
                    parsed = pd.to_datetime(sample_series.astype(str), errors="coerce")
                    parse_rate = float(parsed.notna().mean())
                    valid_parsed = parsed.dropna()
                    if len(valid_parsed) > 5 and valid_parsed.is_monotonic_increasing:
                        is_monotonic = True
            except Exception:
                parse_rate = 0.0

            if parse_rate < 0.5 and name_score < 0.5: continue
            confidence = min(1.0, max(0.0, (0.55 * parse_rate) + (0.35 * name_score) + (0.10 if is_monotonic else 0.0)))
            candidates.append(ColumnCandidate(column_name=str(col), confidence=round(confidence, 4), reasons=reasons))

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def detect_target_candidates(self, df: pd.DataFrame, excluded_cols: List[str]) -> List[ColumnCandidate]:
        candidates: List[ColumnCandidate] = []
        for col in df.columns:
            if str(col) in excluded_cols: continue
            series = df[col]
            is_numeric = bool(pd.api.types.is_numeric_dtype(series))
            if not is_numeric:
                try:
                    num_series = pd.to_numeric(series.dropna().head(100), errors="coerce")
                    if float(num_series.notna().mean()) >= 0.8: is_numeric = True
                except Exception: is_numeric = False
            if not is_numeric: continue

            name_score = self._match_name_patterns(str(col), self.TARGET_PATTERNS)
            valid_nums = pd.to_numeric(series, errors="coerce").dropna()
            if len(valid_nums) == 0: continue

            std_val = float(valid_nums.std()) if len(valid_nums) > 1 else 0.0
            variance_penalty = 0.5 if std_val == 0.0 else 1.0
            negative_ratio = float((valid_nums < 0).mean())
            positivity_bonus = 0.1 if negative_ratio == 0.0 else (-0.2 if negative_ratio > 0.1 else 0.0)

            confidence = min(1.0, max(0.0, (0.60 * name_score + 0.30 * 1.0 + positivity_bonus) * variance_penalty))
            candidates.append(ColumnCandidate(column_name=str(col), confidence=round(confidence, 4)))

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def detect_optional_columns(self, df: pd.DataFrame, excluded_cols: List[str]) -> Dict[str, Optional[ColumnCandidate]]:
        results: Dict[str, Optional[ColumnCandidate]] = {}
        for role, patterns in self.WEATHER_PATTERNS.items():
            candidates: List[ColumnCandidate] = []
            for col in df.columns:
                if str(col) in excluded_cols: continue
                name_score = self._match_name_patterns(str(col), patterns)
                if name_score >= 0.6:
                    candidates.append(ColumnCandidate(column_name=str(col), confidence=name_score))
            candidates.sort(key=lambda c: c.confidence, reverse=True)
            if candidates and candidates[0].confidence >= self.min_confidence:
                results[role] = candidates[0]
                excluded_cols.append(candidates[0].column_name)
            else:
                results[role] = None
        return results

    def detect_schema(
        self,
        df: pd.DataFrame,
        override_timestamp: Optional[str] = None,
        override_consumption: Optional[str] = None,
    ) -> DetectedSchema:
        ts_candidates = self.detect_timestamp_candidates(df)
        if override_timestamp:
            selected_ts = override_timestamp
        else:
            if not ts_candidates or ts_candidates[0].confidence < self.min_confidence:
                raise SchemaDetectionError(f"No suitable timestamp column detected among {list(df.columns)}.")
            if len(ts_candidates) > 1 and (ts_candidates[0].confidence - ts_candidates[1].confidence) < self.ambiguity_delta and ts_candidates[1].confidence >= 0.7:
                raise AmbiguousSchemaError("timestamp", [(c.column_name, c.confidence) for c in ts_candidates[:3]])
            selected_ts = ts_candidates[0].column_name

        target_candidates = self.detect_target_candidates(df, excluded_cols=[selected_ts])
        if override_consumption:
            selected_target = override_consumption
        else:
            if not target_candidates or target_candidates[0].confidence < self.min_confidence:
                remaining = [c for c in df.columns if str(c) != selected_ts and pd.api.types.is_numeric_dtype(df[c])]
                if len(remaining) == 1:
                    selected_target = str(remaining[0])
                else:
                    raise SchemaDetectionError(f"No suitable energy consumption target column detected among {list(df.columns)}.")
            else:
                if len(target_candidates) > 1 and (target_candidates[0].confidence - target_candidates[1].confidence) < self.ambiguity_delta and target_candidates[1].confidence >= 0.7:
                    raise AmbiguousSchemaError("consumption", [(c.column_name, c.confidence) for c in target_candidates[:3]])
                selected_target = target_candidates[0].column_name

        used_cols = [selected_ts, selected_target]
        optional_matches = self.detect_optional_columns(df, excluded_cols=used_cols)

        return DetectedSchema(
            timestamp=selected_ts,
            consumption=selected_target,
            temperature=optional_matches["temperature"].column_name if optional_matches.get("temperature") else None,
            humidity=optional_matches["humidity"].column_name if optional_matches.get("humidity") else None,
            wind_speed=optional_matches["wind_speed"].column_name if optional_matches.get("wind_speed") else None,
            pressure=optional_matches["pressure"].column_name if optional_matches.get("pressure") else None,
            holiday=optional_matches["holiday"].column_name if optional_matches.get("holiday") else None,
        )


class FrequencyDetector:
    """Detects time-series frequency and sets adaptive lag/rolling configurations."""

    FREQ_DEFINITIONS = [
        ("15-minute", "15min", 900, 60, 96, 672),
        ("30-minute", "30min", 1800, 120, 48, 336),
        ("hourly", "1h", 3600, 300, 24, 168),
        ("daily", "1D", 86400, 3600, 1, 7),
        ("weekly", "1W", 604800, 21600, 0, 1),
    ]

    def detect_frequency(self, timestamps: pd.Series) -> FrequencyInfo:
        ts = pd.to_datetime(timestamps, errors="coerce").dropna()
        if len(ts) < 3:
            return FrequencyInfo(
                name="irregular", pandas_alias="1h", median_delta_seconds=3600.0,
                periods_per_day=24, periods_per_week=168, is_regular=False,
                irregular_gap_percentage=100.0,
                recommended_lag_indices={"lag_1": 1, "lag_2": 2, "lag_day": 24, "lag_week": 168},
                resampling_recommended=True,
            )

        deltas = ts.diff().dropna()
        delta_seconds = deltas.dt.total_seconds()
        valid_deltas = delta_seconds[delta_seconds > 0]
        median_sec = float(valid_deltas.median()) if len(valid_deltas) > 0 else 3600.0

        matched_freq = None
        for name, alias, target_sec, tol_sec, per_day, per_week in self.FREQ_DEFINITIONS:
            if abs(median_sec - target_sec) <= tol_sec:
                matched_freq = (name, alias, target_sec, per_day, per_week)
                break

        if matched_freq:
            name, alias, target_sec, per_day, per_week = matched_freq
            deviations = (delta_seconds < target_sec * 0.9) | (delta_seconds > target_sec * 1.1)
            irregular_pct = round(float(deviations.mean()) * 100.0, 2)
            is_reg = irregular_pct <= 15.0

            if name == "15-minute":
                lags = {"lag_1": 1, "lag_4": 4, "lag_8": 8, "lag_day": 96, "lag_week": 672}
            elif name == "30-minute":
                lags = {"lag_1": 1, "lag_2": 2, "lag_4": 4, "lag_day": 48, "lag_week": 336}
            elif name == "hourly":
                lags = {"lag_1": 1, "lag_2": 2, "lag_3": 3, "lag_day": 24, "lag_48": 48, "lag_week": 168}
            elif name == "daily":
                lags = {"lag_1": 1, "lag_2": 2, "lag_week": 7, "lag_14": 14, "lag_month": 30}
            elif name == "weekly":
                lags = {"lag_1": 1, "lag_2": 2, "lag_month": 4}
            else:
                lags = {"lag_1": 1, "lag_2": 2, "lag_day": 24, "lag_week": 168}

            return FrequencyInfo(
                name=name, pandas_alias=alias, median_delta_seconds=median_sec,
                periods_per_day=per_day, periods_per_week=per_week,
                is_regular=is_reg, irregular_gap_percentage=irregular_pct,
                recommended_lag_indices=lags, resampling_recommended=not is_reg,
            )
        else:
            est_per_day = max(1, int(round(86400 / max(1.0, median_sec))))
            return FrequencyInfo(
                name="irregular", pandas_alias="1h", median_delta_seconds=median_sec,
                periods_per_day=est_per_day, periods_per_week=est_per_day * 7,
                is_regular=False, irregular_gap_percentage=100.0,
                recommended_lag_indices={"lag_1": 1, "lag_2": 2, "lag_day": est_per_day, "lag_week": est_per_day * 7},
                resampling_recommended=True,
            )


class DataValidator:
    """Validates dataset sanity, variance, ordering, and peak anomalies."""

    def __init__(self, min_observations: int = 50, max_missing_ratio: float = 0.20) -> None:
        self.min_observations = min_observations
        self.max_missing_ratio = max_missing_ratio

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        errors: List[str] = []
        warnings: List[str] = []
        diag: Dict[str, Any] = {}

        if "timestamp" not in df.columns: errors.append("Required column 'timestamp' is missing.")
        if "consumption" not in df.columns: errors.append("Required column 'consumption' is missing.")
        if errors:
            return ValidationReport(is_valid=False, errors=errors, warnings=warnings, diagnostics=diag)

        n_rows = len(df)
        diag["num_rows"] = n_rows

        if n_rows < self.min_observations:
            errors.append(f"Insufficient observations: dataset has {n_rows} rows, but at least {self.min_observations} are required.")

        ts = df["timestamp"]
        parsed_ts = pd.to_datetime(ts, errors="coerce")
        null_ts = int(parsed_ts.isna().sum())
        if null_ts > 0: errors.append(f"Timestamp column contains {null_ts} unparseable datetime values.")

        if not parsed_ts.is_monotonic_increasing:
            warnings.append("Timestamps are not in strict chronological order. Pipeline will sort them.")

        dup_ts = int(parsed_ts.duplicated().sum())
        diag["duplicate_timestamps"] = dup_ts
        if dup_ts > 0:
            warnings.append(f"Detected {dup_ts} duplicate timestamps. They will be aggregated during normalization.")

        target = df["consumption"]
        num_target = pd.to_numeric(target, errors="coerce")
        missing_target = int(num_target.isna().sum())
        missing_ratio = (missing_target / n_rows) if n_rows > 0 else 1.0
        diag["missing_target_count"] = missing_target
        diag["missing_target_ratio"] = round(missing_ratio, 4)

        if missing_ratio > self.max_missing_ratio:
            errors.append(f"High missing target values: {missing_target} rows ({missing_ratio*100:.1f}%) are NaN.")
        elif missing_target > 0:
            warnings.append(f"Target contains {missing_target} missing values. Forward-fill will be applied.")

        valid_targets = num_target.dropna()
        if len(valid_targets) > 0:
            target_std = float(valid_targets.std()) if len(valid_targets) > 1 else 0.0
            if target_std < 1e-6 or float(valid_targets.min()) == float(valid_targets.max()):
                errors.append("Target series is constant (zero variance).")

            negative_count = int((valid_targets < 0).sum())
            diag["negative_target_count"] = negative_count
            if negative_count > 0:
                warnings.append(f"Target contains {negative_count} negative consumption values (generation offset).")

            q25, q75 = float(valid_targets.quantile(0.25)), float(valid_targets.quantile(0.75))
            spike_threshold = q75 + (3.0 * (q75 - q25))
            spikes = valid_targets[valid_targets > spike_threshold]
            diag["detected_energy_spikes"] = len(spikes)
            if len(spikes) > 0:
                warnings.append(f"Detected {len(spikes)} high energy demand peaks (> {spike_threshold:.2f}) preserved for training.")

        return ValidationReport(is_valid=len(errors) == 0, errors=errors, warnings=warnings, diagnostics=diag)


# =====================================================================
# 3. PREPROCESSING & FEATURE ENGINEERING LAYER
# =====================================================================

class SchemaNormalizer:
    """Maps arbitrary schemas into standard canonical DataFrame."""

    def normalize(self, df: pd.DataFrame, schema: Union[DetectedSchema, Dict[str, str]]) -> pd.DataFrame:
        mapping = schema.to_mapping() if isinstance(schema, DetectedSchema) else schema
        rename_dict = {raw_col: canon_col for canon_col, raw_col in mapping.items() if raw_col in df.columns}
        
        subset_df = df[list(rename_dict.keys())].copy()
        norm_df = subset_df.rename(columns=rename_dict)

        norm_df["timestamp"] = pd.to_datetime(norm_df["timestamp"], errors="coerce")
        norm_df = norm_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        if norm_df["timestamp"].duplicated().any():
            numeric_cols = [c for c in norm_df.columns if c != "timestamp"]
            norm_df = norm_df.groupby("timestamp", as_index=False)[numeric_cols].mean()

        for col in norm_df.columns:
            if col != "timestamp":
                norm_df[col] = pd.to_numeric(norm_df[col], errors="coerce").astype(np.float64)
                if norm_df[col].isna().any():
                    norm_df[col] = norm_df[col].ffill().bfill()

        return norm_df.reset_index(drop=True)


class TimeSeriesFeatureEngineer:
    """Generates calendar, cyclic, lag, and rolling features with ZERO data leakage."""

    def __init__(self, freq_info: FrequencyInfo) -> None:
        self.freq_info = freq_info
        self.lag_indices = freq_info.recommended_lag_indices
        per_day = max(1, freq_info.periods_per_day)

        if freq_info.name in ["15-minute", "30-minute", "hourly"]:
            short_win = max(3, per_day // 4)
            long_win = per_day
        elif freq_info.name == "daily":
            short_win, long_win = 3, 7
        else:
            short_win, long_win = 2, max(4, per_day)

        self.rolling_windows = {"short": short_win, "long": long_win}
        self.calendar_features = ["hour", "day", "day_of_week", "month", "quarter", "year", "is_weekend"]
        self.cyclic_features = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos"]

        max_lag = max(self.lag_indices.values()) if self.lag_indices else 1
        max_roll = max(self.rolling_windows.values()) if self.rolling_windows else 1
        self.warmup_rows = max(max_lag, max_roll + 1)
        self.config: Optional[FeatureEngineeringConfig] = None

    def _extract_calendar_and_cyclic(self, timestamps: pd.Series) -> pd.DataFrame:
        ts = pd.to_datetime(timestamps)
        df_cal = pd.DataFrame(index=ts.index)
        df_cal["hour"] = ts.dt.hour
        df_cal["day"] = ts.dt.day
        df_cal["day_of_week"] = ts.dt.dayofweek
        df_cal["month"] = ts.dt.month
        df_cal["quarter"] = ts.dt.quarter
        df_cal["year"] = ts.dt.year
        df_cal["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

        df_cal["hour_sin"] = np.sin(2 * np.pi * df_cal["hour"] / 24.0)
        df_cal["hour_cos"] = np.cos(2 * np.pi * df_cal["hour"] / 24.0)
        df_cal["weekday_sin"] = np.sin(2 * np.pi * df_cal["day_of_week"] / 7.0)
        df_cal["weekday_cos"] = np.cos(2 * np.pi * df_cal["day_of_week"] / 7.0)
        df_cal["month_sin"] = np.sin(2 * np.pi * (df_cal["month"] - 1) / 12.0)
        df_cal["month_cos"] = np.cos(2 * np.pi * (df_cal["month"] - 1) / 12.0)
        return df_cal

    def create_features(self, df: pd.DataFrame, drop_warmup: bool = True) -> Tuple[pd.DataFrame, FeatureEngineeringConfig]:
        feat_df = df.copy()
        cal_cyclic = self._extract_calendar_and_cyclic(feat_df["timestamp"])
        for col in cal_cyclic.columns: feat_df[col] = cal_cyclic[col]

        for lag_name, lag_steps in self.lag_indices.items():
            feat_df[lag_name] = feat_df["consumption"].shift(lag_steps)

        # STRICT SHIFT(1) TO PREVENT TARGET LEAKAGE
        target_shifted = feat_df["consumption"].shift(1)
        short_w, long_w = self.rolling_windows["short"], self.rolling_windows["long"]

        feat_df[f"rolling_mean_{short_w}"] = target_shifted.rolling(window=short_w, min_periods=1).mean()
        feat_df[f"rolling_std_{short_w}"] = target_shifted.rolling(window=short_w, min_periods=1).std().fillna(0.0)
        feat_df[f"rolling_mean_{long_w}"] = target_shifted.rolling(window=long_w, min_periods=1).mean()
        feat_df[f"rolling_std_{long_w}"] = target_shifted.rolling(window=long_w, min_periods=1).std().fillna(0.0)
        feat_df[f"rolling_min_{long_w}"] = target_shifted.rolling(window=long_w, min_periods=1).min()
        feat_df[f"rolling_max_{long_w}"] = target_shifted.rolling(window=long_w, min_periods=1).max()

        active_weather = [c for c in ["temperature", "humidity", "wind_speed", "pressure", "holiday"] if c in feat_df.columns]
        for w_col in active_weather: feat_df[f"{w_col}_lag_1"] = feat_df[w_col].shift(1)

        feature_cols = [c for c in feat_df.columns if c not in {"timestamp", "consumption"}]
        config = FeatureEngineeringConfig(
            frequency_name=self.freq_info.name,
            lag_indices=self.lag_indices,
            rolling_windows=self.rolling_windows,
            calendar_features=self.calendar_features,
            cyclic_features=self.cyclic_features,
            weather_features=active_weather,
            feature_columns=feature_cols,
            warmup_rows=self.warmup_rows,
        )
        self.config = config
        if drop_warmup: feat_df = feat_df.iloc[self.warmup_rows:].reset_index(drop=True)
        return feat_df, config

    def generate_single_step_features(
        self,
        history_df: pd.DataFrame,
        next_timestamp: pd.Timestamp,
        future_weather: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        if self.config is None:
            raise ValueError("Feature engineer not fitted. Call create_features first.")

        row_dict: Dict[str, float] = {}
        cal_cyclic = self._extract_calendar_and_cyclic(pd.Series([next_timestamp]))
        for col in cal_cyclic.columns: row_dict[col] = float(cal_cyclic[col].iloc[0])

        consumption_series = history_df["consumption"].values
        n_hist = len(consumption_series)

        for lag_name, lag_steps in self.config.lag_indices.items():
            row_dict[lag_name] = float(consumption_series[-lag_steps]) if n_hist >= lag_steps else float(consumption_series[0])

        short_w, long_w = self.config.rolling_windows["short"], self.config.rolling_windows["long"]
        short_hist = consumption_series[-short_w:] if n_hist >= short_w else consumption_series
        long_hist = consumption_series[-long_w:] if n_hist >= long_w else consumption_series

        row_dict[f"rolling_mean_{short_w}"] = float(np.mean(short_hist))
        row_dict[f"rolling_std_{short_w}"] = float(np.std(short_hist)) if len(short_hist) > 1 else 0.0
        row_dict[f"rolling_mean_{long_w}"] = float(np.mean(long_hist))
        row_dict[f"rolling_std_{long_w}"] = float(np.std(long_hist)) if len(long_hist) > 1 else 0.0
        row_dict[f"rolling_min_{long_w}"] = float(np.min(long_hist))
        row_dict[f"rolling_max_{long_w}"] = float(np.max(long_hist))

        for w_col in self.config.weather_features:
            val = float(future_weather[w_col]) if (future_weather and w_col in future_weather) else (
                float(history_df[w_col].iloc[-1]) if w_col in history_df.columns else 0.0
            )
            row_dict[w_col] = val
            row_dict[f"{w_col}_lag_1"] = float(history_df[w_col].iloc[-1]) if w_col in history_df.columns else val

        return pd.DataFrame([row_dict])[self.config.feature_columns]


# =====================================================================
# 4. CANDIDATE FORECASTING MODELS
# =====================================================================

class BaselineForecaster:
    """Naive & Seasonal-Naive baseline model."""

    def __init__(self, strategy: str = "last_value", seasonal_period: int = 1) -> None:
        self.strategy = strategy
        self.seasonal_period = max(1, seasonal_period)
        self.last_observed_value: Optional[float] = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineForecaster":
        y_arr = np.asarray(y, dtype=np.float64)
        if len(y_arr) == 0: raise ValueError("Empty target series.")
        self.last_observed_value = float(y_arr[-1])
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted: raise ValueError("Model must be fitted.")
        if len(X) == 0: return np.array([], dtype=np.float64)
        if self.strategy == "last_value" and "lag_1" in X.columns:
            return X["lag_1"].values.astype(np.float64)
        elif self.strategy == "seasonal" and "lag_day" in X.columns:
            return X["lag_day"].values.astype(np.float64)
        return np.full(shape=(len(X),), fill_value=self.last_observed_value, dtype=np.float64)


class RandomForestForecaster:
    """Random Forest regressor tailored for time-series features."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 12, random_state: int = 42, n_jobs: int = -1) -> None:
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=5, min_samples_leaf=2, random_state=random_state, n_jobs=n_jobs)
        self.feature_names_: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestForecaster":
        self.feature_names_ = list(X.columns)
        self.model.fit(X.values if isinstance(X, pd.DataFrame) else np.asarray(X), y.values if isinstance(y, pd.Series) else np.asarray(y))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted: raise ValueError("Model must be fitted.")
        X_mat = X[self.feature_names_].values if isinstance(X, pd.DataFrame) and self.feature_names_ else X
        return np.asarray(self.model.predict(X_mat), dtype=np.float64)

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_fitted or not hasattr(self.model, "feature_importances_"): return {}
        return {feat: round(float(imp), 4) for feat, imp in zip(self.feature_names_, self.model.feature_importances_)}


class XGBoostForecaster:
    """Extreme Gradient Boosting (XGBoost) forecaster."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.05, random_state: int = 42, n_jobs: int = -1) -> None:
        self.feature_names_: List[str] = []
        self.is_fitted = False
        if HAS_XGBOOST:
            self.model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.8, random_state=random_state, n_jobs=n_jobs, tree_method="hist")
        else:
            self.model = HistGradientBoostingRegressor(max_iter=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostForecaster":
        self.feature_names_ = list(X.columns)
        self.model.fit(X.values if isinstance(X, pd.DataFrame) else np.asarray(X), y.values if isinstance(y, pd.Series) else np.asarray(y))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted: raise ValueError("Model must be fitted.")
        X_mat = X[self.feature_names_].values if isinstance(X, pd.DataFrame) and self.feature_names_ else X
        return np.asarray(self.model.predict(X_mat), dtype=np.float64)

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_fitted or not hasattr(self.model, "feature_importances_"): return {}
        return {feat: round(float(imp), 4) for feat, imp in zip(self.feature_names_, self.model.feature_importances_)}


class SARIMAForecaster:
    """Seasonal ARIMA state-space model."""

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), seasonal_order: Tuple[int, int, int, int] = (1, 0, 0, 24), use_exog: bool = False) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_exog = use_exog
        self.fitted_model_ = None
        self.last_y_val_ = 0.0
        self.is_fitted = False
        self._fallback_ridge = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SARIMAForecaster":
        y_vec = y.values if isinstance(y, pd.Series) else np.asarray(y, dtype=np.float64)
        self.last_y_val_ = float(y_vec[-1]) if len(y_vec) > 0 else 0.0
        exog_data = X.values if (self.use_exog and X is not None and len(X.columns) > 0) else None

        if HAS_STATSMODELS:
            try:
                model = SARIMAX(endog=y_vec, exog=exog_data, order=self.order, seasonal_order=self.seasonal_order if self.seasonal_order[3] > 1 else (0,0,0,0), enforce_stationarity=False, enforce_invertibility=False)
                self.fitted_model_ = model.fit(disp=False, maxiter=50)
            except Exception:
                try:
                    fallback = SARIMAX(endog=y_vec, order=(1, 1, 0), enforce_stationarity=False, enforce_invertibility=False)
                    self.fitted_model_ = fallback.fit(disp=False, maxiter=30)
                except Exception:
                    self._fallback_ridge = Ridge(alpha=1.0).fit(X.values if isinstance(X, pd.DataFrame) else X, y_vec)
        else:
            self._fallback_ridge = Ridge(alpha=1.0).fit(X.values if isinstance(X, pd.DataFrame) else X, y_vec)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted: raise ValueError("Model must be fitted.")
        n_steps = len(X)
        if n_steps == 0: return np.array([], dtype=np.float64)
        if self.fitted_model_ is not None:
            try:
                exog_data = X.values if (self.use_exog and X is not None and len(X.columns) > 0) else None
                preds = np.asarray(self.fitted_model_.forecast(steps=n_steps, exog=exog_data), dtype=np.float64)
                return np.nan_to_num(preds, nan=self.last_y_val_)
            except Exception:
                pass
        if self._fallback_ridge is not None:
            return self._fallback_ridge.predict(X.values if isinstance(X, pd.DataFrame) else X).astype(np.float64)
        return np.full(shape=(n_steps,), fill_value=self.last_y_val_, dtype=np.float64)


# =====================================================================
# 5. EVALUATION, MODEL SELECTION & FUTURE FORECASTER
# =====================================================================

class ModelEvaluator:
    """Computes MAE, RMSE, WAPE, R2, and epsilon-safe MAPE."""

    def __init__(self, epsilon: float = 1e-5) -> None:
        self.epsilon = epsilon

    def evaluate(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], model_name: str = "Model") -> EvaluationMetrics:
        yt = np.asarray(y_true, dtype=np.float64).ravel()
        yp = np.asarray(y_pred, dtype=np.float64).ravel()

        if len(yt) != len(yp) or len(yt) == 0:
            raise ValueError("Invalid array sizes.")

        errors = yt - yp
        abs_errors = np.abs(errors)
        sq_errors = np.square(errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(sq_errors)))
        safe_denom = np.maximum(np.abs(yt), self.epsilon)
        mape = float(np.mean(abs_errors / safe_denom) * 100.0)
        sum_yt = float(np.sum(np.abs(yt)))
        wape = float((np.sum(abs_errors) / max(sum_yt, self.epsilon)) * 100.0)

        ss_tot = float(np.sum(np.square(yt - np.mean(yt))))
        r2 = float(1.0 - (np.sum(sq_errors) / max(ss_tot, self.epsilon))) if ss_tot > 0 else 0.0

        return EvaluationMetrics(model_name=model_name, mae=round(mae, 4), rmse=round(rmse, 4), mape=round(mape, 2), wape=round(wape, 2), r2=round(r2, 4), sample_count=len(yt))

    def create_comparison_table(self, metrics_list: List[EvaluationMetrics]) -> pd.DataFrame:
        rows = [{"Model": m.model_name, "MAE": m.mae, "RMSE": m.rmse, "MAPE (%)": f"{m.mape:.2f}%", "WAPE (%)": f"{m.wape:.2f}%", "R2": m.r2} for m in metrics_list]
        return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


class ModelSelector:
    """Selects top validation model, retrains on Train + Val history, and scores Test set."""

    def __init__(self, evaluator: Optional[ModelEvaluator] = None, selection_metric: str = "mae") -> None:
        self.evaluator = evaluator or ModelEvaluator()
        self.selection_metric = selection_metric.lower()

    def get_default_candidates(self, seasonal_period: int = 24) -> Dict[str, Any]:
        return {
            "Naive Baseline": BaselineForecaster(strategy="last_value", seasonal_period=seasonal_period),
            "Random Forest": RandomForestForecaster(n_estimators=100, max_depth=12, random_state=42),
            "XGBoost": XGBoostForecaster(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42),
            "SARIMA": SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 0, 0, min(seasonal_period, 24))),
        }

    def select_and_retrain(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series,
        seasonal_period: int = 24,
    ) -> Tuple[str, Any, Dict[str, EvaluationMetrics], EvaluationMetrics, pd.DataFrame]:
        models = self.get_default_candidates(seasonal_period=seasonal_period)
        val_metrics: Dict[str, EvaluationMetrics] = {}

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                val_metrics[name] = self.evaluator.evaluate(y_true=y_val, y_pred=preds, model_name=name)
            except Exception:
                val_metrics[name] = EvaluationMetrics(model_name=name, mae=float("inf"), rmse=float("inf"), mape=float("inf"), wape=float("inf"), r2=-1.0, sample_count=len(y_val))

        comparison_table = self.evaluator.create_comparison_table(list(val_metrics.values()))
        best_name = min(val_metrics.keys(), key=lambda k: getattr(val_metrics[k], self.selection_metric, val_metrics[k].mae))

        # Retrain on full historical data (Train + Validation)
        X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_train_val = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

        retrained_model = self.get_default_candidates(seasonal_period=seasonal_period)[best_name]
        retrained_model.fit(X_train_val, y_train_val)

        # Out-of-sample Test evaluation
        test_preds = retrained_model.predict(X_test)
        test_metrics = self.evaluator.evaluate(y_true=y_test, y_pred=test_preds, model_name=best_name)

        return best_name, retrained_model, val_metrics, test_metrics, comparison_table


class FutureForecaster:
    """Recursive multi-step forecasting engine with dynamic feature regeneration."""

    def __init__(self, model: Any, feature_engineer: TimeSeriesFeatureEngineer, freq_info: FrequencyInfo) -> None:
        self.model = model
        self.feature_engineer = feature_engineer
        self.freq_info = freq_info

    def generate_forecast(self, history_df: pd.DataFrame, horizon: int = 24, future_weather: Optional[List[Dict[str, float]]] = None) -> ForecastResult:
        buffer_df = history_df.copy().reset_index(drop=True)
        buffer_df["timestamp"] = pd.to_datetime(buffer_df["timestamp"])
        curr_ts = buffer_df["timestamp"].iloc[-1]
        delta = pd.to_timedelta(self.freq_info.median_delta_seconds, unit="s")

        forecast_points: List[ForecastPoint] = []
        forecast_rows: List[Dict[str, Any]] = []

        for step in range(1, horizon + 1):
            curr_ts = curr_ts + delta
            step_weather = future_weather[step - 1] if (future_weather and step - 1 < len(future_weather)) else None

            feat_row = self.feature_engineer.generate_single_step_features(history_df=buffer_df, next_timestamp=curr_ts, future_weather=step_weather)
            pred_val = max(0.0, float(self.model.predict(feat_row)[0]))

            forecast_points.append(ForecastPoint(timestamp=curr_ts.isoformat(), predicted_consumption=pred_val, step_ahead=step))
            forecast_rows.append({"timestamp": curr_ts, "predicted_consumption": pred_val, "step_ahead": step})

            new_row = {"timestamp": curr_ts, "consumption": pred_val}
            for w_col in self.feature_engineer.config.weather_features:
                new_row[w_col] = step_weather[w_col] if (step_weather and w_col in step_weather) else (buffer_df[w_col].iloc[-1] if w_col in buffer_df.columns else 0.0)

            buffer_df = pd.concat([buffer_df, pd.DataFrame([new_row])], ignore_index=True)

        return ForecastResult(
            model_name=getattr(self.model, "__class__", type(self.model)).__name__,
            horizon=horizon,
            start_timestamp=forecast_points[0].timestamp if forecast_points else "",
            end_timestamp=forecast_points[-1].timestamp if forecast_points else "",
            forecasts=forecast_points,
            forecast_df=pd.DataFrame(forecast_rows),
        )


# =====================================================================
# 6. PIPELINE ORCHESTRATOR
# =====================================================================

class EnergyForecastingPipeline:
    """Production-grade orchestrator for energy consumption time-series forecasting."""

    def __init__(
        self,
        models_dir: str = "models",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        default_horizon: int = 24,
    ) -> None:
        self.models_dir = models_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.default_horizon = default_horizon

        os.makedirs(self.models_dir, exist_ok=True)
        self.profiler = DatasetProfiler()
        self.schema_detector = SchemaDetector()
        self.normalizer = SchemaNormalizer()
        self.validator = DataValidator()
        self.frequency_detector = FrequencyDetector()
        self.evaluator = ModelEvaluator()
        self.model_selector = ModelSelector(evaluator=self.evaluator)

    def _split_chronological(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        X_train = df.iloc[:train_end][feature_cols].copy().reset_index(drop=True)
        y_train = df.iloc[:train_end]["consumption"].copy().reset_index(drop=True)

        X_val = df.iloc[train_end:val_end][feature_cols].copy().reset_index(drop=True)
        y_val = df.iloc[train_end:val_end]["consumption"].copy().reset_index(drop=True)

        X_test = df.iloc[val_end:][feature_cols].copy().reset_index(drop=True)
        y_test = df.iloc[val_end:]["consumption"].copy().reset_index(drop=True)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def run(
        self,
        data_source: Union[str, pd.DataFrame],
        horizon: Optional[int] = None,
        override_timestamp: Optional[str] = None,
        override_consumption: Optional[str] = None,
        future_weather: Optional[List[Dict[str, float]]] = None,
        model_tag: str = "energy_model",
    ) -> PipelineOutput:
        forecast_horizon = horizon or self.default_horizon
        raw_df = pd.read_csv(data_source) if isinstance(data_source, str) else data_source.copy()

        profile = self.profiler.profile(raw_df)
        detected_schema = self.schema_detector.detect_schema(df=raw_df, override_timestamp=override_timestamp, override_consumption=override_consumption)

        normalized_df = self.normalizer.normalize(df=raw_df, schema=detected_schema)
        val_report = self.validator.validate(normalized_df)
        if not val_report.is_valid:
            raise PipelineValidationError(val_report.errors)

        freq_info = self.frequency_detector.detect_frequency(normalized_df["timestamp"])
        feature_engineer = TimeSeriesFeatureEngineer(freq_info=freq_info)
        featured_df, feat_config = feature_engineer.create_features(normalized_df, drop_warmup=True)

        if len(featured_df) < 30:
            raise PipelineValidationError([f"Sample count after warmup ({len(featured_df)}) is too small for chronological splitting."])

        X_train, y_train, X_val, y_val, X_test, y_test = self._split_chronological(featured_df, feat_config.feature_columns)
        seasonal_period = freq_info.periods_per_day if freq_info.periods_per_day > 1 else 7

        best_name, best_model, val_metrics, test_metrics, comp_table = self.model_selector.select_and_retrain(
            X_train, y_train, X_val, y_val, X_test, y_test, seasonal_period=seasonal_period
        )

        future_forecaster = FutureForecaster(model=best_model, feature_engineer=feature_engineer, freq_info=freq_info)
        forecast_res = future_forecaster.generate_forecast(history_df=normalized_df, horizon=forecast_horizon, future_weather=future_weather)

        model_path = os.path.join(self.models_dir, f"{model_tag}_{best_name.lower().replace(' ', '_')}.joblib")
        metadata_path = os.path.join(self.models_dir, f"{model_tag}_metadata.json")

        joblib.dump(best_model, model_path)
        metadata_payload = {
            "selected_model": best_name,
            "detected_schema": detected_schema.to_mapping(),
            "frequency_info": freq_info.to_dict(),
            "feature_config": {"feature_columns": feat_config.feature_columns, "lag_indices": feat_config.lag_indices, "rolling_windows": feat_config.rolling_windows},
            "validation_metrics": {k: v.to_dict() for k, v in val_metrics.items()},
            "test_metrics": test_metrics.to_dict(),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)

        return PipelineOutput(
            dataset_profile=profile.to_dict(),
            schema=detected_schema.to_mapping(),
            frequency=freq_info.to_dict(),
            validation=val_report.to_dict(),
            features=feat_config.feature_columns,
            model_results={k: v.to_dict() for k, v in val_metrics.items()},
            selected_model=best_name,
            test_metrics=test_metrics.to_dict(),
            forecast=forecast_res.to_dict()["forecasts"],
            model_path=os.path.abspath(model_path),
            metadata_path=os.path.abspath(metadata_path),
        )


# =====================================================================
# 7. SYNTHETIC DATA GENERATOR & CLI DEMO RUNNER
# =====================================================================

def generate_sample_datasets(target_dir: str = "data") -> None:
    """Generate 5 distinct valid time-series schemas and 1 invalid CSV for demonstration."""
    os.makedirs(target_dir, exist_ok=True)
    np.random.seed(42)

    # 1. Dataset 1: Hourly - DateTime, Energy_Usage (720 rows)
    d1 = pd.date_range("2026-01-01", periods=720, freq="1h")
    h1 = d1.hour.to_numpy()
    e1 = np.maximum(2.0, 15.0 + 8.0 * np.sin(2 * np.pi * (h1 - 6) / 24) + np.random.normal(0, 1.5, 720))
    pd.DataFrame({"DateTime": d1.strftime("%Y-%m-%d %H:%M:%S"), "Energy_Usage": np.round(e1, 2)}).to_csv(os.path.join(target_dir, "dataset1.csv"), index=False)

    # 2. Dataset 2: 30-min - timestamp, load (1000 rows)
    d2 = pd.date_range("2026-02-01", periods=1000, freq="30min")
    h2 = d2.hour.to_numpy() + d2.minute.to_numpy() / 60.0
    l2 = 120.0 + 40.0 * np.sin(2 * np.pi * (h2 - 7) / 24) + np.random.normal(0, 5.0, 1000)
    l2[350] += 80.0 # Peak event
    pd.DataFrame({"timestamp": d2.strftime("%Y-%m-%dT%H:%M:%S"), "load": np.round(l2, 2)}).to_csv(os.path.join(target_dir, "dataset2.csv"), index=False)

    # 3. Dataset 3: 15-min with Weather - reading_time, power_consumption, temperature, humidity (1500 rows)
    d3 = pd.date_range("2026-03-01", periods=1500, freq="15min")
    h3 = d3.hour.to_numpy() + d3.minute.to_numpy() / 60.0
    t3 = 18.0 + 10.0 * np.sin(2 * np.pi * (h3 - 9) / 24) + np.random.normal(0, 1.0, 1500)
    hum3 = 60.0 - 20.0 * np.sin(2 * np.pi * (h3 - 9) / 24) + np.random.normal(0, 3.0, 1500)
    p3 = 45.0 + 15.0 * np.sin(2 * np.pi * (h3 - 6) / 24) + 0.8 * np.maximum(0, t3 - 22.0) + np.random.normal(0, 2.0, 1500)
    pd.DataFrame({"reading_time": d3.strftime("%m/%d/%Y %H:%M"), "power_consumption": np.round(p3, 2), "temperature": np.round(t3, 1), "humidity": np.round(hum3, 1)}).to_csv(os.path.join(target_dir, "dataset3.csv"), index=False)

    # 4. Dataset 4: Daily - date, electricity_demand (365 rows)
    d4 = pd.date_range("2025-01-01", periods=365, freq="1D")
    dem4 = 500.0 + 120.0 * np.sin(2 * np.pi * (d4.dayofyear.to_numpy() - 15) / 365) + np.random.normal(0, 20.0, 365)
    pd.DataFrame({"date": d4.strftime("%Y-%m-%d"), "electricity_demand": np.round(dem4, 1)}).to_csv(os.path.join(target_dir, "dataset4.csv"), index=False)

    # 5. Dataset 5: Hourly with Temp - time, value, temp (600 rows)
    d5 = pd.date_range("2026-04-01", periods=600, freq="1h")
    h5 = d5.hour.to_numpy()
    t5 = 15.0 + 8.0 * np.sin(2 * np.pi * (h5 - 8) / 24) + np.random.normal(0, 1.2, 600)
    v5 = 30.0 + 12.0 * np.sin(2 * np.pi * (h5 - 5) / 24) + 0.5 * t5 + np.random.normal(0, 1.8, 600)
    pd.DataFrame({"time": d5.strftime("%Y-%m-%d %H:%M"), "value": np.round(v5, 2), "temp": np.round(t5, 1)}).to_csv(os.path.join(target_dir, "dataset5.csv"), index=False)

    # 6. Invalid non-time-series CSV
    pd.DataFrame({"Name": ["Alice", "Bob", "Charlie", "Diana"], "Age": [24, 30, 22, 28], "Marks": [88.5, 92.0, 79.5, 95.0]}).to_csv(os.path.join(target_dir, "invalid.csv"), index=False)


def run_pipeline_on_file(file_path: str, horizon: int = 24, models_dir: str = "models") -> Optional[PipelineOutput]:
    print("\n" + "=" * 75)
    print(f"  PROCESSING DATASET: {os.path.basename(file_path)}")
    print("=" * 75)
    pipeline = EnergyForecastingPipeline(models_dir=models_dir, default_horizon=horizon)

    try:
        output = pipeline.run(data_source=file_path, horizon=horizon, model_tag=os.path.splitext(os.path.basename(file_path))[0])

        print("\n--- 1. DATASET INTELLIGENCE & SCHEMA ---")
        p = output.dataset_profile
        print(f"Rows: {p['num_rows']}, Columns: {p['num_columns']}")
        print(f"Detected Schema: {output.schema}")
        print(f"Detected Frequency: {output.frequency['name']} (alias: {output.frequency['pandas_alias']})")

        print("\n--- 2. CANDIDATE MODEL VALIDATION LEADERBOARD ---")
        rows = [{"Model": k, "MAE": v["mae"], "RMSE": v["rmse"], "MAPE (%)": f"{v['mape']:.2f}%", "R2": v["r2"]} for k, v in output.model_results.items()]
        print(pd.DataFrame(rows).sort_values("MAE").to_string(index=False))

        print(f"\n--- 3. MODEL SELECTION & HELD-OUT TEST PERFORMANCE ---")
        print(f"Selected Winner: >>> {output.selected_model} <<< (Lowest Validation MAE)")
        tm = output.test_metrics
        print(f"  MAE:  {tm['mae']:.4f} | RMSE: {tm['rmse']:.4f} | MAPE: {tm['mape']:.2f}% | R2: {tm['r2']:.4f}")

        print(f"\n--- 4. RECURSIVE MULTI-STEP FUTURE FORECAST (Horizon={horizon}) ---")
        df_forecast = pd.DataFrame(output.forecast)
        print(df_forecast.head(5).to_string(index=False))
        if len(df_forecast) > 5:
            print(f"... [{len(df_forecast) - 5} more forecast points] ...")
            print(df_forecast.tail(2).to_string(index=False))

        print(f"\n--- 5. ARTIFACT PERSISTENCE ---")
        print(f"Saved Model:    {output.model_path}")
        print(f"Saved Metadata: {output.metadata_path}")
        return output

    except (SchemaDetectionError, PipelineValidationError) as e:
        print(f"\n[PIPELINE REJECTED DATASET] Expected error: {str(e)}")
        return None
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR]: {str(e)}")
        raise e


def run_full_demo(data_dir: str = "data") -> None:
    print("\n" + "#" * 75)
    print("  ENERGY CONSUMPTION FORECASTING PIPELINE — FULL DEMO")
    print("#" * 75)
    generate_sample_datasets(data_dir)

    for fname in ["dataset1.csv", "dataset2.csv", "dataset3.csv", "dataset4.csv", "dataset5.csv", "invalid.csv"]:
        fpath = os.path.join(data_dir, fname)
        run_pipeline_on_file(fpath, horizon=24)


def main() -> None:
    parser = argparse.ArgumentParser(description="Energy Consumption Forecasting Single-File ML Pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to input energy CSV file.")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon (default: 24).")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save models.")
    parser.add_argument("--demo", action="store_true", help="Run demo across all sample datasets.")
    args = parser.parse_args()

    if args.demo or not args.data:
        run_full_demo()
    else:
        run_pipeline_on_file(args.data, horizon=args.horizon, models_dir=args.output_dir)


if __name__ == "__main__":
    main()
