"""Intelligent schema detection for heterogeneous energy time-series datasets."""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


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


@dataclass
class ColumnCandidate:
    """A scored candidate column for a specific canonical role."""
    column_name: str
    confidence: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class DetectedSchema:
    """Canonical mapping from raw column names to standard internal names."""
    timestamp: str
    consumption: str
    temperature: Optional[str] = None
    humidity: Optional[str] = None
    wind_speed: Optional[str] = None
    pressure: Optional[str] = None
    holiday: Optional[str] = None

    def to_mapping(self) -> Dict[str, str]:
        """Return dict of {canonical_name: raw_column_name}."""
        mapping = {
            "timestamp": self.timestamp,
            "consumption": self.consumption,
        }
        if self.temperature:
            mapping["temperature"] = self.temperature
        if self.humidity:
            mapping["humidity"] = self.humidity
        if self.wind_speed:
            mapping["wind_speed"] = self.wind_speed
        if self.pressure:
            mapping["pressure"] = self.pressure
        if self.holiday:
            mapping["holiday"] = self.holiday
        return mapping


@dataclass
class SchemaDetectionResult:
    """Result of schema detection with telemetry and candidate rankings."""
    detected_schema: DetectedSchema
    timestamp_candidates: List[ColumnCandidate]
    target_candidates: List[ColumnCandidate]
    optional_matches: Dict[str, Optional[ColumnCandidate]]
    confidence_summary: Dict[str, float]


class SchemaDetector:
    """Identifies canonical time-series roles from raw CSV column names and data traits."""

    TIMESTAMP_PATTERNS = [
        (r"^timestamp$", 1.0),
        (r"^datetime$", 1.0),
        (r"^date_time$", 1.0),
        (r"^reading_time$", 0.95),
        (r"^recorded_at$", 0.95),
        (r"^measurement_time$", 0.95),
        (r"^record_time$", 0.95),
        (r"^time_stamp$", 0.95),
        (r"^date$", 0.85),
        (r"^time$", 0.85),
        (r".*time.*", 0.65),
        (r".*date.*", 0.65),
        (r".*period.*", 0.50),
        (r"^utc_timestamp$", 1.0),
    ]

    TARGET_PATTERNS = [
        (r"^consumption$", 1.0),
        (r"^energy_usage$", 1.0),
        (r"^power_consumption$", 1.0),
        (r"^electricity_demand$", 1.0),
        (r"^electricity_usage$", 1.0),
        (r"^active_power$", 0.95),
        (r"^energy$", 0.95),
        (r"^load$", 0.90),
        (r"^demand$", 0.90),
        (r"^power$", 0.90),
        (r"^usage$", 0.85),
        (r"^electricity$", 0.85),
        (r"^kwh$", 0.85),
        (r"^kw$", 0.80),
        (r"^mw$", 0.80),
        (r"^mwh$", 0.80),
        (r"^value$", 0.60),
        (r"^target$", 0.60),
        (r".*consumption.*", 0.85),
        (r".*demand.*", 0.80),
        (r".*load.*", 0.75),
        (r".*energy.*", 0.75),
        (r".*power.*", 0.70),
    ]

    WEATHER_PATTERNS = {
        "temperature": [
            (r"^temperature$", 1.0),
            (r"^temp$", 0.95),
            (r"^temp_c$", 0.95),
            (r"^temperature_c$", 0.95),
            (r"^air_temp$", 0.90),
            (r"^deg_c$", 0.80),
            (r".*temperature.*", 0.80),
            (r".*temp.*", 0.70),
        ],
        "humidity": [
            (r"^humidity$", 1.0),
            (r"^hum$", 0.95),
            (r"^rh$", 0.90),
            (r"^relative_humidity$", 0.95),
            (r".*humidity.*", 0.80),
            (r".*hum.*", 0.70),
        ],
        "wind_speed": [
            (r"^wind_speed$", 1.0),
            (r"^windspeed$", 0.95),
            (r"^wind$", 0.90),
            (r"^wspd$", 0.85),
            (r".*wind.*", 0.75),
        ],
        "pressure": [
            (r"^pressure$", 1.0),
            (r"^press$", 0.90),
            (r"^baro$", 0.85),
            (r"^barometer$", 0.90),
            (r"^pres$", 0.80),
            (r".*pressure.*", 0.75),
        ],
        "holiday": [
            (r"^holiday$", 1.0),
            (r"^is_holiday$", 1.0),
            (r"^public_holiday$", 0.95),
            (r".*holiday.*", 0.80),
        ],
    }

    def __init__(self, min_confidence: float = 0.40, ambiguity_delta: float = 0.05) -> None:
        """Initialize detector.
        
        Args:
            min_confidence: Minimum score to accept a column candidate.
            ambiguity_delta: Maximum delta between top 2 candidates to trigger ambiguity warning.
        """
        self.min_confidence = min_confidence
        self.ambiguity_delta = ambiguity_delta

    def _match_name_patterns(self, col_name: str, patterns: List[Tuple[str, float]]) -> float:
        """Match sanitized column name against regex patterns."""
        clean_name = col_name.strip().lower().replace(" ", "_").replace("-", "_")
        best_score = 0.0
        for pat, score in patterns:
            if re.fullmatch(pat, clean_name):
                return score
            if re.search(pat, clean_name):
                best_score = max(best_score, score * 0.8)
        return best_score

    def detect_timestamp_candidates(self, df: pd.DataFrame) -> List[ColumnCandidate]:
        """Score all columns for suitability as the timestamp index."""
        candidates: List[ColumnCandidate] = []
        n_rows = len(df)
        sample_size = min(n_rows, 200)

        for col in df.columns:
            series = df[col]
            clean_name = str(col).strip()
            name_score = self._match_name_patterns(clean_name, self.TIMESTAMP_PATTERNS)
            reasons: List[str] = []

            # 1. Parseability test
            sample_series = series.dropna().head(sample_size)
            if len(sample_series) == 0:
                continue

            parse_rate = 0.0
            is_monotonic = False
            try:
                # If already datetime
                if pd.api.types.is_datetime64_any_dtype(series):
                    parse_rate = 1.0
                    reasons.append("Already datetime dtype")
                else:
                    # Convert to string and test datetime parsing
                    str_sample = sample_series.astype(str)
                    # Don't mistake purely small integers/floats (e.g. 1..100) for timestamps easily
                    if pd.api.types.is_numeric_dtype(sample_series):
                        # Numeric epoch timestamps (e.g. > 1e8) could be timestamps
                        min_val = sample_series.min()
                        if min_val > 1e8:
                            parsed = pd.to_datetime(sample_series, unit="s", errors="coerce")
                            parse_rate = float(parsed.notna().mean())
                            reasons.append("Parsed numeric epoch timestamp")
                        else:
                            # Not an epoch timestamp, likely an ID or small counter
                            parse_rate = 0.0
                    else:
                        parsed = pd.to_datetime(str_sample, errors="coerce")
                        parse_rate = float(parsed.notna().mean())
                        if parse_rate > 0.5:
                            reasons.append(f"Successfully parsed {parse_rate*100:.1f}% sample values as datetime")
                        
                        # Check monotonicity if parsed
                        valid_parsed = parsed.dropna()
                        if len(valid_parsed) > 5 and valid_parsed.is_monotonic_increasing:
                            is_monotonic = True
                            reasons.append("Sample timestamps are monotonically increasing")
            except Exception as e:
                parse_rate = 0.0

            if parse_rate < 0.5 and name_score < 0.5:
                continue

            # Composite timestamp confidence
            confidence = (0.55 * parse_rate) + (0.35 * name_score) + (0.10 if is_monotonic else 0.0)
            confidence = min(1.0, max(0.0, confidence))

            if name_score > 0.7:
                reasons.append(f"Column name strongly matches timestamp keywords (score {name_score:.2f})")

            candidates.append(
                ColumnCandidate(
                    column_name=str(col),
                    confidence=round(confidence, 4),
                    reasons=reasons,
                )
            )

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def detect_target_candidates(
        self, df: pd.DataFrame, excluded_cols: List[str]
    ) -> List[ColumnCandidate]:
        """Score numeric columns for energy consumption target role."""
        candidates: List[ColumnCandidate] = []

        for col in df.columns:
            if str(col) in excluded_cols:
                continue

            series = df[col]
            reasons: List[str] = []
            
            # Target must be numeric or convertable to numeric
            is_numeric = bool(pd.api.types.is_numeric_dtype(series))
            numeric_ratio = 1.0
            if not is_numeric:
                try:
                    num_series = pd.to_numeric(series.dropna().head(100), errors="coerce")
                    numeric_ratio = float(num_series.notna().mean())
                    if numeric_ratio >= 0.8:
                        is_numeric = True
                        reasons.append(f"Contains {numeric_ratio*100:.1f}% numeric values")
                except Exception:
                    is_numeric = False

            if not is_numeric:
                continue

            clean_name = str(col).strip()
            name_score = self._match_name_patterns(clean_name, self.TARGET_PATTERNS)

            # Reject columns that do not look like energy-related targets
            if name_score < 0.5:
                continue

            # Check value distribution
            valid_nums = pd.to_numeric(series, errors="coerce").dropna()
            if len(valid_nums) == 0:
                continue

            std_val = float(valid_nums.std()) if len(valid_nums) > 1 else 0.0
            if std_val == 0.0:
                reasons.append("Warning: Column has zero variance (constant value)")
                variance_penalty = 0.5
            else:
                variance_penalty = 1.0

            # Energy consumption is typically non-negative
            negative_ratio = float((valid_nums < 0).mean())
            positivity_bonus = 0.1 if negative_ratio == 0.0 else (-0.2 if negative_ratio > 0.1 else 0.0)

            if name_score > 0:
                reasons.append(f"Name match score: {name_score:.2f}")

            confidence = (0.60 * name_score + 0.30 * numeric_ratio + positivity_bonus) * variance_penalty
            confidence = min(1.0, max(0.0, confidence))

            candidates.append(
                ColumnCandidate(
                    column_name=str(col),
                    confidence=round(confidence, 4),
                    reasons=reasons,
                )
            )

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def detect_optional_columns(
        self, df: pd.DataFrame, excluded_cols: List[str]
    ) -> Dict[str, Optional[ColumnCandidate]]:
        """Identify optional weather and environmental predictors."""
        results: Dict[str, Optional[ColumnCandidate]] = {}

        for role, patterns in self.WEATHER_PATTERNS.items():
            candidates: List[ColumnCandidate] = []
            for col in df.columns:
                if str(col) in excluded_cols:
                    continue
                series = df[col]
                name_score = self._match_name_patterns(str(col), patterns)
                if name_score >= 0.6:
                    is_num = bool(pd.api.types.is_numeric_dtype(series))
                    conf = name_score * (1.0 if is_num else 0.7)
                    candidates.append(
                        ColumnCandidate(
                            column_name=str(col),
                            confidence=round(conf, 4),
                            reasons=[f"Matches '{role}' pattern with score {name_score:.2f}"],
                        )
                    )
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
    ) -> SchemaDetectionResult:
        """Run full schema detection on a DataFrame.

        Args:
            df: Input raw DataFrame.
            override_timestamp: User-specified timestamp column name.
            override_consumption: User-specified target column name.

        Returns:
            SchemaDetectionResult containing selected schema and candidates.

        Raises:
            SchemaDetectionError: When no valid timestamp or target can be discovered.
            AmbiguousSchemaError: When required candidates are too close.
        """

        # ============================================================
        # 1. TIMESTAMP DETECTION
        # ============================================================
        ts_candidates = self.detect_timestamp_candidates(df)

        if override_timestamp:
            if override_timestamp not in df.columns:
                raise SchemaDetectionError(
                    f"Provided override timestamp '{override_timestamp}' not in columns."
                )

            selected_ts = override_timestamp
            ts_conf = 1.0

        else:
            if not ts_candidates or ts_candidates[0].confidence < self.min_confidence:
                raise SchemaDetectionError(
                    f"No suitable timestamp column detected among {list(df.columns)}. "
                    "Ensure your dataset contains a datetime, date, timestamp, "
                    "or reading_time column."
                )

            # Check timestamp ambiguity
            if (
                len(ts_candidates) > 1
                and (
                    ts_candidates[0].confidence
                    - ts_candidates[1].confidence
                ) < self.ambiguity_delta
                and ts_candidates[1].confidence >= 0.7
            ):
                raise AmbiguousSchemaError(
                    "timestamp",
                    [
                        (c.column_name, c.confidence)
                        for c in ts_candidates[:3]
                    ],
                )

            selected_ts = ts_candidates[0].column_name
            ts_conf = ts_candidates[0].confidence

        # ============================================================
        # 2. TARGET / ENERGY CONSUMPTION DETECTION
        # ============================================================

        target_candidates = self.detect_target_candidates(
            df,
            excluded_cols=[selected_ts],
        )

        # IMPORTANT:
        # Always initialize these variables before any conditional branch.
        # This prevents:
        # "cannot access local variable 'selected_target'"
        selected_target = None
        target_conf = 0.0

        # ------------------------------------------------------------
        # Case A: User explicitly supplied target column
        # ------------------------------------------------------------
        if override_consumption:

            if override_consumption not in df.columns:
                raise SchemaDetectionError(
                    f"Provided override consumption "
                    f"'{override_consumption}' not in columns."
                )

            selected_target = override_consumption
            target_conf = 1.0

            # Add override column to candidates if not already present
            if not any(
                c.column_name == selected_target
                for c in target_candidates
            ):
                target_candidates.insert(
                    0,
                    ColumnCandidate(
                        column_name=selected_target,
                        confidence=target_conf,
                        reasons=["Explicitly provided target column"],
                    ),
                )

        # ------------------------------------------------------------
        # Case B: Automatically detect target
        # ------------------------------------------------------------
        else:

            # --------------------------------------------------------
            # B1. Strong energy-related target detected
            # --------------------------------------------------------
            if (
                target_candidates
                and target_candidates[0].confidence >= self.min_confidence
            ):

                # Check target ambiguity
                if (
                    len(target_candidates) > 1
                    and (
                        target_candidates[0].confidence
                        - target_candidates[1].confidence
                    ) < self.ambiguity_delta
                    and target_candidates[1].confidence >= 0.7
                ):
                    raise AmbiguousSchemaError(
                        "consumption",
                        [
                            (c.column_name, c.confidence)
                            for c in target_candidates[:3]
                        ],
                    )

                selected_target = target_candidates[0].column_name
                target_conf = target_candidates[0].confidence

            # --------------------------------------------------------
            # B2. Fallback for simple 2-column datasets
            #
            # Example:
            # timestamp | value
            #
            # If one column is timestamp and the other is numeric,
            # use the numeric column as the target.
            # --------------------------------------------------------
            else:

                remaining_numeric = [
                    c
                    for c in df.columns
                    if str(c) != str(selected_ts)
                    and pd.api.types.is_numeric_dtype(df[c])
                ]

                if len(remaining_numeric) == 1:

                    selected_target = str(remaining_numeric[0])
                    target_conf = 0.50

                    target_candidates.append(
                        ColumnCandidate(
                            column_name=selected_target,
                            confidence=target_conf,
                            reasons=[
                                "Only remaining numeric column "
                                "in dataset"
                            ],
                        )
                    )

                # ----------------------------------------------------
                # B3. No valid target found
                # ----------------------------------------------------
                else:

                    raise SchemaDetectionError(
                        f"No suitable energy consumption target column "
                        f"detected among {list(df.columns)}. "
                        "Ensure your dataset contains an energy, "
                        "consumption, load, demand, power, electricity, "
                        "kWh, kW, MW, or MWh column."
                    )

        # ============================================================
        # 3. FINAL SAFETY CHECK
        # ============================================================

        # This makes absolutely sure selected_target can never be
        # used before receiving a value.
        if selected_target is None:
            raise SchemaDetectionError(
                "Target column could not be selected after schema detection."
            )

        # ============================================================
        # 4. OPTIONAL ENVIRONMENTAL COLUMNS
        # ============================================================

        used_cols = [
            selected_ts,
            selected_target,
        ]

        optional_matches = self.detect_optional_columns(
            df,
            excluded_cols=used_cols,
        )

        # ============================================================
        # 5. BUILD DETECTED SCHEMA
        # ============================================================

        detected_schema = DetectedSchema(
            timestamp=selected_ts,
            consumption=selected_target,
            temperature=(
                optional_matches["temperature"].column_name
                if optional_matches.get("temperature")
                else None
            ),
            humidity=(
                optional_matches["humidity"].column_name
                if optional_matches.get("humidity")
                else None
            ),
            wind_speed=(
                optional_matches["wind_speed"].column_name
                if optional_matches.get("wind_speed")
                else None
            ),
            pressure=(
                optional_matches["pressure"].column_name
                if optional_matches.get("pressure")
                else None
            ),
            holiday=(
                optional_matches["holiday"].column_name
                if optional_matches.get("holiday")
                else None
            ),
        )

        # ============================================================
        # 6. RETURN RESULT
        # ============================================================

        return SchemaDetectionResult(
            detected_schema=detected_schema,
            timestamp_candidates=ts_candidates,
            target_candidates=target_candidates,
            optional_matches=optional_matches,
            confidence_summary={
                "timestamp": ts_conf,
                "consumption": target_conf,
            },
        )