from __future__ import annotations

import json
import re
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:  # pragma: no cover - compatibility for older Pydantic versions
    from pydantic import BaseModel, Field

    ConfigDict = None

from src.config import OUTPUT_DIR


NON_FEATURE_COLUMNS = [
    "customer_id",
    "churned",
    "churn",
    "target",
    "label",
    "last_interaction_date",
]

TARGET_COLUMN_ALIASES = ["churned", "churn", "target", "label"]


class TopRiskFactors(BaseModel):
    """Structured schema for the explanation factors attached to a prediction."""

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:
        class Config:  # type: ignore[no-redef]
            extra = "forbid"

    for_churn: list[str] = Field(default_factory=list)
    for_not_churn: list[str] = Field(default_factory=list)


class ChurnPredictionOutput(BaseModel):
    """Structured schema for the churn prediction output sent to downstream consumers."""

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:
        class Config:  # type: ignore[no-redef]
            extra = "forbid"

    customer_id: str = Field(min_length=1)
    churn_probability: float = Field(ge=0.0, le=1.0)
    risk_tier: str = Field(pattern="^(low|medium|high)$")
    predicted_label: int = Field(ge=0, le=1)
    top_risk_factors: TopRiskFactors
    generated_at: str


def _to_serializable(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python values for JSON output."""
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _extract_probability(model_output: Any) -> float:
    """Extract churn probability from various model output shapes safely."""
    if model_output is None:
        return 0.0

    if isinstance(model_output, pd.DataFrame):
        if model_output.empty:
            return 0.0
        row = model_output.iloc[0]
        # Prefer common probability column names.
        for key in ["probability", "churn_probability", "score", "pred_proba", "proba", "1"]:
            if key in row.index:
                return float(row[key])
        # Fallback to second column if available, else first.
        if len(row.index) > 1:
            return float(row.iloc[1])
        return float(row.iloc[0])

    if isinstance(model_output, pd.Series):
        if model_output.empty:
            return 0.0
        return float(model_output.iloc[0])

    arr = np.asarray(model_output)
    if arr.size == 0:
        return 0.0

    # Common cases:
    # - (n_samples, 2): choose positive class probability at index 1
    # - (n_samples, 1) or (n_samples,): use first value
    if arr.ndim == 2:
        if arr.shape[1] > 1:
            return float(arr[0, 1])
        return float(arr[0, 0])

    return float(arr.flatten()[0])


def _compute_risk_tier(probability: float) -> str:
    if probability >= 0.70:
        return "high"
    if probability >= 0.40:
        return "medium"
    return "low"


def _prepare_model_inputs(
    input_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return direct input and deterministic encoded fallback without runtime reference fitting."""
    base_model_input = input_df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")

    fallback_input = base_model_input.copy()
    for col in fallback_input.columns:
        if pd.api.types.is_bool_dtype(fallback_input[col]):
            fallback_input[col] = fallback_input[col].astype(int)

    encoded_fallback = pd.get_dummies(fallback_input, drop_first=False, dummy_na=True, dtype=float)
    encoded_fallback = encoded_fallback.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if encoded_fallback.empty:
        encoded_fallback = pd.DataFrame(index=[0])

    return base_model_input, encoded_fallback


def _extract_expected_feature_count(error_text: str) -> int | None:
    """Parse expected feature count from model shape-mismatch error text."""
    if not error_text:
        return None

    match = re.search(r"expected:\s*(\d+)\s*,\s*got\s*(\d+)", error_text)
    if match:
        return int(match.group(1))
    return None


def _align_feature_count(model_input: pd.DataFrame | np.ndarray, expected_features: int) -> pd.DataFrame | np.ndarray:
    """Pad/truncate columns to match model expected feature count for robust single-row inference."""
    if expected_features <= 0:
        return model_input

    if isinstance(model_input, pd.DataFrame):
        current_features = model_input.shape[1]
        if current_features == expected_features:
            return model_input
        if current_features > expected_features:
            return model_input.iloc[:, :expected_features]

        padded = model_input.copy()
        for idx in range(expected_features - current_features):
            padded[f"__pad_feature_{idx}"] = 0.0
        return padded

    array_input = np.asarray(model_input)
    if array_input.ndim == 1:
        array_input = array_input.reshape(1, -1)

    current_features = array_input.shape[1]
    if current_features == expected_features:
        return array_input
    if current_features > expected_features:
        return array_input[:, :expected_features]

    missing = expected_features - current_features
    zero_pad = np.zeros((array_input.shape[0], missing), dtype=array_input.dtype)
    return np.hstack([array_input, zero_pad])


def _coerce_numeric_2d(model_input: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert inference input to numeric-only 2D numpy float array for strict XGBoost compatibility."""
    if isinstance(model_input, pd.DataFrame):
        numeric_df = model_input.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return numeric_df.to_numpy(dtype=float)

    array_input = np.asarray(model_input)
    if array_input.ndim == 1:
        array_input = array_input.reshape(1, -1)

    if array_input.dtype == object:
        numeric_df = pd.DataFrame(array_input).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return numeric_df.to_numpy(dtype=float)

    return array_input.astype(float, copy=False)


@lru_cache(maxsize=1)
def _load_reference_data() -> pd.DataFrame:
    """Load most-recent preprocessed dataset for explanation baselines."""
    project_root = Path(__file__).resolve().parents[1]
    preprocessed_dir = project_root / "data" / "Pre_processed_data"
    if not preprocessed_dir.exists():
        return pd.DataFrame()

    candidates = sorted(
        [
            *preprocessed_dir.glob("*.csv"),
            *preprocessed_dir.glob("*.xlsx"),
            *preprocessed_dir.glob("*.json"),
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return pd.DataFrame()

    candidate = candidates[0]
    try:
        if candidate.suffix.lower() == ".csv":
            return pd.read_csv(candidate)
        if candidate.suffix.lower() == ".xlsx":
            return pd.read_excel(candidate)
        if candidate.suffix.lower() == ".json":
            return pd.read_json(candidate)
    except Exception:
        return pd.DataFrame()

    return pd.DataFrame()


def _find_target_column(df: pd.DataFrame) -> str | None:
    """Find a supported binary target column in reference data."""
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for alias in TARGET_COLUMN_ALIASES:
        if alias in lowered:
            return str(lowered[alias])
    return None


def _normalize_binary_target(series: pd.Series) -> pd.Series:
    """Normalize target values to {0,1}; invalid values become NaN."""
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.where(numeric.isin([0, 1]))

    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "1": 1,
                "0": 0,
                "yes": 1,
                "no": 0,
                "true": 1,
                "false": 0,
                "y": 1,
                "n": 0,
                "churn": 1,
                "not churn": 0,
            }
        )
    )
    return pd.to_numeric(mapped, errors="coerce")


def _pretty_feature_name(name: str) -> str:
    return str(name).replace("_", " ").strip().lower()


def _safe_format_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return "unknown"
    except Exception:
        pass

    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.10g}"
    return str(value)


def _format_rate_comparison(rate: float, baseline: float) -> str:
    relation = "higher" if rate >= baseline else "lower"
    return f"churn rate {rate:.1%}, {relation} than baseline {baseline:.1%}"


def _try_build_numeric_bins(series: pd.Series, max_bins: int = 6) -> np.ndarray | None:
    """Build stable numeric bin edges; returns None when binning is not possible."""
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return None

    unique_count = int(cleaned.nunique())
    if unique_count < 2:
        return None

    bins = min(max_bins, unique_count)
    try:
        _, edges = pd.qcut(cleaned, q=bins, retbins=True, duplicates="drop")
        edges = np.unique(np.asarray(edges, dtype=float))
        if edges.size < 3:
            return None
        return edges
    except Exception:
        return None


def _summarize_risk_factors(customer_input: dict[str, Any]) -> dict[str, list[str]]:
    """Create top churn and retention factors from labeled reference data."""
    ref = _load_reference_data()
    if ref.empty:
        return {
            "for_churn": ["Reference baseline data is unavailable for factor generation."],
            "for_not_churn": ["Reference baseline data is unavailable for factor generation."],
        }

    target_col = _find_target_column(ref)
    if target_col is None:
        return {
            "for_churn": ["No target column found in reference data for churn factor analysis."],
            "for_not_churn": ["No target column found in reference data for retention factor analysis."],
        }

    ref = ref.copy()
    ref[target_col] = _normalize_binary_target(ref[target_col])
    ref = ref.dropna(subset=[target_col])
    if ref.empty or ref[target_col].nunique() < 2:
        return {
            "for_churn": ["Reference target values are not usable for churn factor analysis."],
            "for_not_churn": ["Reference target values are not usable for retention factor analysis."],
        }

    overall_rate = float(ref[target_col].mean())
    churn_items: list[tuple[float, str]] = []
    retain_items: list[tuple[float, str]] = []

    for feature, value in customer_input.items():
        if feature in NON_FEATURE_COLUMNS or feature == target_col or feature not in ref.columns:
            continue

        try:
            if pd.isna(value):
                continue
        except Exception:
            pass

        feature_name = _pretty_feature_name(feature)
        series = ref[feature]

        if pd.api.types.is_numeric_dtype(series):
            value_num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(value_num):
                continue

            edges = _try_build_numeric_bins(series)
            if edges is not None:
                rate_frame = ref[[feature, target_col]].copy()
                rate_frame[feature] = pd.to_numeric(rate_frame[feature], errors="coerce")
                rate_frame = rate_frame.dropna(subset=[feature])
                if rate_frame.empty:
                    continue

                rate_frame["_bin"] = pd.cut(rate_frame[feature], bins=edges, include_lowest=True)
                bin_rates = rate_frame.groupby("_bin", observed=True)[target_col].mean()
                customer_bin_series = pd.cut(
                    pd.Series([value_num], dtype=float),
                    bins=edges,
                    include_lowest=True,
                )
                customer_bin = customer_bin_series.iloc[0]
                if pd.isna(customer_bin) or customer_bin not in bin_rates.index:
                    continue

                bin_rate = float(bin_rates.loc[customer_bin])
                delta = bin_rate - overall_rate
                score = abs(delta)
                comparison = _format_rate_comparison(bin_rate, overall_rate)
                factor_text = (
                    f"{feature_name}={_safe_format_value(value_num)} "
                    f"(bin {customer_bin}) ({comparison})"
                )
                if delta >= 0:
                    churn_items.append((score, factor_text))
                else:
                    retain_items.append((score, factor_text))
                continue

            # Fallback when bins are unavailable: compare value-specific percentile position.
            numeric_series = pd.to_numeric(series, errors="coerce").dropna()
            if numeric_series.empty:
                continue
            quantile = float((numeric_series <= float(value_num)).mean())
            churn_rate = float(ref.loc[pd.to_numeric(ref[feature], errors="coerce") <= float(value_num), target_col].mean())
            if pd.isna(churn_rate):
                continue

            delta = churn_rate - overall_rate
            score = abs(delta)
            comparison = _format_rate_comparison(churn_rate, overall_rate)
            factor_text = (
                f"{feature_name}={_safe_format_value(value_num)} "
                f"(percentile {quantile:.1%}) ({comparison})"
            )
            if delta >= 0:
                churn_items.append((score, factor_text))
            else:
                retain_items.append((score, factor_text))
            continue

        groups = ref[[feature, target_col]].copy()
        groups[feature] = groups[feature].astype(str).str.strip()
        feature_value = str(value).strip()
        if feature_value == "":
            continue

        by_cat = groups.groupby(feature, dropna=False)[target_col].mean()
        if feature_value not in by_cat.index:
            continue

        cat_rate = float(by_cat.loc[feature_value])
        delta = cat_rate - overall_rate
        score = abs(delta)
        if score < 0.02:
            continue

        comparison = _format_rate_comparison(cat_rate, overall_rate)
        if delta >= 0:
            churn_items.append(
                (
                    score,
                    f"{feature_name}={_safe_format_value(feature_value)} ({comparison})",
                )
            )
        else:
            retain_items.append(
                (
                    score,
                    f"{feature_name}={_safe_format_value(feature_value)} ({comparison})",
                )
            )

    churn_text = [text for _, text in sorted(churn_items, key=lambda x: x[0], reverse=True)[:3]]
    retain_text = [text for _, text in sorted(retain_items, key=lambda x: x[0], reverse=True)[:3]]

    if not churn_text:
        churn_text = ["No strong churn driver was identified for this specific customer profile."]
    if not retain_text:
        retain_text = ["No strong retention driver was identified for this specific customer profile."]

    return {"for_churn": churn_text, "for_not_churn": retain_text}


def _build_prediction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the structured prediction payload via Pydantic."""
    validated = ChurnPredictionOutput(**payload)
    if hasattr(validated, "model_dump"):
        return validated.model_dump(mode="json")
    return validated.dict()


def predict_churn(
    customer_input: dict[str, Any],
    model: Any,
) -> dict[str, Any]:
    """Run single-row churn prediction and return normalized prediction payload."""
    input_row = {k: _to_serializable(v) for k, v in customer_input.items()}
    input_df = pd.DataFrame([input_row])
    model_input_df, transformed_input = _prepare_model_inputs(input_df)
    _ = model_input_df  # kept for potential diagnostics
    transformed_input = _coerce_numeric_2d(transformed_input)

    # Primary path: use encoded numeric input only.
    try:
        if hasattr(model, "predict_proba"):
            raw_output = model.predict_proba(transformed_input)
        else:
            raw_output = model.predict(transformed_input)
    except Exception as transformed_exc:
        expected_features = _extract_expected_feature_count(str(transformed_exc))
        if expected_features is not None:
            aligned_input = _align_feature_count(transformed_input, expected_features)
            aligned_input = _coerce_numeric_2d(aligned_input)
            try:
                if hasattr(model, "predict_proba"):
                    raw_output = model.predict_proba(aligned_input)
                else:
                    raw_output = model.predict(aligned_input)
            except Exception:
                raw_output = model.predict(aligned_input)
        else:
            raise RuntimeError(
                "Model prediction failed on numeric encoded input. "
                f"Original error: {transformed_exc}"
            ) from transformed_exc

    probability = _extract_probability(raw_output)
    probability = max(0.0, min(1.0, probability))
    risk_tier = _compute_risk_tier(probability)

    customer_id = str(input_row.get("customer_id", "UNKNOWN"))
    factors = _summarize_risk_factors(input_row)

    payload = {
        "customer_id": customer_id,
        "churn_probability": round(float(probability), 6),
        "risk_tier": risk_tier,
        "predicted_label": int(probability >= 0.5),
        "top_risk_factors": factors,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    return _build_prediction_payload(payload)


def save_prediction_json(prediction: dict[str, Any], customer_id: str) -> Path:
    """Persist prediction payload in configured output folder using timestamped file name."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_customer_id = str(customer_id).strip() or "UNKNOWN"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{safe_customer_id}_{timestamp}.json"
    output_path = OUTPUT_DIR / file_name

    validated_prediction = _build_prediction_payload(prediction)
    serializable_payload = {
        key: _to_serializable(value) if not isinstance(value, dict) else value
        for key, value in validated_prediction.items()
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(serializable_payload, fh, indent=2, ensure_ascii=False)

    return output_path
