from __future__ import annotations

from typing import Any

from src.predictor import predict_churn
from src.retention_agent.runtime import now_utc_str, log_step
from src.retention_agent.schema_guardrails import PredictChurnStepResult, validate_schema


def compute_risk_tier(probability: float) -> str:
    if probability >= 0.70:
        return "high"
    if probability >= 0.40:
        return "medium"
    return "low"


def mock_predict_churn(customer_id: str, model_features: dict[str, Any]) -> dict[str, Any]:
    contract = str(model_features.get("contract_type", "")).strip().lower()
    tenure = float(model_features.get("tenure_months", 0) or 0)
    satisfaction = float(model_features.get("satisfaction_score", 5) or 5)
    tickets = float(model_features.get("num_support_tickets", 0) or 0)

    score = 0.35
    if "month" in contract:
        score += 0.22
    if tenure < 12:
        score += 0.12
    if satisfaction < 5:
        score += 0.16
    if tickets >= 3:
        score += 0.10

    probability = max(0.0, min(0.98, round(score, 6)))
    risk_tier = compute_risk_tier(probability)

    factors: list[str] = []
    if "month" in contract:
        factors.append("Contract type=Month-to-month")
    if tenure < 12:
        factors.append(f"tenure_months={tenure}")
    if satisfaction < 5:
        factors.append(f"satisfaction_score={satisfaction}")
    if tickets >= 3:
        factors.append(f"num_support_tickets={tickets}")

    return {
        "status": "ok",
        "prediction": {
            "customer_id": customer_id,
            "churn_probability": probability,
            "risk_tier": risk_tier,
            "top_risk_factors": {
                "for_churn": factors[:3],
                "for_not_churn": [],
            },
        },
        "raw_prediction": {
            "customer_id": customer_id,
            "churn_probability": probability,
            "risk_tier": risk_tier,
            "top_risk_factors": {
                "for_churn": factors[:3],
                "for_not_churn": [],
            },
            "generated_at": now_utc_str(),
        },
        "model_source": "mock_fallback",
        "note": "Using mock prediction because trained model was unavailable.",
    }


def predict_churn_step(
    session_id: str,
    tool_trace: list[dict[str, Any]],
    customer_id: str,
    model_features: dict[str, Any],
    prediction_model: Any | None,
) -> dict[str, Any]:
    log_step(
        session_id,
        "tool_call_requested",
        {
            "tool": "predict_churn",
            "arguments": {
                "customer_id": customer_id,
                "model_features_keys": sorted(model_features.keys()),
            },
        },
    )

    if prediction_model is None:
        result = mock_predict_churn(customer_id, model_features)
    else:
        try:
            payload = predict_churn(model_features, model=prediction_model)
            result = {
                "status": "ok",
                "prediction": {
                    "customer_id": payload.get("customer_id", customer_id),
                    "churn_probability": payload.get("churn_probability"),
                    "risk_tier": payload.get("risk_tier"),
                    "top_risk_factors": payload.get("top_risk_factors", {}),
                },
                "raw_prediction": payload,
                "model_source": "trained_model",
            }
        except Exception as exc:
            result = mock_predict_churn(customer_id, model_features)
            result["error"] = f"Trained model inference failed: {exc}"

    if isinstance(result, dict):
        ok, normalized_result, validation_error = validate_schema(result, PredictChurnStepResult)
        if ok:
            result = normalized_result
        else:
            result = {
                "status": "error",
                "error": f"Result schema validation failed for predict_churn: {validation_error}",
                "raw_result": result,
            }

    log_step(
        session_id,
        "tool_call_completed",
        {
            "tool": "predict_churn",
            "status": result.get("status", "unknown"),
            "model_source": result.get("model_source"),
        },
    )

    tool_trace.append(
        {
            "tool": "predict_churn",
            "arguments": {
                "customer_id": customer_id,
                "model_features": model_features,
            },
            "result": result,
        }
    )
    return result
