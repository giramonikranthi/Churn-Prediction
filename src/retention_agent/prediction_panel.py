import streamlit as st


def _latest_tool_result(tool_trace: list[dict], tool_name: str) -> dict | None:
    for event in reversed(tool_trace or []):
        if event.get("tool") == tool_name and isinstance(event.get("result"), dict):
            return event.get("result")
    return None


def render_prediction_panel(tool_trace: list[dict]) -> None:
    prediction_result = _latest_tool_result(tool_trace, "predict_churn")
    if not prediction_result or prediction_result.get("status") != "ok":
        return

    prediction = prediction_result.get("prediction", {})
    factors = (prediction.get("top_risk_factors") or {}).get("for_churn", [])

    st.subheader("Churn Prediction")
    st.write(f"Probability: {prediction.get('churn_probability')}")
    st.write(f"Risk Tier: {str(prediction.get('risk_tier', 'unknown')).upper()}")
    st.write("Top Risk Factors:")
    if factors:
        for item in factors[:3]:
            st.write(f"- {item}")
    else:
        st.write("- N/A")
