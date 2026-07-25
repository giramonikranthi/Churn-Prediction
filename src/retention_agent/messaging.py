from __future__ import annotations

from typing import Any

from src.load_llm_models import load_groq_client_and_models
from src.retention_agent.runtime import load_polish_system_prompt


def _format_offer_details(offer: dict[str, Any]) -> str:
    parts: list[str] = []
    description = str(offer.get("description") or "").strip()
    lift = offer.get("expected_retention_lift_pct")
    cost = offer.get("estimated_cost_usd")

    if description:
        parts.append(description)
    if isinstance(lift, (int, float)):
        parts.append(f"expected retention lift: {lift}%")
    if isinstance(cost, (int, float)):
        parts.append(f"estimated cost: ${cost}")

    return "; ".join(parts) if parts else "offer details"


def summarize_for_rep(
    customer_id: str,
    prediction_result: dict[str, Any],
    offers_result: dict[str, Any],
    escalation_result: dict[str, Any] | None,
) -> str:
    prediction = prediction_result.get("prediction", {})
    probability = prediction.get("churn_probability")
    risk_tier = str(prediction.get("risk_tier", "unknown")).upper()
    top_factors = (prediction.get("top_risk_factors", {}) or {}).get("for_churn", [])

    offers = offers_result.get("offers", []) if isinstance(offers_result, dict) else []
    best_offer = offers[0] if offers else None
    backup_offer = offers[1] if len(offers) > 1 else None

    probability_pct = None
    if isinstance(probability, (int, float)):
        probability_pct = round(float(probability) * 100.0, 1)

    lines = [
        f"Customer {customer_id} is currently in the {risk_tier.lower()}-risk segment with a churn probability of {probability_pct}%"
        if probability_pct is not None
        else f"Customer {customer_id} is currently in the {risk_tier.lower()}-risk segment.",
        "Top risk factors: " + (", ".join(str(item) for item in top_factors[:3]) if top_factors else "N/A"),
    ]

    if best_offer:
        best_offer_id = str(best_offer.get("offer_id") or "N/A")
        best_offer_title = str(best_offer.get("title") or "Primary Offer")
        lines.append(
            "Recommendation: Present "
            f"{best_offer_title} ({best_offer_id}) as the primary offer."
        )
        lines.append("")
        lines.append("Here's a suggested script for presenting the offer:")

        details = _format_offer_details(best_offer)
        script = (
            f"Hi [Customer Name], I understand your concern about the bill. "
            f"Based on your current plan and usage, I can offer {best_offer_title} ({best_offer_id}). "
            f"It includes {details}. "
            "This can help improve value without changing your service quality. "
            "Would you like me to apply this for you now?"
        )
        lines.append(f'> "{script}"')
    else:
        lines.append("Recommendation: No matching offer was returned for this profile.")

    if backup_offer:
        backup_offer_id = str(backup_offer.get("offer_id") or "N/A")
        backup_offer_title = str(backup_offer.get("title") or "Backup Offer")
        lines.append("")
        lines.append(f"Backup offer if declined: {backup_offer_title} ({backup_offer_id}).")

    if escalation_result:
        summary = escalation_result.get("summary", {}) if isinstance(escalation_result, dict) else {}
        lines.append(
            "Escalation: queued to "
            f"{summary.get('queue', 'retention_supervisors')} "
            f"with priority {summary.get('priority', 'standard')} "
            f"(SLA {summary.get('sla_minutes', 'N/A')} minutes)."
        )
        lines.append("Next Action: inform customer that a supervisor takeover has been initiated.")
    else:
        lines.append("")
        lines.append("Next Action: Present the primary offer, then use the backup offer if needed, and capture the outcome.")

    if prediction_result.get("model_source") == "mock_fallback":
        lines.append("Prediction Source: MOCK fallback used because trained model was unavailable.")

    return "\n".join(lines)


def summarize_escalation_for_rep(
    customer_id: str,
    escalation_reason: str,
    escalation_result: dict[str, Any] | None,
    interaction_result: dict[str, Any] | None,
) -> str:
    escalation_summary = escalation_result.get("summary", {}) if isinstance(escalation_result, dict) else {}
    interaction_summary = interaction_result.get("summary", {}) if isinstance(interaction_result, dict) else {}

    lines = [
        f"Customer ID: {customer_id}",
        "Case Status: Escalation required.",
        f"Reason: {escalation_reason}",
        "Action taken: Escalated to supervisor with conversation context summary.",
        f"Queue: {escalation_summary.get('queue', 'retention_supervisors')}",
        f"Priority: {escalation_summary.get('priority', 'urgent')}",
        f"SLA (minutes): {escalation_summary.get('sla_minutes', 30)}",
        "Customer message: Your case has been forwarded for specialized assistance.",
    ]

    outcome = interaction_summary.get("outcome")
    follow_up_required = interaction_summary.get("follow_up_required")
    if outcome is not None:
        lines.append(f"Interaction Log Outcome: {outcome}")
    if follow_up_required is not None:
        lines.append(f"Follow-up Required: {follow_up_required}")

    lines.append("Next Action: Supervisor handoff in progress. Do not provide further retention commitments.")

    return "\n".join(lines)


def optional_llm_polish(base_text: str) -> str:
    try:
        client, model_predict, _ = load_groq_client_and_models()
        system_prompt = load_polish_system_prompt()
        response = client.chat.completions.create(
            model=model_predict,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": base_text},
            ],
            temperature=0.1,
        )
        polished = response.choices[0].message.content or ""
        return polished.strip() or base_text
    except Exception:
        return base_text
