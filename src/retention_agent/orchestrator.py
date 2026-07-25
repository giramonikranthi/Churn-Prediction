from __future__ import annotations

from typing import Any
from uuid import uuid4

from src.config import MLFLOW_RUN_ID, MODEL_NAME
from src.llm_tools import get_customer_model_features
from src.model_loader import load_mlflow_model
from src.retention_agent.guardrails import (
    apply_input_guardrails,
    apply_output_guardrails,
    detect_escalation,
    detect_out_of_scope,
    detect_small_talk_or_greeting,
    extract_customer_id,
    likely_retention_intent,
    normalize_customer_id,
)
from src.retention_agent.messaging import (
    optional_llm_polish,
    summarize_escalation_for_rep,
    summarize_for_rep,
)
from src.retention_agent.prediction_engine import predict_churn_step
from src.retention_agent.runtime import get_session_jsonl_file, log_step
from src.retention_agent.tool_execution import tool_call


def run_retention_orchestrator(
    user_message: str,
    selected_customer_id: str | None = None,
    prediction_model: Any | None = None,
    max_turns: int = 8,
    session_id: str | None = None,
) -> dict[str, Any]:
    _ = max_turns

    guarded_user_message = apply_input_guardrails(user_message)

    effective_session_id = session_id or f"RET-{uuid4().hex[:10].upper()}"
    tool_trace: list[dict[str, Any]] = []

    log_step(
        effective_session_id,
        "orchestrator_start",
        {
            "selected_customer_id": selected_customer_id,
            "user_message": guarded_user_message,
        },
    )

    out_of_scope, out_of_scope_message = detect_out_of_scope(guarded_user_message)
    log_step(effective_session_id, "out_of_scope_check", {"is_out_of_scope": out_of_scope})
    if out_of_scope:
        log_step(
            effective_session_id,
            "orchestrator_complete",
            {"status": "out_of_scope"},
        )
        return {
            "status": "out_of_scope",
            "assistant_message": apply_output_guardrails(out_of_scope_message),
            "tool_trace": tool_trace,
            "session_id": effective_session_id,
            "resolved_customer_id": None,
            "step_log_file": str(get_session_jsonl_file(effective_session_id)),
        }

    small_talk, small_talk_message = detect_small_talk_or_greeting(guarded_user_message)
    log_step(effective_session_id, "small_talk_check", {"is_small_talk": small_talk})
    if small_talk:
        log_step(
            effective_session_id,
            "orchestrator_complete",
            {"status": "small_talk"},
        )
        return {
            "status": "small_talk",
            "assistant_message": apply_output_guardrails(small_talk_message),
            "tool_trace": tool_trace,
            "session_id": effective_session_id,
            "resolved_customer_id": None,
            "step_log_file": str(get_session_jsonl_file(effective_session_id)),
        }

    extracted_customer_id = extract_customer_id(guarded_user_message)
    selected_id = (selected_customer_id or "").strip()
    customer_id = extracted_customer_id or selected_id
    customer_id = normalize_customer_id(customer_id) if customer_id else customer_id
    log_step(effective_session_id, "customer_id_resolved", {"customer_id": customer_id})

    if not customer_id:
        if likely_retention_intent(guarded_user_message):
            message = "I can help with that. Could you share the customer ID so I can pull up their profile?"
        else:
            message = "Sure, I can help. Share the customer details when you are ready, and we can take it step by step."
        log_step(
            effective_session_id,
            "orchestrator_complete",
            {"status": "needs_clarification", "reason": "missing_customer_id"},
        )
        return {
            "status": "needs_clarification",
            "assistant_message": apply_output_guardrails(message),
            "tool_trace": tool_trace,
            "session_id": effective_session_id,
            "resolved_customer_id": None,
            "step_log_file": str(get_session_jsonl_file(effective_session_id)),
        }

    lookup_result = tool_call(
        effective_session_id,
        tool_trace,
        "lookup_customer",
        {"customer_id": customer_id},
    )

    if lookup_result.get("status") == "error":
        message = "I hit a validation issue while retrieving customer data. Please try again."
        log_step(
            effective_session_id,
            "orchestrator_complete",
            {"status": "error", "reason": "lookup_customer_schema_validation_failed"},
        )
        return {
            "status": "error",
            "assistant_message": apply_output_guardrails(message),
            "tool_trace": tool_trace,
            "session_id": effective_session_id,
            "resolved_customer_id": customer_id,
            "step_log_file": str(get_session_jsonl_file(effective_session_id)),
        }

    if lookup_result.get("status") != "found":
        interaction_log_result = tool_call(
            effective_session_id,
            tool_trace,
            "log_interaction",
            {
                "customer_id": customer_id,
                "agent_id": "llm_retention_agent",
                "channel": "chat",
                "outcome": "no_action",
                "risk_tier": "low",
                "notes": "Customer id was not found in dataset.",
                "follow_up_required": False,
                "metadata": {
                    "session_id": effective_session_id,
                    "escalated": False,
                },
            },
        )
        message = "Customer ID not found. Please re-enter a valid customer_id from the dataset."
        log_step(
            effective_session_id,
            "orchestrator_complete",
            {"status": "not_found", "customer_id": customer_id},
        )
        return {
            "status": "not_found",
            "assistant_message": apply_output_guardrails(message),
            "tool_trace": tool_trace,
            "session_id": effective_session_id,
            "resolved_customer_id": customer_id,
            "step_log_file": str(get_session_jsonl_file(effective_session_id)),
        }

    should_escalate, escalation_category, escalation_reason = detect_escalation(guarded_user_message)
    escalation_result: dict[str, Any] | None = None
    if should_escalate:
        escalation_result = tool_call(
            effective_session_id,
            tool_trace,
            "escalate_to_supervisor",
            {
                "customer_id": customer_id,
                "reason": escalation_reason,
                "conversation_summary": guarded_user_message,
                "risk_tier": "high",
                "escalation_category": escalation_category,
                "attempted_offer_ids": [],
                "context": {
                    "session_id": effective_session_id,
                },
            },
        )

        interaction_log_result = tool_call(
            effective_session_id,
            tool_trace,
            "log_interaction",
            {
                "customer_id": customer_id,
                "agent_id": "llm_retention_agent",
                "channel": "chat",
                "outcome": "escalated",
                "risk_tier": "high",
                "notes": "Escalation-first path due to legal/compliance/dispute signal.",
                "follow_up_required": True,
                "next_action": "Supervisor handoff",
                "metadata": {
                    "session_id": effective_session_id,
                    "escalated": True,
                    "path": "escalation_first",
                },
            },
        )

        escalation_base_summary = summarize_escalation_for_rep(
            customer_id=customer_id,
            escalation_reason=escalation_reason,
            escalation_result=escalation_result,
            interaction_result=interaction_log_result,
        )
        escalation_response = apply_output_guardrails(optional_llm_polish(escalation_base_summary))

        summary = escalation_result.get("summary", {}) if isinstance(escalation_result, dict) else {}
        escalation_queue = summary.get("queue", "retention_supervisors")
        escalation_priority = summary.get("priority", "urgent")
        escalation_sla = summary.get("sla_minutes", 30)

        log_step(
            effective_session_id,
            "escalation_response_prepared",
            {
                "customer_id": customer_id,
                "reason": escalation_reason,
                "queue": escalation_queue,
                "priority": escalation_priority,
                "sla_minutes": escalation_sla,
                "response_style": "summary_prompt_polished",
                "summary_source": "tool_outputs",
                "customer_message": "Your case has been forwarded for specialized assistance.",
            },
        )

        log_step(
            effective_session_id,
            "orchestrator_complete",
            {
                "status": "ok",
                "customer_id": customer_id,
                "escalated": True,
                "tool_trace_count": len(tool_trace),
                "path": "escalation_first",
            },
        )

        return {
            "status": "ok",
            "assistant_message": escalation_response,
            "tool_trace": tool_trace,
            "session_id": effective_session_id,
            "resolved_customer_id": customer_id,
            "step_log_file": str(get_session_jsonl_file(effective_session_id)),
        }

    model_obj = prediction_model
    if model_obj is None:
        try:
            model_obj = load_mlflow_model(run_id=MLFLOW_RUN_ID, model_name=MODEL_NAME)
            log_step(effective_session_id, "mlflow_model_loaded", {"status": "ok", "source": "orchestrator"})
        except Exception as exc:
            model_obj = None
            log_step(
                effective_session_id,
                "mlflow_model_loaded",
                {"status": "fallback_to_mock", "error": str(exc)},
            )
    else:
        log_step(effective_session_id, "mlflow_model_loaded", {"status": "ok", "source": "caller"})

    model_features = lookup_result.get("model_features")
    if not isinstance(model_features, dict):
        model_features = get_customer_model_features(customer_id)

    if not isinstance(model_features, dict):
        message = "Customer profile was found, but model features are unavailable for prediction."
        log_step(
            effective_session_id,
            "orchestrator_complete",
            {"status": "error", "reason": "missing_model_features"},
        )
        return {
            "status": "error",
            "assistant_message": apply_output_guardrails(message),
            "tool_trace": tool_trace,
            "session_id": effective_session_id,
            "resolved_customer_id": customer_id,
            "step_log_file": str(get_session_jsonl_file(effective_session_id)),
        }

    prediction_result = predict_churn_step(
        effective_session_id,
        tool_trace,
        customer_id,
        model_features,
        model_obj,
    )

    if prediction_result.get("status") != "ok":
        tool_call(
            effective_session_id,
            tool_trace,
            "log_interaction",
            {
                "customer_id": customer_id,
                "agent_id": "llm_retention_agent",
                "channel": "chat",
                "outcome": "no_action",
                "risk_tier": "low",
                "notes": "Prediction step failed schema validation.",
                "follow_up_required": True,
                "next_action": "Retry prediction flow",
                "metadata": {
                    "session_id": effective_session_id,
                    "schema_validation_failed": True,
                    "step": "predict_churn",
                },
            },
        )
        message = "I could not complete prediction safely due to a validation issue. Please try again."
        log_step(
            effective_session_id,
            "orchestrator_complete",
            {"status": "error", "reason": "predict_churn_schema_validation_failed"},
        )
        return {
            "status": "error",
            "assistant_message": apply_output_guardrails(message),
            "tool_trace": tool_trace,
            "session_id": effective_session_id,
            "resolved_customer_id": customer_id,
            "step_log_file": str(get_session_jsonl_file(effective_session_id)),
        }

    prediction = prediction_result.get("prediction", {})
    risk_tier = str(prediction.get("risk_tier", "low")).lower()
    contract_type = str(((lookup_result.get("profile") or {}).get("contract") or {}).get("contract_type") or "")

    offers_result = tool_call(
        effective_session_id,
        tool_trace,
        "get_retention_offers",
        {
            "risk_tier": risk_tier,
            "contract_type": contract_type,
            "max_results": 3,
        },
    )

    offered_ids = [
        str(offer.get("offer_id"))
        for offer in (offers_result.get("offers", []) if isinstance(offers_result, dict) else [])
        if offer.get("offer_id")
    ]

    tool_call(
        effective_session_id,
        tool_trace,
        "log_interaction",
        {
            "customer_id": customer_id,
            "agent_id": "llm_retention_agent",
            "channel": "chat",
            "outcome": "pending",
            "risk_tier": risk_tier,
            "offered_offer_ids": offered_ids,
            "notes": "Deterministic orchestrator completed tool chain.",
            "follow_up_required": False,
            "next_action": "Present offers to customer",
            "metadata": {
                "session_id": effective_session_id,
                "escalated": False,
            },
        },
    )

    deterministic_response = summarize_for_rep(
        customer_id=customer_id,
        prediction_result=prediction_result,
        offers_result=offers_result,
        escalation_result=None,
    )

    final_response = apply_output_guardrails(optional_llm_polish(deterministic_response))

    log_step(
        effective_session_id,
        "orchestrator_complete",
        {
            "status": "ok",
            "customer_id": customer_id,
            "escalated": False,
            "tool_trace_count": len(tool_trace),
        },
    )

    return {
        "status": "ok",
        "assistant_message": final_response,
        "tool_trace": tool_trace,
        "session_id": effective_session_id,
        "resolved_customer_id": customer_id,
        "step_log_file": str(get_session_jsonl_file(effective_session_id)),
    }
