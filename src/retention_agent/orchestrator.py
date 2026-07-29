from __future__ import annotations

from typing import Any, TypedDict
from uuid import uuid4

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - fallback when dependency is unavailable
    END = None
    StateGraph = None

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


class OrchestratorState(TypedDict, total=False):
    user_message: str
    selected_customer_id: str | None
    prediction_model: Any | None
    session_id: str
    guarded_user_message: str
    tool_trace: list[dict[str, Any]]
    status: str
    assistant_message: str
    resolved_customer_id: str | None
    step_log_file: str | None
    customer_id: str | None
    lookup_result: dict[str, Any] | None
    should_escalate: bool
    escalation_category: str | None
    escalation_reason: str | None
    escalation_result: dict[str, Any] | None
    interaction_log_result: dict[str, Any] | None
    model_obj: Any | None
    model_features: dict[str, Any] | None
    prediction_result: dict[str, Any] | None
    offers_result: dict[str, Any] | None
    risk_tier: str
    contract_type: str


def _initialize_state(state: OrchestratorState) -> OrchestratorState:
    effective_session_id = state.get("session_id") or f"RET-{uuid4().hex[:10].upper()}"
    state["session_id"] = effective_session_id
    state["tool_trace"] = []
    state["guarded_user_message"] = apply_input_guardrails(state["user_message"])
    state["status"] = "ready"
    log_step(
        effective_session_id,
        "orchestrator_start",
        {
            "selected_customer_id": state.get("selected_customer_id"),
            "user_message": state["guarded_user_message"],
        },
    )
    return state


def _evaluate_request(state: OrchestratorState) -> OrchestratorState:
    guarded_user_message = state["guarded_user_message"]
    out_of_scope, out_of_scope_message = detect_out_of_scope(guarded_user_message)
    log_step(state["session_id"], "out_of_scope_check", {"is_out_of_scope": out_of_scope})
    if out_of_scope:
        state["status"] = "out_of_scope"
        state["assistant_message"] = apply_output_guardrails(out_of_scope_message)
        state["resolved_customer_id"] = None
        state["step_log_file"] = str(get_session_jsonl_file(state["session_id"]))
        return state

    small_talk, small_talk_message = detect_small_talk_or_greeting(guarded_user_message)
    log_step(state["session_id"], "small_talk_check", {"is_small_talk": small_talk})
    if small_talk:
        state["status"] = "small_talk"
        state["assistant_message"] = apply_output_guardrails(small_talk_message)
        state["resolved_customer_id"] = None
        state["step_log_file"] = str(get_session_jsonl_file(state["session_id"]))
        return state

    state["status"] = "ready"
    return state


def _resolve_customer(state: OrchestratorState) -> OrchestratorState:
    guarded_user_message = state["guarded_user_message"]
    extracted_customer_id = extract_customer_id(guarded_user_message)
    selected_id = (state.get("selected_customer_id") or "").strip()
    customer_id = extracted_customer_id or selected_id
    customer_id = normalize_customer_id(customer_id) if customer_id else customer_id
    log_step(state["session_id"], "customer_id_resolved", {"customer_id": customer_id})

    if not customer_id:
        if likely_retention_intent(guarded_user_message):
            message = "I can help with that. Could you share the customer ID so I can pull up their profile?"
        else:
            message = "Sure, I can help. Share the customer details when you are ready, and we can take it step by step."
        state["status"] = "needs_clarification"
        state["assistant_message"] = apply_output_guardrails(message)
        state["resolved_customer_id"] = None
        state["step_log_file"] = str(get_session_jsonl_file(state["session_id"]))
        return state

    state["customer_id"] = customer_id
    lookup_result = tool_call(
        state["session_id"],
        state["tool_trace"],
        "lookup_customer",
        {"customer_id": customer_id},
    )
    state["lookup_result"] = lookup_result

    if lookup_result.get("status") == "error":
        state["status"] = "error"
        state["assistant_message"] = apply_output_guardrails(
            "I hit a validation issue while retrieving customer data. Please try again."
        )
        state["resolved_customer_id"] = customer_id
        state["step_log_file"] = str(get_session_jsonl_file(state["session_id"]))
        return state

    if lookup_result.get("status") != "found":
        tool_call(
            state["session_id"],
            state["tool_trace"],
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
                    "session_id": state["session_id"],
                    "escalated": False,
                },
            },
        )
        state["status"] = "not_found"
        state["assistant_message"] = apply_output_guardrails(
            "Customer ID not found. Please re-enter a valid customer_id from the dataset."
        )
        state["resolved_customer_id"] = customer_id
        state["step_log_file"] = str(get_session_jsonl_file(state["session_id"]))
        return state

    state["status"] = "ready"
    return state


def _route_after_evaluate_request(state: OrchestratorState) -> str:
    return "finalize" if state.get("status") in {"out_of_scope", "small_talk"} else "resolve_customer"


def _route_after_resolve_customer(state: OrchestratorState) -> str:
    return "finalize" if state.get("status") in {"needs_clarification", "not_found", "error"} else "handle_escalation"


def _handle_escalation(state: OrchestratorState) -> OrchestratorState:
    should_escalate, escalation_category, escalation_reason = detect_escalation(state["guarded_user_message"])
    state["should_escalate"] = should_escalate
    state["escalation_category"] = escalation_category
    state["escalation_reason"] = escalation_reason

    if should_escalate:
        escalation_result = tool_call(
            state["session_id"],
            state["tool_trace"],
            "escalate_to_supervisor",
            {
                "customer_id": state["customer_id"],
                "reason": escalation_reason,
                "conversation_summary": state["guarded_user_message"],
                "risk_tier": "high",
                "escalation_category": escalation_category,
                "attempted_offer_ids": [],
                "context": {"session_id": state["session_id"]},
            },
        )
        state["escalation_result"] = escalation_result

        interaction_log_result = tool_call(
            state["session_id"],
            state["tool_trace"],
            "log_interaction",
            {
                "customer_id": state["customer_id"],
                "agent_id": "llm_retention_agent",
                "channel": "chat",
                "outcome": "escalated",
                "risk_tier": "high",
                "notes": "Escalation-first path due to legal/compliance/dispute signal.",
                "follow_up_required": True,
                "next_action": "Supervisor handoff",
                "metadata": {
                    "session_id": state["session_id"],
                    "escalated": True,
                    "path": "escalation_first",
                },
            },
        )
        state["interaction_log_result"] = interaction_log_result

        escalation_base_summary = summarize_escalation_for_rep(
            customer_id=state["customer_id"],
            escalation_reason=escalation_reason,
            escalation_result=escalation_result,
            interaction_result=interaction_log_result,
        )
        state["assistant_message"] = apply_output_guardrails(optional_llm_polish(escalation_base_summary))

        summary = escalation_result.get("summary", {}) if isinstance(escalation_result, dict) else {}
        escalation_queue = summary.get("queue", "retention_supervisors")
        escalation_priority = summary.get("priority", "urgent")
        escalation_sla = summary.get("sla_minutes", 30)

        log_step(
            state["session_id"],
            "escalation_response_prepared",
            {
                "customer_id": state["customer_id"],
                "reason": escalation_reason,
                "queue": escalation_queue,
                "priority": escalation_priority,
                "sla_minutes": escalation_sla,
                "response_style": "summary_prompt_polished",
                "summary_source": "tool_outputs",
                "customer_message": "Your case has been forwarded for specialized assistance.",
            },
        )
        state["status"] = "ok"
        state["resolved_customer_id"] = state["customer_id"]
        state["step_log_file"] = str(get_session_jsonl_file(state["session_id"]))
        return state

    state["status"] = "ready"
    return state


def _route_after_handle_escalation(state: OrchestratorState) -> str:
    return "finalize" if state.get("status") == "ok" else "predict_and_offer"


def _predict_and_offer(state: OrchestratorState) -> OrchestratorState:
    customer_id = state["customer_id"]
    model_obj = state.get("prediction_model")
    if model_obj is None:
        try:
            model_obj = load_mlflow_model(run_id=MLFLOW_RUN_ID, model_name=MODEL_NAME)
            log_step(state["session_id"], "mlflow_model_loaded", {"status": "ok", "source": "orchestrator"})
        except Exception as exc:
            model_obj = None
            log_step(
                state["session_id"],
                "mlflow_model_loaded",
                {"status": "fallback_to_mock", "error": str(exc)},
            )
    else:
        log_step(state["session_id"], "mlflow_model_loaded", {"status": "ok", "source": "caller"})

    state["model_obj"] = model_obj
    model_features = state["lookup_result"].get("model_features") if state["lookup_result"] else None
    if not isinstance(model_features, dict):
        model_features = get_customer_model_features(customer_id)

    if not isinstance(model_features, dict):
        state["status"] = "error"
        state["assistant_message"] = apply_output_guardrails(
            "Customer profile was found, but model features are unavailable for prediction."
        )
        state["resolved_customer_id"] = customer_id
        state["step_log_file"] = str(get_session_jsonl_file(state["session_id"]))
        return state

    state["model_features"] = model_features
    prediction_result = predict_churn_step(
        state["session_id"],
        state["tool_trace"],
        customer_id,
        model_features,
        model_obj,
    )
    state["prediction_result"] = prediction_result

    if prediction_result.get("status") != "ok":
        tool_call(
            state["session_id"],
            state["tool_trace"],
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
                    "session_id": state["session_id"],
                    "schema_validation_failed": True,
                    "step": "predict_churn",
                },
            },
        )
        state["status"] = "error"
        state["assistant_message"] = apply_output_guardrails(
            "I could not complete prediction safely due to a validation issue. Please try again."
        )
        state["resolved_customer_id"] = customer_id
        state["step_log_file"] = str(get_session_jsonl_file(state["session_id"]))
        return state

    prediction = prediction_result.get("prediction", {})
    risk_tier = str(prediction.get("risk_tier", "low")).lower()
    contract_type = str(((state["lookup_result"].get("profile") or {}).get("contract") or {}).get("contract_type") or "")
    state["risk_tier"] = risk_tier
    state["contract_type"] = contract_type

    offers_result = tool_call(
        state["session_id"],
        state["tool_trace"],
        "get_retention_offers",
        {
            "risk_tier": risk_tier,
            "contract_type": contract_type,
            "max_results": 3,
        },
    )
    state["offers_result"] = offers_result

    offered_ids = [
        str(offer.get("offer_id"))
        for offer in (offers_result.get("offers", []) if isinstance(offers_result, dict) else [])
        if offer.get("offer_id")
    ]

    tool_call(
        state["session_id"],
        state["tool_trace"],
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
                "session_id": state["session_id"],
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
    state["assistant_message"] = apply_output_guardrails(optional_llm_polish(deterministic_response))
    state["status"] = "ok"
    state["resolved_customer_id"] = customer_id
    state["step_log_file"] = str(get_session_jsonl_file(state["session_id"]))
    return state


def _finalize(state: OrchestratorState) -> OrchestratorState:
    complete_payload: dict[str, Any] = {
        "status": state.get("status", "ok"),
        "customer_id": state.get("customer_id"),
        "escalated": state.get("status") == "ok" and state.get("should_escalate") is True,
        "tool_trace_count": len(state.get("tool_trace", [])),
    }
    if state.get("status") == "ok" and state.get("should_escalate") is True:
        complete_payload["path"] = "escalation_first"
    log_step(state["session_id"], "orchestrator_complete", complete_payload)
    return state


def _build_graph():
    if StateGraph is None:
        raise ImportError("langgraph is required to run the graph-based orchestrator")
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("initialize", _initialize_state)
    workflow.add_node("evaluate_request", _evaluate_request)
    workflow.add_node("resolve_customer", _resolve_customer)
    workflow.add_node("handle_escalation", _handle_escalation)
    workflow.add_node("predict_and_offer", _predict_and_offer)
    workflow.add_node("finalize", _finalize)

    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "evaluate_request")
    workflow.add_conditional_edges(
        "evaluate_request",
        _route_after_evaluate_request,
        {"finalize": "finalize", "resolve_customer": "resolve_customer"},
    )
    workflow.add_conditional_edges(
        "resolve_customer",
        _route_after_resolve_customer,
        {"finalize": "finalize", "handle_escalation": "handle_escalation"},
    )
    workflow.add_conditional_edges(
        "handle_escalation",
        _route_after_handle_escalation,
        {"finalize": "finalize", "predict_and_offer": "predict_and_offer"},
    )
    workflow.add_edge("predict_and_offer", "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile()


def run_retention_orchestrator(
    user_message: str,
    selected_customer_id: str | None = None,
    prediction_model: Any | None = None,
    max_turns: int = 8,
    session_id: str | None = None,
) -> dict[str, Any]:
    _ = max_turns
    initial_state: OrchestratorState = {
        "user_message": user_message,
        "selected_customer_id": selected_customer_id,
        "prediction_model": prediction_model,
        "session_id": session_id,
    }

    graph = _build_graph()
    result = graph.invoke(initial_state)
    return {
        "status": result.get("status", "ok"),
        "assistant_message": result.get("assistant_message", ""),
        "tool_trace": result.get("tool_trace", []),
        "session_id": result.get("session_id"),
        "resolved_customer_id": result.get("resolved_customer_id"),
        "step_log_file": result.get("step_log_file"),
    }
