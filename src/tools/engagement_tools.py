from __future__ import annotations

from typing import Any
from uuid import uuid4

from src.tools.common import (
    OUTPUT_DATA_DIR,
    append_jsonl,
    build_process_meta,
    normalize_lower,
    normalize_text,
    now_utc_iso,
    parse_iso_datetime,
)


def _coerce_outcome(value: str) -> str:
    normalized = normalize_lower(value)
    aliases = {
        "retained": "retained",
        "saved": "retained",
        "callback": "callback_scheduled",
        "callback_scheduled": "callback_scheduled",
        "declined": "declined_offer",
        "declined_offer": "declined_offer",
        "escalated": "escalated",
        "no_action": "no_action",
        "pending": "pending",
    }
    return aliases.get(normalized, "pending")


def _coerce_escalation_category(value: str | None) -> str:
    normalized = normalize_lower(value)
    allowed = {
        "customer_request",
        "compliance",
        "fraud",
        "identity_theft",
        "billing_dispute",
        "legal",
        "safety",
        "abuse",
        "technical_blocker",
        "policy_exception",
        "high_risk_retention",
        "other",
    }
    return normalized if normalized in allowed else "other"


def _derive_escalation_priority(risk_tier: str, category: str) -> tuple[str, int]:
    urgent_categories = {"safety", "legal", "compliance", "abuse"}
    if category in urgent_categories:
        return "urgent", 15

    if normalize_lower(risk_tier) == "high":
        return "urgent", 30
    if normalize_lower(risk_tier) == "medium":
        return "high", 60
    return "standard", 120


def log_interaction(
    customer_id: str,
    agent_id: str,
    channel: str,
    outcome: str,
    risk_tier: str,
    offer_id: str | None = None,
    notes: str | None = None,
    follow_up_required: bool = False,
    offered_offer_ids: list[str] | None = None,
    accepted_offer_id: str | None = None,
    declined_reason: str | None = None,
    conversation_start_utc: str | None = None,
    conversation_end_utc: str | None = None,
    sentiment: str | None = None,
    next_action: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start_dt = parse_iso_datetime(conversation_start_utc)
    end_dt = parse_iso_datetime(conversation_end_utc)
    duration_seconds = None
    if start_dt and end_dt and end_dt >= start_dt:
        duration_seconds = int((end_dt - start_dt).total_seconds())

    normalized_outcome = _coerce_outcome(outcome)
    offer_list = [str(item).strip() for item in (offered_offer_ids or []) if str(item).strip()]

    primary_offer = normalize_text(accepted_offer_id) or normalize_text(offer_id) or None
    if primary_offer and primary_offer not in offer_list:
        offer_list.append(primary_offer)

    record = {
        "interaction_id": f"INT-{uuid4().hex[:12].upper()}",
        "event_time": now_utc_iso(),
        "customer_id": normalize_text(customer_id),
        "agent": {
            "agent_id": normalize_text(agent_id),
            "agent_type": "llm_retention_agent",
        },
        "channel": normalize_lower(channel) or "unknown",
        "outcome": normalized_outcome,
        "risk_tier": normalize_lower(risk_tier) or "unknown",
        "conversation": {
            "start_time": start_dt.isoformat() if start_dt else None,
            "end_time": end_dt.isoformat() if end_dt else None,
            "duration_seconds": duration_seconds,
            "sentiment": normalize_lower(sentiment) or "unknown",
        },
        "offers": {
            "offered_offer_ids": offer_list,
            "accepted_offer_id": primary_offer,
            "declined_reason": normalize_text(declined_reason) or None,
        },
        "notes": normalize_text(notes) or None,
        "follow_up": {
            "required": bool(follow_up_required),
            "next_action": normalize_text(next_action) or None,
        },
        "compliance": {
            "contains_sensitive_data": False,
            "consent_captured": True,
            "pii_redacted": True,
            "policy_version": "retention_policy_v1",
        },
        "audit": {
            "schema_version": "1.0.0",
            "recorded_by": "retention_orchestrator",
            "recorded_at": now_utc_iso(),
        },
        "metadata": metadata if isinstance(metadata, dict) else {},
    }

    session_id = ""
    if isinstance(metadata, dict):
        session_id = normalize_text(metadata.get("session_id"))

    session_output_file = OUTPUT_DATA_DIR / "retention_interactions.jsonl"
    if session_id:
        session_output_file = OUTPUT_DATA_DIR / f"{session_id}_interactions.jsonl"

    append_jsonl(session_output_file, record)

    return {
        "status": "logged",
        "interaction": record,
        "summary": {
            "customer_id": record["customer_id"],
            "outcome": record["outcome"],
            "accepted_offer_id": record["offers"]["accepted_offer_id"],
            "follow_up_required": record["follow_up"]["required"],
        },
        "log_file": str(session_output_file),
        "session_output_file": str(session_output_file),
        "meta": build_process_meta(
            tool_name="log_interaction",
            source="jsonl",
            flags={
                "logged": True,
                "follow_up_required": record["follow_up"]["required"],
                "has_offers": len(record["offers"]["offered_offer_ids"]) > 0,
                "has_session_id": bool(session_id),
            },
        ),
    }


def escalate_to_supervisor(
    customer_id: str,
    reason: str,
    conversation_summary: str,
    risk_tier: str,
    attempted_offer_ids: list[str] | None = None,
    escalation_category: str | None = None,
    supervisor_queue: str | None = None,
    requested_by: str | None = None,
    user_visible_message: str | None = None,
    incident_reference: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    category = _coerce_escalation_category(escalation_category)
    priority, sla_minutes = _derive_escalation_priority(risk_tier, category)
    queue_name = normalize_text(supervisor_queue) or "retention_supervisors"

    attempted = [str(item).strip() for item in (attempted_offer_ids or []) if str(item).strip()]

    escalation = {
        "escalation_id": f"ESC-{uuid4().hex[:12].upper()}",
        "created_at": now_utc_iso(),
        "customer_id": normalize_text(customer_id),
        "risk_tier": normalize_lower(risk_tier) or "unknown",
        "category": category,
        "reason": normalize_text(reason),
        "conversation_summary": normalize_text(conversation_summary),
        "attempted_offer_ids": attempted,
        "request": {
            "requested_by": normalize_text(requested_by) or "llm_retention_agent",
            "user_visible_message": normalize_text(user_visible_message) or None,
            "incident_reference": normalize_text(incident_reference) or None,
        },
        "routing": {
            "target_queue": queue_name,
            "priority": priority,
            "sla_minutes": sla_minutes,
            "status": "queued",
        },
        "handoff_packet": {
            "customer_context_included": True,
            "model_decision_context_included": True,
            "offer_history_included": bool(attempted),
            "summary_word_count": len(normalize_text(conversation_summary).split()),
        },
        "audit": {
            "schema_version": "1.0.0",
            "recorded_at": now_utc_iso(),
            "recorded_by": "retention_orchestrator",
        },
        "context": context if isinstance(context, dict) else {},
    }

    session_id = ""
    if isinstance(context, dict):
        session_id = normalize_text(context.get("session_id"))

    session_output_file = OUTPUT_DATA_DIR / "retention_escalations.jsonl"
    if session_id:
        session_output_file = OUTPUT_DATA_DIR / f"{session_id}_escalations.jsonl"

    append_jsonl(session_output_file, escalation)

    return {
        "status": "escalated",
        "escalation": escalation,
        "summary": {
            "escalation_id": escalation["escalation_id"],
            "customer_id": escalation["customer_id"],
            "queue": escalation["routing"]["target_queue"],
            "priority": escalation["routing"]["priority"],
            "sla_minutes": escalation["routing"]["sla_minutes"],
        },
        "log_file": str(session_output_file),
        "session_output_file": str(session_output_file),
        "meta": build_process_meta(
            tool_name="escalate_to_supervisor",
            source="jsonl",
            flags={
                "escalated": True,
                "has_session_id": bool(session_id),
                "priority": escalation["routing"]["priority"],
                "queue_assigned": bool(escalation["routing"]["target_queue"]),
            },
        ),
    }
