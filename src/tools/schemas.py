from __future__ import annotations

from typing import Any


LLM_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": "Retrieve a customer profile by ID from local input dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Unique customer id, for example TC-000178.",
                    }
                },
                "required": ["customer_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_retention_offers",
            "description": "Return retention offers filtered by risk tier and contract type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_tier": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "contract_type": {
                        "type": "string",
                        "description": "Contract type such as month-to-month, one year, two year.",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3,
                    },
                },
                "required": ["risk_tier", "contract_type"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_interaction",
            "description": "Record retention conversation outcome in an audit-friendly schema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "channel": {
                        "type": "string",
                        "description": "Call channel, for example phone, chat, email.",
                    },
                    "outcome": {
                        "type": "string",
                        "enum": [
                            "retained",
                            "callback_scheduled",
                            "declined_offer",
                            "escalated",
                            "no_action",
                            "pending",
                        ],
                        "description": "Production-like conversation disposition.",
                    },
                    "risk_tier": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "offer_id": {"type": ["string", "null"]},
                    "offered_offer_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "accepted_offer_id": {"type": ["string", "null"]},
                    "declined_reason": {"type": ["string", "null"]},
                    "conversation_start_utc": {
                        "type": ["string", "null"],
                        "description": "ISO-8601 start timestamp.",
                    },
                    "conversation_end_utc": {
                        "type": ["string", "null"],
                        "description": "ISO-8601 end timestamp.",
                    },
                    "sentiment": {
                        "type": ["string", "null"],
                        "description": "Customer sentiment label such as positive/neutral/negative.",
                    },
                    "next_action": {"type": ["string", "null"]},
                    "notes": {"type": ["string", "null"]},
                    "follow_up_required": {"type": "boolean", "default": False},
                    "metadata": {
                        "type": "object",
                        "description": "Additional non-sensitive event metadata.",
                        "default": {},
                    },
                },
                "required": [
                    "customer_id",
                    "agent_id",
                    "channel",
                    "outcome",
                    "risk_tier",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_supervisor",
            "description": "Escalate customer case with context summary to a human supervisor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "conversation_summary": {"type": "string"},
                    "risk_tier": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "escalation_category": {
                        "type": ["string", "null"],
                        "description": (
                            "Reason category for handoff, e.g. customer_request, compliance, "
                            "billing_dispute, legal, safety, abuse, technical_blocker, policy_exception."
                        ),
                    },
                    "supervisor_queue": {
                        "type": ["string", "null"],
                        "description": "Target queue name for supervisor routing.",
                    },
                    "requested_by": {
                        "type": ["string", "null"],
                        "description": "Actor initiating escalation, e.g. llm_retention_agent.",
                    },
                    "user_visible_message": {
                        "type": ["string", "null"],
                        "description": "Message safe to display to end customer during handoff.",
                    },
                    "incident_reference": {
                        "type": ["string", "null"],
                        "description": "Optional external incident/ticket id.",
                    },
                    "attempted_offer_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "context": {
                        "type": "object",
                        "description": "Non-sensitive structured context to aid supervisor handoff.",
                        "default": {},
                    },
                },
                "required": [
                    "customer_id",
                    "reason",
                    "conversation_summary",
                    "risk_tier",
                ],
                "additionalProperties": False,
            },
        },
    },
]

PREDICT_CHURN_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "predict_churn",
        "description": (
            "Predict churn from customer model features. "
            "Use model_features from lookup_customer when available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Optional customer id for traceability and fallback lookup.",
                },
                "model_features": {
                    "type": "object",
                    "description": "Model input feature map, typically from lookup_customer.model_features.",
                },
            },
            "additionalProperties": False,
        },
    },
}


def get_llm_tool_schemas(include_predict_churn: bool = False) -> list[dict[str, Any]]:
    if include_predict_churn:
        return [*LLM_TOOL_SCHEMAS, PREDICT_CHURN_TOOL_SCHEMA]
    return list(LLM_TOOL_SCHEMAS)
