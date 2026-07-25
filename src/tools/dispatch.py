from __future__ import annotations

import json
from typing import Any, Callable

from src.tools.customer_tools import get_customer_model_features, lookup_customer
from src.tools.engagement_tools import escalate_to_supervisor, log_interaction
from src.tools.offer_tools import OFFER_CATALOG, get_retention_offers
from src.tools.schemas import LLM_TOOL_SCHEMAS, PREDICT_CHURN_TOOL_SCHEMA, get_llm_tool_schemas


TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "lookup_customer": lookup_customer,
    "get_retention_offers": get_retention_offers,
    "log_interaction": log_interaction,
    "escalate_to_supervisor": escalate_to_supervisor,
}


def dispatch_tool_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    func = TOOL_REGISTRY.get(tool_name)
    if func is None:
        return {
            "status": "error",
            "error": f"Unknown tool: {tool_name}",
        }

    try:
        return func(**arguments)
    except TypeError as exc:
        return {
            "status": "error",
            "error": f"Invalid arguments for {tool_name}: {exc}",
        }
    except Exception as exc:  # pragma: no cover
        return {
            "status": "error",
            "error": f"Tool {tool_name} failed: {exc}",
        }


def dispatch_tool_call_from_json(tool_name: str, arguments_json: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as exc:
        return {
            "status": "error",
            "error": f"Invalid JSON arguments for {tool_name}: {exc}",
        }

    if not isinstance(parsed, dict):
        return {
            "status": "error",
            "error": f"JSON arguments for {tool_name} must decode to an object.",
        }

    return dispatch_tool_call(tool_name, parsed)


__all__ = [
    "OFFER_CATALOG",
    "LLM_TOOL_SCHEMAS",
    "PREDICT_CHURN_TOOL_SCHEMA",
    "get_customer_model_features",
    "lookup_customer",
    "get_retention_offers",
    "log_interaction",
    "escalate_to_supervisor",
    "get_llm_tool_schemas",
    "dispatch_tool_call",
    "dispatch_tool_call_from_json",
]
