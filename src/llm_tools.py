"""Compatibility facade for retention tooling.

This module re-exports tool functions and schemas from smaller files under src/tools/
so existing imports keep working while code stays modular and easier to monitor.
"""

from src.tools.dispatch import (
    LLM_TOOL_SCHEMAS,
    OFFER_CATALOG,
    PREDICT_CHURN_TOOL_SCHEMA,
    dispatch_tool_call,
    dispatch_tool_call_from_json,
    escalate_to_supervisor,
    get_customer_model_features,
    get_llm_tool_schemas,
    get_retention_offers,
    log_interaction,
    lookup_customer,
)

__all__ = [
    "LLM_TOOL_SCHEMAS",
    "OFFER_CATALOG",
    "PREDICT_CHURN_TOOL_SCHEMA",
    "lookup_customer",
    "get_customer_model_features",
    "get_retention_offers",
    "log_interaction",
    "escalate_to_supervisor",
    "get_llm_tool_schemas",
    "dispatch_tool_call",
    "dispatch_tool_call_from_json",
]
