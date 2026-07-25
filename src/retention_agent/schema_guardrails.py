from __future__ import annotations

from typing import Any, Literal

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, Field

    ConfigDict = None


class _ArgsBaseModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:
        class Config:  # type: ignore[no-redef]
            extra = "forbid"


class _ResultBaseModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="allow")
    else:
        class Config:  # type: ignore[no-redef]
            extra = "allow"


class LookupCustomerArgs(_ArgsBaseModel):
    customer_id: str = Field(min_length=1)


class LookupCustomerResult(_ResultBaseModel):
    status: Literal["found", "not_found"]


class GetRetentionOffersArgs(_ArgsBaseModel):
    risk_tier: Literal["low", "medium", "high"]
    contract_type: str = Field(min_length=1)
    max_results: int = Field(default=3, ge=1, le=10)


class GetRetentionOffersResult(_ResultBaseModel):
    risk_tier: str
    contract_type: str
    offers: list[dict[str, Any]]


class LogInteractionArgs(_ArgsBaseModel):
    customer_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    channel: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    risk_tier: str = Field(min_length=1)
    offer_id: str | None = None
    offered_offer_ids: list[str] = Field(default_factory=list)
    accepted_offer_id: str | None = None
    declined_reason: str | None = None
    conversation_start_utc: str | None = None
    conversation_end_utc: str | None = None
    sentiment: str | None = None
    next_action: str | None = None
    notes: str | None = None
    follow_up_required: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class LogInteractionResult(_ResultBaseModel):
    status: Literal["logged"]
    interaction: dict[str, Any]


class EscalateToSupervisorArgs(_ArgsBaseModel):
    customer_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    conversation_summary: str = Field(min_length=1)
    risk_tier: Literal["low", "medium", "high"]
    escalation_category: str | None = None
    supervisor_queue: str | None = None
    requested_by: str | None = None
    user_visible_message: str | None = None
    incident_reference: str | None = None
    attempted_offer_ids: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class EscalateToSupervisorResult(_ResultBaseModel):
    status: Literal["escalated"]
    escalation: dict[str, Any]


class PredictPayload(_ResultBaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: Literal["low", "medium", "high"]
    top_risk_factors: dict[str, Any]


class PredictChurnStepResult(_ResultBaseModel):
    status: Literal["ok"]
    prediction: PredictPayload
    raw_prediction: dict[str, Any]
    model_source: Literal["trained_model", "mock_fallback"]


TOOL_ARG_SCHEMAS: dict[str, type[BaseModel]] = {
    "lookup_customer": LookupCustomerArgs,
    "get_retention_offers": GetRetentionOffersArgs,
    "log_interaction": LogInteractionArgs,
    "escalate_to_supervisor": EscalateToSupervisorArgs,
}

TOOL_RESULT_SCHEMAS: dict[str, type[BaseModel]] = {
    "lookup_customer": LookupCustomerResult,
    "get_retention_offers": GetRetentionOffersResult,
    "log_interaction": LogInteractionResult,
    "escalate_to_supervisor": EscalateToSupervisorResult,
}


def validate_schema(payload: dict[str, Any], schema: type[BaseModel]) -> tuple[bool, dict[str, Any], str]:
    try:
        if hasattr(schema, "model_validate"):
            validated = schema.model_validate(payload)  # type: ignore[attr-defined]
            if hasattr(validated, "model_dump"):
                return True, validated.model_dump(mode="python"), ""
            return True, validated.dict(), ""

        validated = schema.parse_obj(payload)  # type: ignore[attr-defined]
        return True, validated.dict(), ""
    except Exception as exc:
        return False, payload, str(exc)
