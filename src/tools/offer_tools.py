from __future__ import annotations

from typing import Any

from src.tools.common import build_process_meta, normalize_contract_type, normalize_lower, now_utc_iso


OFFER_CATALOG: list[dict[str, Any]] = [
    {
        "offer_id": "OFF-001",
        "title": "3-Month Loyalty Discount",
        "description": "15% off monthly plan for 3 months.",
        "eligible_risk_tiers": ["high", "medium"],
        "contract_types": ["month-to-month"],
        "expected_retention_lift_pct": 14,
        "estimated_cost_usd": 42,
        "priority": 100,
    },
    {
        "offer_id": "OFF-002",
        "title": "Contract Upgrade Bonus",
        "description": "No-fee move to one-year contract plus one-time $30 credit.",
        "eligible_risk_tiers": ["high", "medium", "low"],
        "contract_types": ["month-to-month"],
        "expected_retention_lift_pct": 11,
        "estimated_cost_usd": 30,
        "priority": 90,
    },
    {
        "offer_id": "OFF-003",
        "title": "Premium Support Add-on",
        "description": "Priority support queue and proactive service checks for 6 months.",
        "eligible_risk_tiers": ["high"],
        "contract_types": ["month-to-month", "one year", "two year"],
        "expected_retention_lift_pct": 9,
        "estimated_cost_usd": 24,
        "priority": 80,
    },
    {
        "offer_id": "OFF-004",
        "title": "Data Boost Pack",
        "description": "Complimentary +20GB/month for 4 months.",
        "eligible_risk_tiers": ["medium", "low"],
        "contract_types": ["one year", "two year"],
        "expected_retention_lift_pct": 6,
        "estimated_cost_usd": 18,
        "priority": 70,
    },
    {
        "offer_id": "OFF-005",
        "title": "Price Lock Renewal",
        "description": "Freeze current monthly rate for 12 months on renewal.",
        "eligible_risk_tiers": ["high", "medium"],
        "contract_types": ["one year", "two year"],
        "expected_retention_lift_pct": 10,
        "estimated_cost_usd": 12,
        "priority": 85,
    },
]


def get_retention_offers(risk_tier: str, contract_type: str, max_results: int = 3) -> dict[str, Any]:
    normalized_tier = normalize_lower(risk_tier)
    normalized_contract = normalize_contract_type(contract_type)

    filtered = [
        offer
        for offer in OFFER_CATALOG
        if normalized_tier in offer["eligible_risk_tiers"]
        and normalized_contract in offer["contract_types"]
    ]

    filtered.sort(
        key=lambda item: (
            item.get("priority", 0),
            item.get("expected_retention_lift_pct", 0),
        ),
        reverse=True,
    )

    requested = max(1, int(max_results))
    returned = min(len(filtered), requested)

    return {
        "risk_tier": normalized_tier,
        "contract_type": normalized_contract,
        "offers": filtered[:requested],
        "catalog_size": len(OFFER_CATALOG),
        "returned": returned,
        "meta": build_process_meta(
            tool_name="get_retention_offers",
            source="in_memory_catalog",
            flags={
                "offers_found": returned > 0,
                "returned_count": returned,
                "requested_count": requested,
                "risk_tier_normalized": normalized_tier == normalize_lower(risk_tier),
                "contract_type_normalized": normalized_contract == normalize_contract_type(contract_type),
            },
        ),
        "retrieved_at": now_utc_iso(),
    }
