from __future__ import annotations

from typing import Any

from src.tools.common import (
    VAL_DATA_DIR,
    build_process_meta,
    coerce_float,
    coerce_int,
    is_missing,
    normalize_binary,
    normalize_contract_type,
    normalize_gender,
    normalize_internet_service,
    normalize_payment_method,
    normalize_text,
    now_utc_iso,
    select_customer_row,
    select_customer_row_with_source,
)


def _build_model_features_from_row(row: dict[str, Any]) -> dict[str, Any]:
    blocked_columns = {"churned", "churn", "target", "label"}
    payload: dict[str, Any] = {}
    for key, value in row.items():
        key_name = str(key).strip()
        if key_name.lower() in blocked_columns:
            continue
        if is_missing(value):
            payload[key_name] = None
        else:
            payload[key_name] = value.item() if hasattr(value, "item") else value
    return payload


def get_customer_model_features(customer_id: str) -> dict[str, Any] | None:
    row = select_customer_row(customer_id)
    if row is None:
        return None
    return _build_model_features_from_row(row)


def lookup_customer(customer_id: str) -> dict[str, Any]:
    row, data_source = select_customer_row_with_source(customer_id)
    used_sqlite = data_source == "sqlite"
    used_json_fallback = data_source == "json_fallback"

    if row is None:
        return {
            "status": "not_found",
            "customer_id": customer_id,
            "message": "Customer record not found in local input dataset.",
            "source": data_source,
            "source_path": str(VAL_DATA_DIR),
            "used_sqlite": used_sqlite,
            "used_json_fallback": used_json_fallback,
            "meta": build_process_meta(
                tool_name="lookup_customer",
                source=data_source,
                flags={
                    "found": False,
                    "used_sqlite": used_sqlite,
                    "used_json_fallback": used_json_fallback,
                },
            ),
        }

    monthly_charges = coerce_float(row.get("monthly_charges"))
    total_charges = coerce_float(row.get("total_charges"))
    tenure_months = coerce_float(row.get("tenure_months"))

    profile = {
        "demographics": {
            "age": coerce_int(row.get("age")),
            "gender": normalize_gender(row.get("gender")),
        },
        "contract": {
            "contract_type": normalize_contract_type(row.get("contract_type")),
            "payment_method": normalize_payment_method(row.get("payment_method")),
            "internet_service": normalize_internet_service(row.get("internet_service")),
            "phone_service": normalize_binary(row.get("phone_service")),
            "additional_services": coerce_int(row.get("num_additional_services")),
        },
        "tenure": {
            "tenure_months": tenure_months,
            "last_interaction_date": normalize_text(row.get("last_interaction_date")) or None,
        },
        "charges": {
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "estimated_clv_12m": (
                round((monthly_charges or 0.0) * 12.0, 2)
                if monthly_charges is not None
                else None
            ),
        },
        "satisfaction": {
            "satisfaction_score": coerce_float(row.get("satisfaction_score")),
            "support_tickets": coerce_int(row.get("num_support_tickets")),
        },
    }

    return {
        "status": "found",
        "customer_id": normalize_text(row.get("customer_id")),
        "profile": profile,
        "model_features": _build_model_features_from_row(row),
        "source": data_source,
        "source_path": str(VAL_DATA_DIR),
        "used_sqlite": used_sqlite,
        "used_json_fallback": used_json_fallback,
        "meta": build_process_meta(
            tool_name="lookup_customer",
            source=data_source,
            flags={
                "found": True,
                "used_sqlite": used_sqlite,
                "used_json_fallback": used_json_fallback,
                "has_profile": True,
                "has_model_features": True,
            },
        ),
        "retrieved_at": now_utc_iso(),
    }
