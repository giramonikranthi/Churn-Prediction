from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_FILE_PATTERN
from src.data_loader import list_available_files, load_single_file
from src.session_store import log_output_record


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VAL_DATA_DIR = PROJECT_ROOT / "data" / "Val_data"
SQLITE_DB_PATH = PROJECT_ROOT / "data" / "SQLite_storage" / "retention_sessions.db"
OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "Output_data"
INTERACTION_LOG_FILE = OUTPUT_DATA_DIR / "retention_interactions.jsonl"
ESCALATION_LOG_FILE = OUTPUT_DATA_DIR / "retention_escalations.jsonl"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_process_meta(
    *,
    tool_name: str,
    source: str | None = None,
    flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tool": normalize_text(tool_name),
        "generated_at": now_utc_iso(),
        "flags": flags if isinstance(flags, dict) else {},
    }
    if source:
        payload["source"] = normalize_text(source)
    return payload


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_lower(value: Any) -> str:
    return normalize_text(value).lower()


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, int):
        return float(value)

    text = normalize_text(value)
    if text == "":
        return None

    try:
        return float(text)
    except ValueError:
        return None


def coerce_int(value: Any) -> int | None:
    as_float = coerce_float(value)
    if as_float is None:
        return None
    return int(round(as_float))


def normalize_binary(value: Any) -> str:
    normalized = normalize_lower(value)
    if normalized in {"yes", "y", "true", "1"}:
        return "yes"
    if normalized in {"no", "n", "false", "0"}:
        return "no"
    return "unknown"


def normalize_gender(value: Any) -> str:
    normalized = normalize_lower(value)
    if normalized in {"m", "male"}:
        return "male"
    if normalized in {"f", "female"}:
        return "female"
    return "unknown"


def normalize_contract_type(value: Any) -> str:
    normalized = normalize_lower(value)
    if "month" in normalized:
        return "month-to-month"
    if "one" in normalized and "year" in normalized:
        return "one year"
    if "two" in normalized and "year" in normalized:
        return "two year"
    return normalized or "unknown"


def normalize_internet_service(value: Any) -> str:
    normalized = normalize_lower(value)
    if normalized in {"", "nan", "none", "null"}:
        return "none"
    if "fiber" in normalized:
        return "fiber optic"
    if "dsl" in normalized:
        return "dsl"
    if normalized == "no":
        return "none"
    return normalized


def normalize_payment_method(value: Any) -> str:
    normalized = normalize_lower(value)
    aliases = {
        "bt": "bank transfer",
        "cc": "credit card",
    }
    return aliases.get(normalized, normalized or "unknown")


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = normalize_lower(value)
    return text in {"", "nan", "none", "null"}


def _load_input_customer_dataset_from_sqlite() -> pd.DataFrame:
    if not SQLITE_DB_PATH.exists():
        return pd.DataFrame()

    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            rows = conn.execute(
                """
                SELECT customer_id, record_json
                FROM customer_validation_data
                ORDER BY ingested_at_time DESC
                """
            ).fetchall()
    except sqlite3.Error:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for customer_id, record_json in rows:
        cid = normalize_text(customer_id)
        if not cid:
            continue
        lowered = cid.lower()
        if lowered in seen_ids:
            continue

        try:
            payload = json.loads(record_json) if record_json else {}
        except (TypeError, json.JSONDecodeError):
            payload = {}

        if not isinstance(payload, dict):
            payload = {}

        payload["customer_id"] = cid
        records.append(payload)
        seen_ids.add(lowered)

    if not records:
        return pd.DataFrame()

    dataset = pd.DataFrame(records)
    dataset.columns = [str(col).strip() for col in dataset.columns]
    if "customer_id" in dataset.columns:
        dataset["customer_id"] = dataset["customer_id"].astype(str).str.strip()
    return dataset


def load_input_customer_dataset() -> pd.DataFrame:
    dataset, _ = load_input_customer_dataset_with_source()
    return dataset


def load_input_customer_dataset_with_source() -> tuple[pd.DataFrame, str]:
    # Prefer SQLite-backed records for runtime lookup; fallback to JSON files.
    sqlite_dataset = _load_input_customer_dataset_from_sqlite()
    if not sqlite_dataset.empty:
        return sqlite_dataset, "sqlite"

    if not VAL_DATA_DIR.exists():
        return pd.DataFrame(), "json_fallback"

    preferred = VAL_DATA_DIR / "churn_data_v1.json"
    if preferred.exists():
        dataset = load_single_file(preferred)
    else:
        candidates = list_available_files(VAL_DATA_DIR, DATA_FILE_PATTERN)
        if not candidates:
            return pd.DataFrame(), "json_fallback"
        dataset = load_single_file(candidates[0])

    dataset.columns = [str(col).strip() for col in dataset.columns]
    if "customer_id" in dataset.columns:
        dataset["customer_id"] = dataset["customer_id"].astype(str).str.strip()

    return dataset, "json_fallback"


def select_customer_row(customer_id: str) -> dict[str, Any] | None:
    row, _ = select_customer_row_with_source(customer_id)
    return row


def select_customer_row_with_source(customer_id: str) -> tuple[dict[str, Any] | None, str]:
    dataset, source = load_input_customer_dataset_with_source()
    if dataset.empty or "customer_id" not in dataset.columns:
        return None, source

    match = dataset.loc[
        dataset["customer_id"].astype(str).str.lower() == normalize_lower(customer_id)
    ]
    if match.empty:
        return None, source

    row = match.iloc[0]
    return {column: row[column] for column in match.columns}, source


def append_jsonl(file_path: Path, payload: dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    log_output_record(file_path=file_path, payload=payload)


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None
