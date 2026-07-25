from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.logger_utils import logger
from src.session_store import log_session_event, set_session_status


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
PROMPT_FILE = PROJECT_ROOT / "prompts" / "retention_polish_system_prompt.md"
DEFAULT_POLISH_SYSTEM_PROMPT = (
    "You are a retention assistant. Rewrite the provided recommendation "
    "for a human retention representative in a natural, professional tone. "
    "Keep all facts unchanged. Do not add new data. Keep escalation details concise and easy to read. "
    "Never expose raw tool-call syntax or internal function names. "
    "For non-escalation recommendations, always include a heading exactly as: "
    "Here's a suggested script for presenting the offer: "
    "followed by one natural quoted script the representative can read directly."
)


def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def now_ist_str() -> str:
    ist = timezone(timedelta(hours=5, minutes=30), name="IST")
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")


def get_session_jsonl_file(session_id: str) -> Path:
    return LOG_DIR / f"{session_id}.jsonl"


def get_session_log_file(session_id: str) -> Path:
    return LOG_DIR / f"{session_id}.log"


def log_step(session_id: str, step: str, payload: dict[str, Any], source: str = "orchestrator") -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp_ist": now_ist_str(),
        "session_id": session_id,
        "step": step,
        "payload": payload,
        "source": source,
    }

    session_jsonl_file = get_session_jsonl_file(session_id)
    session_log_file = get_session_log_file(session_id)
    event_json = json.dumps(event, ensure_ascii=True)

    with session_jsonl_file.open("a", encoding="utf-8") as handle:
        handle.write(event_json + "\n")

    with session_log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{event['timestamp_ist']} | {source} | {step} | {event_json}\n")

    log_session_event(
        session_id=session_id,
        source=source,
        step=step,
        payload=payload,
        event_timestamp=event["timestamp_ist"],
        session_jsonl_file=str(session_jsonl_file),
    )

    if step == "chat_session_closed":
        set_session_status(
            session_id=session_id,
            status="closed",
            reason=str(payload.get("reason") or "manual_close"),
        )

    logger.info("retention_agent_step | session_id=%s | step=%s | source=%s", session_id, step, source)


def load_polish_system_prompt() -> str:
    try:
        prompt_text = PROMPT_FILE.read_text(encoding="utf-8").strip()
        if prompt_text:
            return prompt_text
    except Exception:
        pass
    return DEFAULT_POLISH_SYSTEM_PROMPT
