from datetime import datetime, timedelta, timezone
from uuid import uuid4

import streamlit as st


SESSION_TTL_MINUTES = 45
SESSION_ID_PREFIX = "RET-"


def _new_session_id() -> str:
    return f"{SESSION_ID_PREFIX}{uuid4().hex[:10].upper()}"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _seed_chat_history() -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": (
                "Hi! Tell me what the customer said, and I will help you with the next best response."
            ),
        }
    ]


def _init_session_state() -> None:
    st.session_state.chat_history = _seed_chat_history()
    st.session_state.latest_tool_trace = []
    st.session_state.active_session_id = _new_session_id()
    st.session_state.active_customer_id = ""
    st.session_state.session_started = _now_utc_iso()
    st.session_state.last_activity = st.session_state.session_started


def _parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_session_expired() -> bool:
    last_activity = _parse_iso_utc(st.session_state.get("last_activity"))
    if not last_activity:
        return False
    return datetime.now(timezone.utc) - last_activity > timedelta(minutes=SESSION_TTL_MINUTES)


def _touch_last_activity() -> None:
    st.session_state.last_activity = _now_utc_iso()


def start_new_chat(log_step=None, reason: str = "manual_new_chat") -> None:
    previous_session_id = str(st.session_state.get("active_session_id") or "").strip()
    if previous_session_id and callable(log_step):
        log_step(previous_session_id, "chat_session_closed", {"reason": reason})

    _init_session_state()

    if callable(log_step):
        log_step(
            st.session_state.active_session_id,
            "chat_session_started",
            {
                "reason": reason,
                "session_ttl_minutes": SESSION_TTL_MINUTES,
            },
        )


def ensure_chat_state(log_step=None) -> None:
    required_keys = {
        "chat_history",
        "latest_tool_trace",
        "active_session_id",
        "active_customer_id",
        "session_started",
        "last_activity",
    }
    if any(key not in st.session_state for key in required_keys):
        _init_session_state()
        if callable(log_step):
            log_step(
                st.session_state.active_session_id,
                "chat_session_started",
                {
                    "reason": "state_initialized",
                    "session_ttl_minutes": SESSION_TTL_MINUTES,
                },
            )
        return

    if _is_session_expired():
        start_new_chat(log_step=log_step, reason="inactive_ttl_expired")


def render_chat_controls(log_step) -> None:
    st.subheader("Session")
    st.caption(f"Session ID: {st.session_state.active_session_id}")
    st.caption(f"Session timeout: {SESSION_TTL_MINUTES} minutes of inactivity")
    if st.button("New Chat", use_container_width=True):
        start_new_chat(log_step=log_step, reason="manual_new_chat")
        st.rerun()


def render_chat_history() -> None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def render_chat_section(run_orchestrator, log_step) -> None:
    st.caption("Share the customer issue naturally. If you know the customer ID, include it.")
    user_prompt = st.chat_input("Ask TeleConnect AI anything about this customer case...")

    if not user_prompt:
        return

    _touch_last_activity()
    session_id = st.session_state.active_session_id
    log_step(session_id, "user_input_received", {"user_prompt": user_prompt})

    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running agent..."):
            result = run_orchestrator(
                user_message=user_prompt,
                selected_customer_id=st.session_state.active_customer_id or None,
                prediction_model=None,
                session_id=session_id,
            )

        assistant_reply = result.get("assistant_message", "No response returned.")
        st.write(assistant_reply)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

        returned_session_id = str(result.get("session_id") or "").strip()
        if returned_session_id:
            st.session_state.active_session_id = returned_session_id

        resolved_customer_id = str(result.get("resolved_customer_id") or "").strip()
        if resolved_customer_id:
            st.session_state.active_customer_id = resolved_customer_id

        st.session_state.latest_tool_trace = result.get("tool_trace", [])
        log_step(
            st.session_state.active_session_id,
            "tool_trace_snapshot",
            {
                "status": result.get("status"),
                "tool_trace": st.session_state.latest_tool_trace,
            },
        )
        log_step(
            st.session_state.active_session_id,
            "assistant_response_ready",
            {
                "status": result.get("status"),
                "tool_trace_count": len(st.session_state.latest_tool_trace),
                "active_customer_id": st.session_state.active_customer_id,
            },
        )
        _touch_last_activity()
