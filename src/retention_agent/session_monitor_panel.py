from __future__ import annotations

import streamlit as st

from src.session_store import (
    fetch_recent_session_events,
    fetch_recent_sessions,
    get_session_status_counts,
)


def _render_status_cards() -> None:
    counts = get_session_status_counts()
    active_count = int(counts.get("active", 0))
    closed_count = int(counts.get("closed", 0))
    expired_count = int(counts.get("expired", 0))
    total_count = int(counts.get("total", active_count + closed_count + expired_count))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", total_count)
    c2.metric("Active", active_count)
    c3.metric("Closed", closed_count)
    c4.metric("Expired", expired_count)


def render_session_monitor_panel() -> None:
    st.subheader("Session Monitor")
    _render_status_cards()

    sessions = fetch_recent_sessions(limit=15)
    if sessions:
        st.caption("Recent sessions")
        st.dataframe(sessions, use_container_width=True, hide_index=True)
    else:
        st.info("No session data found yet.")

    events = fetch_recent_session_events(limit=30)
    if events:
        st.caption("Recent events")
        st.dataframe(events, use_container_width=True, hide_index=True)
