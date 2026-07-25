import streamlit as st

from src.retention_agent.chat_panel import (
    ensure_chat_state,
    render_chat_controls,
    render_chat_history,
    render_chat_section,
)
from src.retention_agent.prediction_panel import render_prediction_panel
from src.retention_agent.profile_panel import render_profile_panel
from src.retention_agent.session_monitor_panel import render_session_monitor_panel
from src.retention_agent.runtime import log_step
from src.retention_orchestrator import run_retention_orchestrator


def _app_log_step(session_id: str, step: str, payload: dict) -> None:
    log_step(session_id, step, payload, source="app")


def _ensure_panel_state() -> None:
    defaults = {
        "show_customer_profile": False,
        "show_churn_prediction": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    st.set_page_config(
        page_title="TeleConnect Retention Assistant",
        page_icon="R",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 880px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 0.2rem 0.1rem;
        }
        div[data-testid="stChatInput"] textarea {
            border-radius: 14px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("TeleConnect Assistant")
    # st.caption(
    #     "Natural-language retention assistant with tool chaining, ambiguity handling, escalation detection, and audit logs."
    # )

    ensure_chat_state(_app_log_step)
    _ensure_panel_state()

    with st.sidebar:
        st.header("How To Use")
        st.markdown(
            "1. Enter a natural-language message from a rep.\n"
            "2. Include customer_id when available (example: TC-000178).\n"
            "3. Agent will chain tools and provide an actionable recommendation."
        )
        st.divider()
        render_chat_controls(_app_log_step)
        st.divider()
        render_session_monitor_panel()

    render_chat_history()
    render_chat_section(run_retention_orchestrator, _app_log_step)

    if st.session_state.latest_tool_trace:
        c1, c2 = st.columns(2)
        if c1.button("Customer Profile", use_container_width=True):
            st.session_state.show_customer_profile = not st.session_state.show_customer_profile
        if c2.button("Churn Prediction", use_container_width=True):
            st.session_state.show_churn_prediction = not st.session_state.show_churn_prediction

        if st.session_state.show_customer_profile:
            render_profile_panel(st.session_state.latest_tool_trace)
            st.divider()

        if st.session_state.show_churn_prediction:
            render_prediction_panel(st.session_state.latest_tool_trace)


if __name__ == "__main__":
    main()
