import streamlit as st


def render_tool_panel(tool_trace: list[dict]) -> None:
    st.subheader("Tool Execution")
    st.caption("Shows tool status for the latest assistant run in this chat session.")
    executed = {event.get("tool") for event in (tool_trace or [])}

    ordered_tools = [
        "lookup_customer",
        "predict_churn",
        "get_retention_offers",
        "log_interaction",
        "escalate_to_supervisor",
    ]

    called_indices = [idx for idx, name in enumerate(ordered_tools) if name in executed]
    max_called_idx = max(called_indices) if called_indices else -1

    for index, tool_name in enumerate(ordered_tools, start=1):
        if tool_name in executed:
            badge = "Executed"
        elif index - 1 <= max_called_idx:
            badge = "Skipped"
        else:
            badge = "Not called"

        st.write(f"- {index}. {tool_name}: {badge}")
