import streamlit as st


def _latest_tool_result(tool_trace: list[dict], tool_name: str) -> dict | None:
    for event in reversed(tool_trace or []):
        if event.get("tool") == tool_name and isinstance(event.get("result"), dict):
            return event.get("result")
    return None


def render_profile_panel(tool_trace: list[dict]) -> None:
    lookup = _latest_tool_result(tool_trace, "lookup_customer")
    if not lookup or lookup.get("status") != "found":
        return

    profile = lookup.get("profile", {})
    contract = profile.get("contract", {})
    tenure = profile.get("tenure", {})
    charges = profile.get("charges", {})
    satisfaction = profile.get("satisfaction", {})

    st.subheader("Customer Profile (Read-only)")
    st.write(f"Customer ID: {lookup.get('customer_id')}")
    st.write(f"Contract: {contract.get('contract_type')}")
    st.write(f"Tenure (months): {tenure.get('tenure_months')}")
    st.write(f"Monthly Charges: {charges.get('monthly_charges')}")
    st.write(f"Satisfaction Score: {satisfaction.get('satisfaction_score')}")
