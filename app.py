import streamlit as st

from src.config import DATA_DIR, DATA_FILE_PATTERN, MLFLOW_RUN_ID, MODEL_NAME
from src.data_loader import list_available_files, load_single_file, get_customer_record
from src.logger_utils import logger
from src.model_loader import load_mlflow_model
from src.predictor import predict_churn, save_prediction_json


@st.cache_resource(show_spinner=False)
def initialize_model(run_id: str, model_name: str):
    return load_mlflow_model(
        run_id=run_id,
        model_name=model_name,
    )


def main():
    st.set_page_config(
        page_title="Churn Prediction App",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Customer Churn Prediction Dashboard")
    st.caption(
        "Load a local customer dataset, select or search a customer, and predict churn using the MLflow model."
    )

    logger.info("Starting Streamlit churn app")

    with st.sidebar:
        st.header("Customer Lookup")
        # st.info(f"Looking for CSV/XLSX/JSON files in: {DATA_DIR}")

        available_files = list_available_files(DATA_DIR, DATA_FILE_PATTERN)

        if not available_files:
            st.error("No CSV/XLSX/JSON files found in the configured folder.")
            st.stop()

        default_file_option = "-- Select data file --"
        file_names = [f.name for f in available_files]
        selected_file_name = st.selectbox(
            "Select data file",
            options=[default_file_option] + file_names,
            index=0,
            key="file_select",
        )

        if selected_file_name == default_file_option:
            st.stop()

        selected_file_path = next(f for f in available_files if f.name == selected_file_name)
        data_df = load_single_file(selected_file_path)
        logger.info("Loaded dataset with %s rows from %s", len(data_df), selected_file_path)

        if data_df.empty:
            st.error(f"Could not load data from {selected_file_name}.")
            st.stop()

        st.success(f"Loaded {len(data_df)} customer rows from {selected_file_name}.")

        default_customer_option = "-- Select customer ID --"
        customer_ids = data_df["customer_id"].dropna().astype(str).tolist()
        search_text = st.text_input("Search customer_id", placeholder="Type a customer ID...")

        if search_text:
            filtered_ids = [cid for cid in customer_ids if search_text.lower() in cid.lower()]
            if filtered_ids:
                selected_customer_value = st.selectbox(
                    "Matching customer_id",
                    options=[default_customer_option] + filtered_ids,
                    index=0,
                    key="customer_id_select_v2",
                )
            else:
                st.warning("No customer IDs matched your search term.")
                selected_customer_value = default_customer_option
        else:
            selected_customer_value = st.selectbox(
                "Select Customer ID",
                options=[default_customer_option] + customer_ids,
                index=0,
                key="customer_id_select_v2",
            )

        selected_customer_id = (
            None if selected_customer_value == default_customer_option else selected_customer_value
        )

    if data_df.empty:
        st.error("No local customer dataset found in the configured folder.")
        st.stop()

    if not selected_customer_id:
        st.info("ENTER or SELECT a CUSTOMER ID to view profile and run prediction.")
        st.stop()

    customer_record = get_customer_record(data_df, selected_customer_id)
    logger.info("Selected customer_id: %s", selected_customer_id)

    if customer_record is None:
        st.error("Customer record not found.")
        logger.warning("Customer record not found for selected customer_id: %s", selected_customer_id)
        st.stop()

    st.subheader("Customer Snapshot")
    customer_preview = customer_record.copy()
    customer_preview = customer_preview.drop(columns=["churned"], errors="ignore")

    edited_customer = st.data_editor(
        customer_preview,
        disabled=["customer_id"],
        width="stretch",
        key="customer_editor",
    )

    if st.button("Predict Churn"):
        try:
            logger.info("Prediction started for customer_id=%s", selected_customer_id)
            with st.spinner("Loading MLflow model and generating prediction..."):
                model = initialize_model(MLFLOW_RUN_ID, MODEL_NAME)
                logger.info("Model load attempt completed for run_id=%s", MLFLOW_RUN_ID)
                prediction_input = edited_customer.iloc[0].to_dict()
                prediction_input["customer_id"] = selected_customer_id
                logger.info("Prediction input prepared for customer_id=%s", selected_customer_id)
                prediction = predict_churn(prediction_input, model=model)

            st.subheader("Prediction Result")
            probability = float(prediction["churn_probability"])
            risk_tier = str(prediction["risk_tier"]).lower()

            risk_palette = {
                "high": {"color": "#dc2626", "label": "HIGH"},
                "medium": {"color": "#d97706", "label": "MEDIUM"},
                "low": {"color": "#16a34a", "label": "LOW"},
            }
            risk_style = risk_palette.get(risk_tier, {"color": "#64748b", "label": risk_tier.upper()})

            prob_col, tier_col, indicator_col = st.columns([1, 1, 2])
            with prob_col:
                st.metric("Churn Probability", f"{probability:.4f}")
            with tier_col:
                st.metric("Risk Tier", risk_style["label"])
            with indicator_col:
                st.markdown("### Risk Indicator")
                st.markdown(
                    (
                        "<div style='padding:0.75rem 1rem;border:1px solid #e5e7eb;"
                        "border-radius:0.6rem;background:#f8fafc;'>"
                        f"<span style='display:inline-block;width:0.75rem;height:0.75rem;"
                        f"border-radius:50%;background:{risk_style['color']};margin-right:0.5rem;'></span>"
                        f"<strong style='color:{risk_style['color']};'>{risk_style['label']}</strong>"
                        " <span style='color:#475569;'>risk based on predicted churn probability</span>"
                        "<div style='margin-top:0.4rem;color:#64748b;font-size:0.9rem;'>"
                        "Low: < 0.40 | Medium: 0.40-0.69 | High: >= 0.70"
                        "</div></div>"
                    ),
                    unsafe_allow_html=True,
                )

            st.subheader("Analyst Summary")
            if probability >= 0.70:
                st.error("High churn risk: immediate retention action is recommended.")
            elif probability >= 0.40:
                st.warning("Medium churn risk: targeted retention action is recommended.")
            else:
                st.success("Low churn risk: maintain engagement and monitor changes.")

            factor_block = prediction.get("top_risk_factors", {})
            churn_factors = factor_block.get("for_churn", [])
            not_churn_factors = factor_block.get("for_not_churn", [])

            left_col, right_col = st.columns(2)

            with left_col:
                st.markdown("### Drivers Increasing Churn Risk")
                if churn_factors:
                    for idx, factor in enumerate(churn_factors, start=1):
                        st.write(f"{idx}. {factor}")
                else:
                    st.info("No churn-driving factors could be derived from reference data.")

            with right_col:
                st.markdown("### Drivers Supporting Retention")
                if not_churn_factors:
                    for idx, factor in enumerate(not_churn_factors, start=1):
                        st.write(f"{idx}. {factor}")
                else:
                    st.info("No retention-driving factors could be derived from reference data.")

            json_path = save_prediction_json(prediction, selected_customer_id)
            logger.info("Prediction saved to JSON: %s", json_path)
            st.success(f"Prediction output saved to JSON: {json_path}")
        except Exception as exc:
            logger.exception("Prediction failed for customer_id=%s", selected_customer_id)
            st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
