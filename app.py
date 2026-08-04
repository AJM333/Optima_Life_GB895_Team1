import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="OptimaLife Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------
# Load Model
# -----------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("capstone_rf_model_v1.joblib")
    encoders = joblib.load("capstone_label_encoders_v1.joblib")
    return model, encoders


model, encoders = load_model()

# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.title("📊 About This Project")

st.sidebar.write("""
This dashboard predicts whether an OptimaLife customer is likely to renew
their subscription using a Random Forest machine learning model.

The prediction is based on customer demographics and engagement metrics.
""")

st.sidebar.markdown("---")

st.sidebar.write("**Model:** Random Forest Classifier")
st.sidebar.write("**Target Variable:** Customer Renewal")
st.sidebar.write("**Course:** GEN BUS 895 Capstone")

# -----------------------------------
# Title
# -----------------------------------
st.title("📊 OptimaLife Customer Churn Prediction Dashboard")

st.write("""
Predict the probability that a customer will renew their subscription
based on engagement behavior and demographic information.
""")

st.divider()

# -----------------------------------
# Customer Inputs
# -----------------------------------
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    total_sessions = st.number_input(
        "Total Number of Sessions",
        min_value=0,
        value=50
    )

    gross_session_length = st.number_input(
        "Gross Total Session Length",
        min_value=0.0,
        value=500.0
    )

    active_days = st.number_input(
        "Active Days",
        min_value=0,
        value=30
    )

    active_quarters = st.number_input(
        "Active Quarters",
        min_value=1,
        value=4
    )

    avg_sessions = st.number_input(
        "Average Sessions per Active Quarter",
        min_value=0.0,
        value=12.5
    )

with col2:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    education = st.selectbox(
        "Education",
        encoders["EDUCATION"].classes_
    )

    income_level = st.selectbox(
        "Income Level",
        encoders["INCOME_LEVEL"].classes_
    )

    device_type = st.selectbox(
        "Device Type",
        encoders["DEVICE_TYPE"].classes_
    )

    tech_comfort = st.slider(
        "Tech Comfort Score",
        min_value=1,
        max_value=10,
        value=5
    )

st.divider()

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("Predict Churn Probability", use_container_width=True):

    # Encode categorical variables
    education_encoded = encoders["EDUCATION"].transform([education])[0]
    income_encoded = encoders["INCOME_LEVEL"].transform([income_level])[0]
    device_encoded = encoders["DEVICE_TYPE"].transform([device_type])[0]

    # Build prediction dataframe
    input_df = pd.DataFrame([{
        "TOTAL_NUM_SESSIONS": total_sessions,
        "GROSS_TOTAL_SESSION_LENGTH": gross_session_length,
        "ACTIVE_DAYS": active_days,
        "ACTIVE_QUARTERS": active_quarters,
        "AVG_SESSIONS_PER_ACTIVE_QUARTER": avg_sessions,
        "AGE": age,
        "EDUCATION": education_encoded,
        "INCOME_LEVEL": income_encoded,
        "DEVICE_TYPE": device_encoded,
        "TECH_COMFORT_SCORE": tech_comfort
    }])

    renewal_probability = model.predict_proba(input_df)[0][1]
    churn_probability = 1 - renewal_probability

    st.header("Prediction Results")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            "Renewal Probability",
            f"{renewal_probability:.1%}"
        )

    with metric2:
        st.metric(
            "Churn Probability",
            f"{churn_probability:.1%}"
        )

    st.progress(float(renewal_probability))

    if renewal_probability >= 0.80:
        st.success("🟢 Low Churn Risk")
    elif renewal_probability >= 0.50:
        st.warning("🟡 Medium Churn Risk")
    else:
        st.error("🔴 High Churn Risk")

    st.subheader("Prediction Summary")

    st.write(
        f"""
        Based on the customer information entered, the Random Forest model predicts
        a **{renewal_probability:.1%} probability** that this customer will renew
        their subscription and a **{churn_probability:.1%} probability** that they
        will churn.

        This prediction can help identify customers who may benefit from
        additional outreach or retention efforts.
        """
    )
