"""
Customer Churn Prediction — Streamlit App
==========================================
Interactive web app for predicting customer churn with explanations.

Run: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# --- Page Config ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# --- Load Model Artifacts ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, 'churn_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    explainer = shap.TreeExplainer(model)
    return model, scaler, feature_names, explainer


# --- Header ---
st.title("Customer Churn Prediction")
st.markdown(
    "Predict whether a telecom customer will churn, with **explainable AI** "
    "showing the key factors driving each prediction."
)
st.divider()

# --- Sidebar Inputs ---
st.sidebar.header("Customer Details")
st.sidebar.markdown("Enter customer information below:")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 50.0, step=0.5)
total_charges = st.sidebar.number_input(
    "Total Charges ($)", min_value=0.0,
    value=float(tenure * monthly_charges), step=10.0
)

st.sidebar.divider()
st.sidebar.subheader("Services")
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.sidebar.divider()
st.sidebar.subheader("Account")
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)",
    "Credit card (automatic)"
])


def build_feature_vector():
    """Build the feature vector matching the training pipeline."""
    data = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        # MultipleLines
        'MultipleLines_No phone service': 1 if multiple_lines == "No phone service" else 0,
        'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0,
        # InternetService
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        # OnlineSecurity
        'OnlineSecurity_No internet service': 1 if online_security == "No internet service" else 0,
        'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
        # OnlineBackup
        'OnlineBackup_No internet service': 1 if online_backup == "No internet service" else 0,
        'OnlineBackup_Yes': 1 if online_backup == "Yes" else 0,
        # DeviceProtection
        'DeviceProtection_No internet service': 1 if device_protection == "No internet service" else 0,
        'DeviceProtection_Yes': 1 if device_protection == "Yes" else 0,
        # TechSupport
        'TechSupport_No internet service': 1 if tech_support == "No internet service" else 0,
        'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
        # StreamingTV
        'StreamingTV_No internet service': 1 if streaming_tv == "No internet service" else 0,
        'StreamingTV_Yes': 1 if streaming_tv == "Yes" else 0,
        # StreamingMovies
        'StreamingMovies_No internet service': 1 if streaming_movies == "No internet service" else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == "Yes" else 0,
        # Contract
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        # PaymentMethod
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
    }
    return pd.DataFrame([data])


# --- Prediction ---
if st.sidebar.button("Predict Churn", type="primary", use_container_width=True):
    try:
        model, scaler, feature_names, explainer = load_artifacts()

        # Build features
        input_df = build_feature_vector()

        # Ensure correct column order
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Scale numerical features
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_scaled = input_df.copy()
        input_scaled[num_cols] = scaler.transform(input_scaled[num_cols])

        # Predict
        churn_prob = model.predict_proba(input_scaled)[0][1]
        churn_pred = "Yes" if churn_prob >= 0.5 else "No"

        # Risk category
        if churn_prob < 0.3:
            risk = "LOW"
            risk_color = "green"
        elif churn_prob < 0.6:
            risk = "MEDIUM"
            risk_color = "orange"
        else:
            risk = "HIGH"
            risk_color = "red"

        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Prediction", churn_pred)
        with col2:
            st.metric("Churn Probability", f"{churn_prob:.1%}")
        with col3:
            st.metric("Risk Level", risk)

        st.divider()

        # SHAP explanation
        st.subheader("Why this prediction?")
        st.markdown("The chart below shows which factors are pushing the prediction "
                    "toward churn (red) or retention (blue).")

        shap_values = explainer.shap_values(input_scaled)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_scaled.iloc[0],
            feature_names=feature_names
        ), max_display=10, show=False)
        st.pyplot(fig)
        plt.close()

    except FileNotFoundError:
        st.error(
            "Model files not found. Please run the notebook first to train "
            "and save the model artifacts."
        )
else:
    st.info("Configure customer details in the sidebar and click **Predict Churn**.")

# --- Footer ---
st.divider()
st.caption("Built with Scikit-learn, XGBoost, SHAP, and Streamlit")
