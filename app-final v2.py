import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Assume you already trained and saved your model + scaler + feature_columns
# For example, load them with joblib or pickle:


@st.cache_resource
def load_artifacts():
    model = load("final_model.pkl")
    scaler = load("scaler.pkl")
    features = load("feature_columns.pkl")
    return model, scaler, features


final_model, scaler, feature_columns = load_artifacts()


def predict_loan_application(model, scaler, application_data, feature_columns):
    """Predict loan default probability for a new application"""
    app_df = pd.DataFrame([application_data])

    # Add any missing columns with 0
    for feature in feature_columns:
        if feature not in app_df.columns:
            app_df[feature] = 0

    # Reorder exactly like training
    app_df = app_df[feature_columns]

    # Scale
    app_scaled = scaler.transform(app_df)

    # Predict
    prediction = model.predict(app_scaled)[0]
    probability = model.predict_proba(app_scaled)[0, 1]

    return prediction, probability


# ---------------- Streamlit UI ----------------
st.set_page_config(
    page_title="Loan Application Risk Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .stButton>button {
        background-color: #0e76a8;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        height: 3em;
        width: 100%;
    }
    .big-font { font-size: 50px !important; font-weight: bold; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("<h1 style='text-align: center; color: #0E76A8;'>💰 Loan Application Risk Predictor</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #555;'>Enter applicant details to predict loan default probability.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- Input Form ----------------
with st.form("loan_form"):
    st.subheader("Applicant Information")
    options_list = ['Yes', 'No']

    col1, col2 = st.columns(2)
    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100,
                                     value=30, help="Age of the applicant")
        person_income = st.number_input(
            "Annual Income ($)", min_value=0, value=50000)
        loan_amnt = st.number_input(
            "Requested Loan Amount ($)", min_value=0, value=20000, step=1000)
        loan_int_rate = st.number_input(
            "Loan Interest Rate in percent (annual)", value=10)

    with col2:
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=850, value=650)
        person_emp_exp = st.number_input(
            "Employment Length (years)", min_value=0, value=5)
        previous_loan_defaults_on_file_Yes = st.selectbox(
            'Defaulted on any previous loan?', options_list)
        home_status = st.selectbox(
            "Home Ownership", ["RENT", "MORTGAGE", "OWN"], index=1)

    submitted = st.form_submit_button("Assess Risk", use_container_width=True)

# ---------------- Results - ---------------
if submitted:
    # Derived Features
    loan_percent_income = loan_amnt / person_income if person_income > 0 else 0
    income_to_age = person_income / person_age if person_age > 0 else 0
    credit_age_ratio = credit_score / person_age if person_age > 0 else 0

    person_home_ownership_MORTGAGE = 1 if home_status == "MORTGAGE" else 0
    person_home_ownership_OWN = 1 if home_status == "OWN" else 0
    person_home_ownership_RENT = 1 if home_status == "RENT" else 0

    previous_loan_defaults_string = previous_loan_defaults_on_file_Yes
    previous_loan_defaults_on_file_Yes = 0 if previous_loan_defaults_on_file_Yes == "Yes" else 1.0

    application_data = {
        "person_age": person_age,
        "person_income": person_income,
        "loan_int_rate": loan_int_rate,
        "loan_amnt": loan_amnt,
        "credit_score": credit_score,
        "person_emp_exp": person_emp_exp,
        # "debt_to_income": debt_to_income,
        "loan_percent_income": loan_percent_income,
        "income_to_age": income_to_age,
        "credit_age_ratio": credit_age_ratio,
        "person_home_ownership_MORTGAGE": person_home_ownership_MORTGAGE,
        "person_home_ownership_OWN": person_home_ownership_OWN,
        "person_home_ownership_RENT": person_home_ownership_RENT,
        "previous_loan_defaults_on_file_Yes": previous_loan_defaults_on_file_Yes,
    }

    with st.spinner("Analyzing application..."):
        pred, prob = predict_loan_application(
            final_model, scaler, application_data, feature_columns)

    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Risk Assessment Result</h2>",
                unsafe_allow_html=True)

    # Big Result Badge
    if prob < 0.10:
        risk_level = "Low"
        color = "green"
    elif prob < 0.60:
        risk_level = "Medium"
        color = "#FF8C00"
    else:
        risk_level = "High"
        color = "red"

    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background-color: {color}20; border-radius: 15px; border: 3px solid {color};"> 
    <h1 style="color: {color}; margin:5px;">{risk_level} Risk</h1> 
    <p style="font-size: 1.5rem; color: {color};">Probability: <strong>{prob:.1%}</strong> ({risk_level} Risk)</p> </div>
    """, unsafe_allow_html=True)

    # Progress bar
    st.progress(prob)

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Credit Score", f"{credit_score}", delta=None)
    with col2:
        st.metric("Annual Income", f"${person_income:,.0f}")
    with col3:
        st.metric("Previous Loan Default",
                  f"{previous_loan_defaults_string}")
    with col4:
        st.metric("Employment Length", f"{person_emp_exp} years")

    # Recommendation
    if pred == 0:
        st.success(
            "✅ Application appears safe. Consider approval with standard terms.")
    else:
        st.error(
            "❌ High risk probability detected. Recommend rejection or require co-signer/additional collateral.")

    st.balloons()
