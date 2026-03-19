import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load Model & Feature Columns
# -----------------------------
model = joblib.load("model/xgb_loan_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

st.set_page_config(page_title="Loan Default Risk Engine", layout="centered")

st.title("💰 Loan Default Risk Prediction")
st.write("Estimate borrower default probability and risk category.")

st.markdown("---")

# -----------------------------
# User Input Section
# -----------------------------
loan_amnt = st.number_input("Loan Amount ($)", 1000, 1000000, 10000)
annual_inc = st.number_input("Annual Income ($)", 1000, 1000000, 50000)
int_rate = st.number_input("Interest Rate (%)", 1.0, 40.0, 12.0)
installment = st.number_input("Installment ($)", 50.0, 5000.0, 300.0)
dti = st.number_input("Debt-to-Income Ratio", 0.0, 60.0, 15.0)
term = st.selectbox("Loan Term (Months)", [36, 60])
grade = st.selectbox("Loan Grade", ["A","B","C","D","E","F","G"])
emp_length = st.number_input("Employment Length (Years)", 0, 40, 5)
home_ownership = st.selectbox("Home Ownership", ["RENT","MORTGAGE","OWN"])
verification_status = st.selectbox("Verification Status", ["Verified","Source Verified","Not Verified"])
delinq_2yrs = st.number_input("Delinquencies (2 Years)", 0, 20, 0)
open_acc = st.number_input("Open Credit Accounts", 0, 50, 5)
pub_rec = st.number_input("Public Records", 0, 10, 0)
revol_bal = st.number_input("Revolving Balance", 0, 500000, 10000)
revol_util = st.number_input("Revolving Utilization (%)", 0.0, 150.0, 30.0)
mort_acc = st.number_input("Mortgage Accounts", 0, 20, 1)
pub_rec_bankruptcies = st.number_input("Public Record Bankruptcies", 0, 5, 0)

# -----------------------------
# Feature Engineering
# -----------------------------
loan_to_income = loan_amnt / (annual_inc + 1)
installment_to_income = installment / (annual_inc / 12 + 1)
revol_bal_to_income = revol_bal / (annual_inc + 1)

grade_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
grade_num = grade_map[grade]

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_dict = {
    "loan_amnt": loan_amnt,
    "term": term,
    "int_rate": int_rate,
    "installment": installment,
    "emp_length": emp_length,
    "annual_inc": annual_inc,
    "dti": dti,
    "delinq_2yrs": delinq_2yrs,
    "open_acc": open_acc,
    "pub_rec": pub_rec,
    "revol_bal": revol_bal,
    "revol_util": revol_util,
    "mort_acc": mort_acc,
    "pub_rec_bankruptcies": pub_rec_bankruptcies,
    "loan_to_income": loan_to_income,
    "installment_to_income": installment_to_income,
    "revol_bal_to_income": revol_bal_to_income,
    "grade_num": grade_num
}

input_df = pd.DataFrame([input_dict])

# -----------------------------
# One-Hot Encoding Alignment
# -----------------------------
input_df = pd.get_dummies(input_df)

# Add missing columns
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Ensure correct column order
input_df = input_df[feature_columns]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Risk"):
    probability = model.predict_proba(input_df)[0][1]
    risk_score = round(probability * 100, 2)

    if risk_score < 30:
        category = "Low Risk"
    elif risk_score < 60:
        category = "Medium Risk"
    else:
        category = "High Risk"

    st.markdown("---")
    st.subheader("📊 Prediction Results")
    st.write(f"**Default Probability:** {round(probability, 4)}")
    st.write(f"**Risk Score (0–100):** {risk_score}")
    st.write(f"**Risk Category:** {category}")
