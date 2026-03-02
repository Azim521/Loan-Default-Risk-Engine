# 💰 Loan Default Risk Prediction & Scoring Engine

An end-to-end machine learning system to predict borrower default probability using 1.3M+ historical Lending Club loan records.

The model estimates probability of default and converts it into a business-friendly **0–100 risk score**.  
The system is deployed as a live Streamlit web application.

---

## 🚀 Live Application

🔗 **Try the live app here:**  
[(Streamlit App)](https://loan-default-risk-engine-gxibehstp7sugrrart2cfa.streamlit.app/)

---

## 📌 Problem Statement

Financial institutions need to evaluate borrower credit risk before approving loans.  
This project builds a predictive model to:

- Estimate probability of default
- Identify key drivers of credit risk
- Convert predictions into interpretable risk scores

---

## 📊 Dataset

- Source: Lending Club Loan Dataset (Kaggle)
- Total Records Used: ~1.3 Million
- Target Variable:
  - `0` → Fully Paid
  - `1` → Charged Off (Default)

Data leakage features were removed to ensure realistic modeling.

---

## 🧹 Data Processing & Feature Engineering

### Key Steps:
- Filtered finalized loans only
- Created binary default variable
- Removed data leakage columns
- Handled missing values using median imputation
- Cleaned employment length feature
- Applied one-hot encoding to categorical variables
- Stratified train-test split

### Engineered Financial Stress Features:
- Loan-to-Income Ratio
- Installment-to-Income Ratio
- Revolving Balance-to-Income Ratio
- Ordinal Loan Grade Encoding

---

## 🤖 Models Compared

| Model | ROC-AUC |
|--------|----------|
| Logistic Regression | 0.706 |
| Random Forest | 0.699 |
| XGBoost | **0.719** |

XGBoost achieved the best ranking performance and was selected as the final model.

---

## 📈 Model Evaluation

- Class imbalance handled using `scale_pos_weight`
- Precision-Recall tradeoff analyzed
- ROC-AUC used as primary evaluation metric
- Final ROC-AUC: **~0.72**

---

## 🔎 Model Explainability (SHAP)

SHAP analysis identified the strongest drivers of default risk:

- Loan Grade
- Interest Rate
- Debt-to-Income Ratio
- Loan Term
- Credit Utilization

This improves transparency and model trustworthiness.

---

## 💰 Risk Scoring System

Predicted probability is converted into a **0–100 risk score**:

- 0–30 → Low Risk  
- 30–60 → Medium Risk  
- 60–100 → High Risk  

This enables practical credit decision-making.

---

## 🛠 Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit

---

## 📦 Deployment

The application is deployed using **Streamlit Cloud** and allows real-time borrower risk prediction.

---

## 📌 Key Takeaways

- Built full ML pipeline from raw data to deployment
- Handled class imbalance in large-scale dataset
- Applied feature engineering based on financial domain knowledge
- Integrated explainable AI (SHAP)
- Delivered production-ready web application

---

## 👤 Author

Azim Sadath  
Aspiring Data Scientist | Machine Learning Enthusiast
