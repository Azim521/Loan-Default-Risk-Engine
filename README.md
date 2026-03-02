# Loan Default Risk Prediction & Scoring Engine

## 📌 Problem Statement
This project builds an end-to-end Loan Default Risk Prediction system using machine learning on 1.3M+ historical Lending Club loan records. The goal is to predict the probability of borrower default and convert it into a business-friendly risk score.

---

## 📊 Dataset
- Source: Lending Club Loan Dataset (Kaggle)
- Total Records After Filtering: ~1.3 Million
- Target Variable:
  - 0 → Fully Paid
  - 1 → Charged Off (Default)

Data leakage features were removed to ensure realistic modeling.

---

## 🧹 Data Preprocessing
- Removed future/payment-based leakage features
- Handled missing values using median imputation
- Converted categorical variables using one-hot encoding
- Engineered financial stress features:
  - Loan-to-Income Ratio
  - Installment-to-Income Ratio
  - Revolving Balance-to-Income Ratio
  - Grade numeric encoding
- Stratified train-test split

---

## 🤖 Models Compared

| Model | ROC-AUC |
|--------|----------|
| Logistic Regression | 0.706 |
| Random Forest | 0.699 |
| XGBoost | 0.719 |

XGBoost provided the best ranking performance.

---

## 📈 Model Evaluation
- Class imbalance handled using scale_pos_weight
- Precision-Recall tradeoff analyzed
- Threshold tuning performed
- SHAP explainability used to interpret model predictions

---

## 🔎 Key Risk Drivers (SHAP Insights)
- Loan Grade
- Interest Rate
- Debt-to-Income Ratio
- Loan Term
- Credit Utilization Indicators

Higher grade levels, higher interest rates, and elevated DTI significantly increase default probability.

---

## 💰 Risk Scoring System
Predicted probability of default is converted into a risk score (0–100):

- 0–30 → Low Risk
- 30–60 → Medium Risk
- 60–100 → High Risk

---

## 🚀 Deployment
A Streamlit web application is built for real-time loan risk prediction.

---

## 🛠 Tech Stack
- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
