import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Risk Engine",
    page_icon="💰",
    layout="wide"
)

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        border: 1px solid #2d3250;
    }
    .metric-label {
        font-size: 13px;
        color: #8b9ab0;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        margin: 0;
    }
    .risk-low    { color: #2ecc71; }
    .risk-medium { color: #f39c12; }
    .risk-high   { color: #e74c3c; }
    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #8b9ab0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 20px 0 8px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #2d3250;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 8px;
        font-size: 15px;
        font-weight: 600;
        margin-top: 16px;
        cursor: pointer;
    }
    .stButton > button:hover {
        opacity: 0.9;
    }
    .insight-box {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        border-left: 4px solid #667eea;
    }
    div[data-testid="stSidebar"] {
        background-color: #161a2b;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Load model
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("model/xgb_loan_model.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model()

# ─────────────────────────────────────────
# Sidebar — grouped inputs
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💰 Loan Default Risk Engine")
    st.markdown("Fill in borrower details to predict default probability.")
    st.markdown("---")

    st.markdown('<div class="section-header">📋 Loan Information</div>', unsafe_allow_html=True)
    loan_amnt = st.number_input("Loan Amount ($)", 1000, 1000000, 10000, step=500)
    term = st.selectbox("Loan Term", [36, 60], format_func=lambda x: f"{x} months")
    int_rate = st.slider("Interest Rate (%)", 1.0, 40.0, 12.0, step=0.1)
    installment = st.number_input("Monthly Installment ($)", 50.0, 5000.0, 300.0, step=10.0)
    purpose = st.selectbox("Loan Purpose", [
        "debt_consolidation", "credit_card", "home_improvement",
        "other", "major_purchase", "small_business", "car",
        "medical", "moving", "vacation", "house", "wedding", "educational"
    ])

    st.markdown('<div class="section-header">👤 Borrower Profile</div>', unsafe_allow_html=True)
    annual_inc = st.number_input("Annual Income ($)", 5000, 2000000, 60000, step=1000)
    emp_length = st.slider("Employment Length (Years)", 0, 40, 5)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    verification_status = st.selectbox("Income Verification", ["Verified", "Source Verified", "Not Verified"])
    addr_state = st.selectbox("State", [
        "CA","NY","TX","FL","IL","PA","OH","GA","NC","MI",
        "NJ","VA","WA","AZ","MA","CO","TN","MO","MD","IN",
        "WI","MN","SC","AL","LA","KY","OR","OK","CT","UT",
        "AR","MS","KS","NV","NM","NE","WV","ID","HI","NH",
        "ME","RI","MT","DE","SD","ND","AK","VT","WY","DC"
    ])

    st.markdown('<div class="section-header">💳 Credit History</div>', unsafe_allow_html=True)
    grade = st.select_slider("Loan Grade", options=["A","B","C","D","E","F","G"],
                              help="A = safest, G = riskiest")
    dti = st.slider("Debt-to-Income Ratio", 0.0, 60.0, 15.0, step=0.1)
    revol_util = st.slider("Revolving Utilization (%)", 0.0, 150.0, 30.0, step=0.1)
    revol_bal = st.number_input("Revolving Balance ($)", 0, 500000, 10000, step=500)
    open_acc = st.number_input("Open Credit Accounts", 0, 50, 8)
    total_acc = st.number_input("Total Credit Accounts", 0, 100, 15)
    delinq_2yrs = st.number_input("Delinquencies (Last 2 Years)", 0, 20, 0)
    pub_rec = st.number_input("Public Records", 0, 10, 0)
    mort_acc = st.number_input("Mortgage Accounts", 0, 20, 0)
    pub_rec_bankruptcies = st.number_input("Bankruptcies", 0, 5, 0)

    predict_btn = st.button("🔍 Predict Default Risk")

# ─────────────────────────────────────────
# Main area — header
# ─────────────────────────────────────────
st.markdown("# 💰 Loan Default Risk Engine")
st.markdown("AI-powered credit risk assessment using XGBoost trained on 1.3M+ Lending Club loans.")
st.markdown("---")

# ─────────────────────────────────────────
# Feature engineering helper
# ─────────────────────────────────────────
def build_input():
    grade_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
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
        "total_acc": total_acc,
        "mort_acc": mort_acc,
        "pub_rec_bankruptcies": pub_rec_bankruptcies,
        "loan_to_income": loan_amnt / (annual_inc + 1),
        "installment_to_income": installment / (annual_inc / 12 + 1),
        "revol_bal_to_income": revol_bal / (annual_inc + 1),
        "grade_num": grade_map[grade],
        "home_ownership": home_ownership,
        "verification_status": verification_status,
        "purpose": purpose,
        "addr_state": addr_state,
    }
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
    return input_df

# ─────────────────────────────────────────
# Gauge chart
# ─────────────────────────────────────────
def draw_gauge(score):
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(theta, [1]*200, color='#2d3250', linewidth=18, solid_capstyle='round')

    # Colored zones
    zones = [
        (np.linspace(np.pi, np.pi*0.67, 100), '#2ecc71'),
        (np.linspace(np.pi*0.67, np.pi*0.33, 100), '#f39c12'),
        (np.linspace(np.pi*0.33, 0, 100), '#e74c3c'),
    ]
    for zone_theta, color in zones:
        ax.plot(zone_theta, [1]*100, color=color, linewidth=18, alpha=0.3, solid_capstyle='butt')

    # Score needle
    score_clamped = max(0, min(100, score))
    needle_angle = np.pi * (1 - score_clamped / 100)
    ax.plot([needle_angle, needle_angle], [0, 0.85], color='white', linewidth=3)
    ax.plot(needle_angle, 0, 'o', color='white', markersize=8)

    # Score fill arc
    fill_end = np.pi * (1 - score_clamped / 100)
    fill_theta = np.linspace(np.pi, fill_end, 200)
    if score_clamped < 30:
        fill_color = '#2ecc71'
    elif score_clamped < 60:
        fill_color = '#f39c12'
    else:
        fill_color = '#e74c3c'
    ax.plot(fill_theta, [1]*200, color=fill_color, linewidth=18, solid_capstyle='butt')

    ax.set_ylim(0, 1.3)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.axis('off')

    ax.text(np.pi/2, 0.15, f"{score:.1f}", ha='center', va='center',
            fontsize=28, fontweight='bold', color='white',
            transform=ax.transData)

    ax.text(np.pi*1.0, 1.25, 'Low', ha='center', va='center',
            fontsize=9, color='#2ecc71')
    ax.text(np.pi*0.5, 1.25, 'Medium', ha='center', va='center',
            fontsize=9, color='#f39c12')
    ax.text(np.pi*0.05, 1.25, 'High', ha='center', va='center',
            fontsize=9, color='#e74c3c')

    plt.tight_layout()
    return fig

# ─────────────────────────────────────────
# Default state — show instructions
# ─────────────────────────────────────────
if not predict_btn:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value" style="font-size:20px; color:#667eea;">XGBoost</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Training Records</div>
            <div class="metric-value" style="font-size:20px; color:#667eea;">1.3M+</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">ROC-AUC</div>
            <div class="metric-value" style="font-size:20px; color:#667eea;">0.7203</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Fill in the borrower details in the sidebar and click **Predict Default Risk** to get a risk assessment.")

# ─────────────────────────────────────────
# Prediction output
# ─────────────────────────────────────────
else:
    input_df = build_input()
    probability = model.predict_proba(input_df)[0][1]
    risk_score = round(probability * 100, 1)

    if risk_score < 30:
        category = "Low Risk"
        cat_color = "risk-low"
        cat_emoji = "✅"
        interpretation = "This borrower profile presents a low probability of default. Strong indicators include manageable debt levels and favorable credit grade."
    elif risk_score < 60:
        category = "Medium Risk"
        cat_color = "risk-medium"
        cat_emoji = "⚠️"
        interpretation = "This borrower profile presents moderate default risk. Some risk factors are present — careful review of debt-to-income ratio and credit history is recommended."
    else:
        category = "High Risk"
        cat_color = "risk-high"
        cat_emoji = "🚨"
        interpretation = "This borrower profile presents elevated default risk. Key risk drivers likely include high DTI, unfavorable loan grade, or high revolving utilization."

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Score</div>
            <div class="metric-value {cat_color}">{risk_score}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Default Probability</div>
            <div class="metric-value {cat_color}">{probability:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Category</div>
            <div class="metric-value {cat_color}" style="font-size:22px;">{cat_emoji} {category}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        loan_to_inc = round(loan_amnt / (annual_inc + 1) * 100, 1)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Loan-to-Income</div>
            <div class="metric-value" style="color:#667eea;">{loan_to_inc}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge + interpretation
    col_gauge, col_info = st.columns([1, 1])
    with col_gauge:
        st.markdown("#### Risk Gauge")
        fig = draw_gauge(risk_score)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_info:
        st.markdown("#### Assessment Summary")
        st.markdown(f"""
        <div class="insight-box">
            <strong>{cat_emoji} {category}</strong><br><br>
            {interpretation}
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Key Input Summary")
        summary_data = {
            "Metric": ["Loan Amount", "Annual Income", "Interest Rate", "DTI Ratio", "Loan Grade", "Revolving Util."],
            "Value": [
                f"${loan_amnt:,}",
                f"${annual_inc:,}",
                f"{int_rate}%",
                f"{dti}%",
                grade,
                f"{revol_util}%"
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

    # Risk factors breakdown
    st.markdown("---")
    st.markdown("#### 📊 Risk Factor Analysis")

    factors = {
        "Interest Rate": min(int_rate / 40, 1.0),
        "Debt-to-Income Ratio": min(dti / 60, 1.0),
        "Revolving Utilization": min(revol_util / 150, 1.0),
        "Loan-to-Income": min((loan_amnt / (annual_inc + 1)) / 0.5, 1.0),
        "Loan Grade Risk": (ord(grade) - ord('A')) / 6,
        "Delinquency History": min(delinq_2yrs / 5, 1.0),
    }

    col_f1, col_f2 = st.columns(2)
    factor_items = list(factors.items())
    for i, (factor, value) in enumerate(factor_items):
        col = col_f1 if i < 3 else col_f2
        with col:
            level = "Low" if value < 0.4 else "Medium" if value < 0.7 else "High"
            bar_color = "#2ecc71" if value < 0.4 else "#f39c12" if value < 0.7 else "#e74c3c"
            bar_pct = int(value * 100)
            st.markdown(f"""
            <div style="margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="font-size:13px; color:#c9d1d9;">{factor}</span>
                    <span style="font-size:12px; color:{bar_color};">{level}</span>
                </div>
                <div style="background:#2d3250; border-radius:4px; height:8px;">
                    <div style="background:{bar_color}; width:{bar_pct}%; height:8px; border-radius:4px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Model: XGBoost · Trained on 1.3M+ Lending Club loans · ROC-AUC: 0.7203 · Built by Azim Sadath")
