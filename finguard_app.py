"""
FinGuard AI – Streamlit Web App
================================
Run with:  streamlit run finguard_app.py

Requirements:
    pip install streamlit scikit-learn pandas numpy joblib matplotlib seaborn

Make sure you have run finguard_model.py first so the  folder exists
with: finguard_model.pkl, scaler.pkl, feature_names.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .metric-card {
        background: rgba(30,41,59,0.8);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 12px; padding: 16px; text-align: center;
    }
    .risk-high   { color: #f43f5e; font-weight: bold; font-size: 1.4em; }
    .risk-medium { color: #f59e0b; font-weight: bold; font-size: 1.4em; }
    .risk-low    { color: #10b981; font-weight: bold; font-size: 1.4em; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
MODEL_DIR = "."

@st.cache_resource
def load_artifacts():
    model    = joblib.load(os.path.join(MODEL_DIR, "finguard_model.pkl"))
    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return model, scaler, features

try:
    model, scaler, feature_names = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error  = str(e)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
    st.title("FinGuard AI")
    st.caption("Credit Risk & Fraud Detection")
    st.divider()

    page = st.radio("Navigation", [
        "🏠 Home",
        "🔍 Credit Risk Predictor",
        "📊 Model Performance",
        "ℹ️ About",
    ])

    st.divider()
    st.markdown("**Dataset:** German Credit Risk (UCI/Kaggle)")
    st.markdown("**Model:** Random Forest (Tuned)")
    if model_loaded:
        st.success("✅ Model loaded")
    else:
        st.error("❌ Model not found – run finguard_model.py first")

# ══════════════ PAGE: Home ════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🛡️ FinGuard AI")
    st.subheader("Credit Risk Classification & Fraud Detection System")
    st.markdown("""
    Welcome to **FinGuard AI** – an end-to-end machine learning system that helps
    financial institutions assess credit risk and detect fraudulent applications.

    ---
    ### What this app does
    - Predicts whether a loan applicant is **good or bad credit risk**
    - Shows the **probability score** and key risk drivers
    - Trained on the **German Credit Risk dataset** (1,000 applicants, 20 features)

    ### How to use
    1. Go to **Credit Risk Predictor** in the sidebar
    2. Enter the applicant's details
    3. Click **Predict Risk** to get an instant decision

    ---
    ### Key Model Metrics
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  "~75–80%", help="On held-out test set")
    col2.metric("AUC-ROC",   "~0.80+",  help="Area under ROC curve")
    col3.metric("Precision", "~82%",    help="Precision for bad credit")
    col4.metric("F1-Score",  "~76%",    help="Harmonic mean P/R")

    st.info("ℹ️ Actual metrics depend on your dataset. Run finguard_model.py to train and see real numbers.")

# ══════════════ PAGE: Predictor ═══════════════════════════════════════════════
elif page == "🔍 Credit Risk Predictor":
    st.title("🔍 Credit Risk Predictor")
    st.markdown("Fill in the applicant's details below and click **Predict**.")

    if not model_loaded:
        st.error(f"Model not loaded: {model_error}")
        st.stop()

    with st.form("predictor_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Personal Info")
            age      = st.slider("Age",             18, 75, 35)
            job      = st.selectbox("Job Type",     [0, 1, 2, 3],
                                     format_func=lambda x: ["Unskilled (non-resident)",
                                                             "Unskilled (resident)",
                                                             "Skilled","Highly Skilled"][x])
            housing  = st.selectbox("Housing",      ["free", "own", "rent"])
            sex      = st.selectbox("Sex",          ["male", "female"])

        with col2:
            st.subheader("Financial Info")
            credit_amount = st.number_input("Credit Amount (DM)", 500, 20000, 3000, step=100)
            duration      = st.slider("Loan Duration (months)",    6, 72, 24)
            saving_accts  = st.selectbox("Saving Accounts",
                                          ["little", "moderate", "quite rich", "rich", "NA"])
            checking_acct = st.selectbox("Checking Account",
                                          ["little", "moderate", "rich", "NA"])

        with col3:
            st.subheader("Loan Details")
            purpose = st.selectbox("Loan Purpose", [
                "car", "furniture/equipment", "radio/TV",
                "domestic appliances", "repairs", "education",
                "business", "vacation/others"
            ])

        submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True)

    if submitted:
        # Build input row matching feature names
        enc = {
            "housing":       {"free": 0, "own": 1, "rent": 2},
            "saving_accts":  {"NA": 0, "little": 1, "moderate": 2, "quite rich": 3, "rich": 4},
            "checking_acct": {"NA": 0, "little": 1, "moderate": 2, "rich": 3},
            "purpose":       {"car": 0, "furniture/equipment": 1, "radio/TV": 2,
                               "domestic appliances": 3, "repairs": 4, "education": 5,
                               "business": 6, "vacation/others": 7},
            "sex":           {"male": 0, "female": 1},
        }

        row = {fn: 0 for fn in feature_names}
        mappings = {
            "Age": age,
            "Job": job,
            "Credit amount": credit_amount,
            "Duration": duration,
            "Housing": enc["housing"].get(housing, 0),
            "Saving accounts": enc["saving_accts"].get(saving_accts, 0),
            "Checking account": enc["checking_acct"].get(checking_acct, 0),
            "Purpose": enc["purpose"].get(purpose, 0),
            "Sex": enc["sex"].get(sex, 0),
        }
        # Fill derived features
        if "Monthly_Payment" in row:
            mappings["Monthly_Payment"] = credit_amount / max(duration, 1)
        if "Credit_per_Age" in row:
            mappings["Credit_per_Age"] = credit_amount / (age + 1)
        if "Total_Account_Strength" in row:
            mappings["Total_Account_Strength"] = (
                enc["saving_accts"].get(saving_accts, 0) +
                enc["checking_acct"].get(checking_acct, 0)
            )

        for k, v in mappings.items():
            if k in row:
                row[k] = v

        X_input = pd.DataFrame([row])[feature_names]
        prob    = model.predict_proba(X_input)[0][1]
        pred    = model.predict(X_input)[0]

        st.divider()
        st.subheader("📋 Prediction Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Score",    f"{prob*100:.1f}%")
        c2.metric("Decision",      "⛔ BAD CREDIT" if pred == 1 else "✅ GOOD CREDIT")
        c3.metric("Confidence",    f"{max(prob, 1-prob)*100:.1f}%")

        if prob > 0.65:
            st.error(f"🚨 **HIGH RISK** – Recommend: Decline application")
        elif prob > 0.35:
            st.warning(f"⚠️ **MEDIUM RISK** – Recommend: Manual review required")
        else:
            st.success(f"✅ **LOW RISK** – Recommend: Approve application")

        # Risk gauge bar
        st.markdown("**Risk Probability Gauge**")
        fig, ax = plt.subplots(figsize=(7, 1.2))
        ax.barh(0, 100, color="#1e293b", height=0.5)
        color = "#f43f5e" if prob > 0.65 else "#f59e0b" if prob > 0.35 else "#10b981"
        ax.barh(0, prob * 100, color=color, height=0.5)
        ax.set_xlim(0, 100); ax.axis("off")
        ax.text(prob * 100, 0, f" {prob*100:.1f}%", va="center", color="white", fontweight="bold")
        fig.patch.set_facecolor("#0f172a")
        st.pyplot(fig)


# ══════════════ PAGE: Model Performance ══════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    # Show saved plots if they exist
    plots = {
        "EDA Plots":          "eda_plots.png",
        "Evaluation Plots":   "evaluation_plots.png",
        "Feature Importance": "feature_importance.png",
    }

    for title, path in plots.items():
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, use_column_width=True)
        else:
            st.warning(f"'{path}' not found. Run finguard_model.py first.")

    # Show comparison table
    csv_path = "model_comparison.csv"
    if os.path.exists(csv_path):
        st.subheader("Model Comparison Table")
        df = pd.read_csv(csv_path, index_col=0)
        st.dataframe(df.style.highlight_max(color="#22c55e20", axis=0), use_container_width=True)


# ══════════════ PAGE: About ═══════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About FinGuard AI")
    st.markdown("""
    ### Project Summary
    **FinGuard AI** is a machine learning mini-project that demonstrates an end-to-end
    AI pipeline for credit risk classification.

    ### Dataset
    - **Name:** German Credit Risk Dataset
    - **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/german-credit) / [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
    - **Size:** 1,000 records, 20 features
    - **Task:** Binary classification – Good vs. Bad credit risk

    ### Models Trained
    | Model | Notes |
    |---|---|
    | Logistic Regression | Baseline linear model |
    | Decision Tree | Interpretable rules |
    | Random Forest | Ensemble – best performer |
    | Gradient Boosting | Sequential boosting |

    ### Pipeline Steps
    1. Problem Definition
    2. Data Acquisition (German Credit Dataset)
    3. Data Cleaning & Preprocessing
    4. Exploratory Data Analysis
    5. Feature Engineering
    6. Model Building & Hyperparameter Tuning
    7. Model Evaluation
    8. Results & Insights
    9. Deployment (this app)
    10. Documentation

    ### Team
    Group Project – AI Mini Project

    ### Tech Stack
    `Python` · `scikit-learn` · `pandas` · `matplotlib` · `seaborn` · `Streamlit`
    """)
