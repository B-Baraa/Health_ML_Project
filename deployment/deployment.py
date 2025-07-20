# ✅ Phase 4:Streamlit App with GAD-7 Integration
#-----------------------------------------------------
# ===== deployment.py =====
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

# --- Simple Login ---
st.set_page_config(page_title="Workplace Stress Predictor", layout="centered")

if "user_logged_in" not in st.session_state:
    st.session_state["user_logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

if not st.session_state["user_logged_in"]:
    st.title("🔐 Login to Access Stress Predictor")
    username_input = st.text_input("Enter your name to continue:")

    if st.button("Login"):
        if username_input.strip():  # Check it's not empty
            st.session_state["user_logged_in"] = True
            st.session_state["username"] = username_input.strip()
            st.success(f"Welcome, {username_input} 👋")
            st.rerun()
        else:
            st.warning("Please enter a valid name to continue.")
    st.stop()
# ✅ After login
st.sidebar.success(f"✅ Logged in as: {st.session_state['username']}")

# 🔓 Logout Button
if st.sidebar.button("🚪 Logout"):
    st.session_state["user_logged_in"] = False
    st.session_state["username"] = ""
    st.rerun()


# Initialize session state
if 'monitoring_data' not in st.session_state:
    st.session_state.monitoring_data = []

MODEL_DIR = Path(__file__).parent
model_path = MODEL_DIR / "stress_prediction_model.pkl"
le_path = MODEL_DIR / "label_encoder.pkl"
# Load model
try:
    model = joblib.load(model_path)
    le = joblib.load(le_path)
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}")
    st.stop()
# Initialize session state for cross-page sharing
if 'monitoring_data' not in st.session_state:
    st.session_state.monitoring_data = []

# Initialize session state for logs if it doesn't exist
if 'monitoring_logs' not in st.session_state:
    st.session_state.monitoring_logs = []
# Page setup
st.set_page_config(page_title="Workplace Stress Predictor", layout="centered")
st.title("🤯 Workplace Stress Predictor")
st.markdown("Enter employee data to predict **stress risk** in the workplace.")

# Questionnaire weights
questionnaire_weights = {
    'self_employed': 0.05,
    'family_history': 0.1,
    'treatment': 0.1,
    'remote_work': 0.0,
    'tech_company': 0.05,
    'mental_vs_physical': 0.1,
    'benefits': 0.1,
    'anonymity': 0.1,
    'no_employees': 0.05,
    'Age': 0.05
}

# Input form
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=75, value=30)
    self_employed = st.selectbox("Are you self-employed?", ['Yes', 'No'])
    family_history = st.selectbox("Do you have a family history of mental illness?", ['Yes', 'No'])
    treatment = st.selectbox("Have you sought treatment for a mental health condition?", ['Yes', 'No'])
    remote_work = st.selectbox("Do you work remotely (outside of an office) at least 50% of the time?", ['Yes', 'No'])
    tech_company = st.selectbox("Is your employer primarily a tech company/organization?", ['Yes', 'No'])
    mental_vs_physical = st.selectbox("Does your employer take mental health as seriously as physical health?", ['Do not know', 'No', 'Yes'])
    work_interfere = st.selectbox("If you have a mental health condition, do you feel it interferes with your worK?", ['Never', 'Rarely', 'Sometimes', 'Often', 'Very often'])
    benefits = st.selectbox("Does your employer provide mental health benefits?", ['Yes', 'No'])
    anonymity = st.selectbox("Is your anonymity protected when seeking treatment?", ['Yes', 'No'])
    no_employees = st.selectbox("Company Size", ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'])
    gad7_score = st.slider("GAD-7 Anxiety Score (0–21)", min_value=0, max_value=21, value=10)
    submitted = st.form_submit_button("Predict Stress Level")

if submitted:
    # Feature mappings
    work_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Very often': 4}
    size_map = {'1-5': 0, '6-25': 1, '26-100': 2, '100-500': 3, '500-1000': 4, 'More than 1000': 5}
    mvp_map = {'Yes': 1, 'No': 0, 'Do not know': 0}

    # Prepare input
    input_dict = {
        'Age': age,
        'self_employed': 1 if self_employed == 'No' else 0,
        'family_history': 1 if family_history == 'Yes' else 0,
        'treatment': 1 if treatment == 'Yes' else 0,
        'remote_work': 1 if remote_work == 'No' else 0,
        'tech_company': 1 if tech_company == 'Yes' else 0,
        'mental_vs_physical': mvp_map[mental_vs_physical],
        'work_interfere': work_map[work_interfere],
        'benefits': 1 if benefits == 'No' else 0,
        'anonymity': 1 if anonymity == 'No' else 0,
        'no_employees': size_map[no_employees],
        'gad7_score': gad7_score
    }

    # Model prediction
    model_input = input_dict.copy()
    model_input.pop('gad7_score')
    pred_encoded = model.predict(pd.DataFrame([model_input]))[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    # Questionnaire Score (10%)
    questionnaire_score = 0
    for feature, weight in questionnaire_weights.items():
        val = input_dict[feature]
        if feature == 'Age':
            questionnaire_score += (val / 100) * weight  # Normalize age
        else:
            questionnaire_score += val * weight

    # Composite Score Calculation
    gad7_weighted = (gad7_score / 21) * 0.35
    work_weighted = (work_map[work_interfere] / 4) * 0.55
    questionnaire_weighted = questionnaire_score * 0.10
    total_score = gad7_weighted + work_weighted + questionnaire_weighted

    # Determine final stress label
    if total_score < 0.30:
        final_label = "low"
    elif total_score < 0.60:
        final_label = "moderate"
    else:
        final_label = "high"
# -----------------------------------------------------
# ===== monitoring session state logging =====
    # Append to monitoring logs
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            **input_dict,
            "predicted_stress_level": final_label
        }

        # Append to session state
        st.session_state.monitoring_data = st.session_state.monitoring_data + [log_entry]

        # Save to Streamlit's persistent storage
        st.session_state['monitoring_data'] = st.session_state.monitoring_data

        # Save to CSV for persistence
        log_path = Path(__file__).parent / "monitoring_logs.csv"
        log_entry_df = pd.DataFrame([log_entry])
        log_entry_df.to_csv(log_path, mode='a', header=not log_path.exists(), index=False)

        st.success("Prediction saved successfully!")

    # Display results
    st.subheader("📊 Prediction Result")
    st.success(f"✅ **Final Stress Level:** {final_label.upper()}")
    st.markdown(f"🎯 **Model Prediction:** {pred_label.upper()}")
    st.markdown(f"🧠 **GAD-7 Score:** {gad7_score}/21")
    st.markdown(f"🧾 **Questionnaire Score:** {round(questionnaire_score, 2)}")