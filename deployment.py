# ✅ Phase 4:Streamlit App with GAD-7 Integration
#-----------------------------------------------------
# deployment.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and label encoder
model = joblib.load("stress_prediction_model.pkl")
le = joblib.load("label_encoder.pkl")
import os
from pathlib import Path

# Path(__file__) to get the script's directory
MODEL_PATH = Path(__file__).parent / "stress_prediction_model.pkl"
model = joblib.load(MODEL_PATH)
# Set page
st.set_page_config(page_title="Workplace Stress Predictor", layout="centered")
st.title("🤯 Workplace Stress Predictor")
st.markdown("Enter employee data to predict **stress risk** in the workplace.")

# Questionnaire Scoring Weights based on feature importance)
questionnaire_weights = {
    'self_employed': 0.05,
    'family_history': 0.1,
    'treatment': 0.1,
    'remote_work': 0.5,
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
    work_interfere = st.selectbox("If you have a mental health condition, do you feel it interferes with your work?", ['Never', 'Rarely', 'Sometimes', 'Often', 'Very often'])
    benefits = st.selectbox("Does your employer provide mental health benefits?", ['Yes', 'No'])
    anonymity = st.selectbox("Is your anonymity protected when seeking treatment?", ['Yes', 'No'])
    no_employees = st.selectbox("Company Size", ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'])
    gad7_score = st.slider("GAD-7 Anxiety Score (0–21)", min_value=0, max_value=21, value=10)
    submitted = st.form_submit_button("Predict Stress Level")


if submitted:
    # Mappings
    work_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Very often': 4}
    size_map = {
        '1-5': 0, '6-25': 1, '26-100': 2, '100-500': 3, '500-1000': 4, 'More than 1000': 5
    }
    mvp_map = {'Yes': 1, 'No': 0, 'Do not know': 0}

    # Input features
    input_dict = {
        'Age': age,
        'self_employed': 1 if self_employed == 'No' else 0,
        'family_history': 1 if family_history == 'Yes' else 0,
        'treatment': 1 if treatment == 'Yes' else 0,
        'remote_work': 1 if remote_work == 'Yes' else 0,
        'tech_company': 1 if tech_company == 'Yes' else 0,
        'mental_vs_physical': 1 if mental_vs_physical == 'No' else 0,
        'work_interfere': work_map[work_interfere],
        'benefits': 1 if benefits == 'No' else 0,
        'anonymity': 1 if anonymity == 'No' else 0,
        'no_employees': size_map[no_employees],
        'gad7_score': gad7_score
    }

    # Model input (exclude gad7_score and remote_work if not used in training)
    model_input = input_dict.copy()
    model_input.pop('gad7_score')
    input_df = pd.DataFrame([model_input])

    # Prediction
    pred_encoded = model.predict(input_df)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    # Questionnaire Score (20%)
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
    # ---------------------------------------------------------
    # Monitoring
    import csv
    from datetime import datetime
    import os

    # 1. Define log file path
    log_file = "monitoring_logs.csv"

    # 2. Prepare row to log
    log_data = {
        "timestamp": datetime.now().isoformat(),
        **input_dict,
        "predicted_stress_level": final_label
    }

    # 3. Write log entry
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)

        # 4. Display prediction and details
        st.subheader("📊 Prediction Result")
        st.success(f"✅ **Final Stress Level:** {final_label.upper()}")
        st.markdown(f"🎯 **Model Prediction:** {pred_label.upper()}")
        st.markdown(f"🧠 **GAD-7 Score:** {gad7_score}/21")
        st.markdown(f"🧾 **Questionnaire Score:** {round(questionnaire_score, 2)}")



