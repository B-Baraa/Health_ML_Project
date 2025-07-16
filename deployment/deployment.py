# ✅ Phase 4:Streamlit App with GAD-7 Integration
#-----------------------------------------------------
# deployment.py
#-----------------------------------------------------

import streamlit as st
import pandas as pd
import joblib

# Load model and label encoder
model = joblib.load("stress_prediction_model.pkl")
le = joblib.load("label_encoder.pkl")

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.set_page_config(page_title="Workplace Stress Predictor", layout="centered")
st.title("🤯 Workplace Stress Predictor")
st.markdown("Enter employee data to predict **stress risk** in the workplace.")

# Input form
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=75, value=30)
    self_employed = st.selectbox("Are you self-employed?", ['Yes', 'No'])
    family_history = st.selectbox("Do you have a family history of mental illness?", ['Yes', 'No'])
    treatment = st.selectbox("Have you sought treatment for a mental health condition?", ['Yes', 'No'])
    remote_work = st.selectbox("Do you work remotely (outside of an office) at least 50% of the time?", ['Yes', 'No'])
    tech_company = st.selectbox(" Is your employer primarily a tech company/organization?", ['Yes', 'No'])
    no_employees = st.selectbox("Company Size", ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'])
    work_interfere = st.selectbox("If you have a mental health condition, do you feel that it interferes with your work?", ['Never', 'Rarely', 'Sometimes', 'Often', 'Very often'])
    mental_vs_physical = st.selectbox("Do you feel that your employer takes mental health as seriously as physical health?", ['Do not know', 'No', 'Yes'])
    benefits = st.selectbox("Does your employer provide mental health benefits?", ['Yes', 'No'])
    anonymity = st.selectbox("Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?", ['Yes', 'No'])
    supervisor = st.selectbox("Would you be willing to discuss a mental health issue with your direct supervisor(s)?", ['Yes', 'No'])
    gad7_score = st.slider("Can you put a number for your Anxiety level from '0' to '20'?", min_value=0, max_value=20, value=10)
    submitted = st.form_submit_button("Predict Stress Level")

# ---------------------------------
# On Submit
# ---------------------------------
if submitted:
    work_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Very often': 4}
    size_map = {
        '1-5': 0, '6-25': 1, '26-50': 2,
        '50-200': 3, '200-500': 4, 'More than 1000': 5
    }

    input_dict = {
        'Age': age,
        'self_employed': 1 if self_employed == 'Yes' else 0,
        'family_history': 1 if family_history == 'Yes' else 0,
        'treatment': 1 if treatment == 'Yes' else 0,
        'remote_work': 1 if remote_work == 'Yes' else 0,
        'tech_company': 1 if tech_company == 'Yes' else 0,
        'mental_vs_physical': 1 if mental_vs_physical == 'Yes' else 0,
        'work_interfere': work_map[work_interfere],
        'benefits': 1 if benefits == 'Yes' else 0,
        'anonymity': 1 if anonymity == 'Yes' else 0,
        'no_employees': size_map[no_employees],
        'supervisor': 1 if supervisor == 'Yes' else 0,
        'gad7_score': gad7_score
    }

    input_df = pd.DataFrame([input_dict])

    # Predict
    pred_encoded = model.predict(input_df)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    # Output
    st.subheader("📊 Prediction Result")
    st.success(f"🎯 Predicted Stress Level: **{pred_label.upper()}**")
