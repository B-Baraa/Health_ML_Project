# monitoring.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Stress Prediction Monitoring", layout="wide")
st.title("📈 Monitoring Dashboard: Workplace Stress Predictor")

# Load monitoring logs
LOG_PATH = "monitoring_logs.csv"

if not os.path.exists(LOG_PATH):
    st.warning("No monitoring logs found yet. Make some predictions first!")
    st.stop()

df = pd.read_csv(LOG_PATH, parse_dates=['timestamp'])

# Filters
with st.sidebar:
    st.header("📊 Filters")
    stress_filter = st.multiselect("Select Stress Levels:", options=df["predicted_stress_level"].unique(), default=df["predicted_stress_level"].unique())
    df_filtered = df[df["predicted_stress_level"].isin(stress_filter)]

# 1. Count of stress levels
st.subheader("🧠 Stress Level Distribution")
st.plotly_chart(
    px.histogram(df_filtered, x="predicted_stress_level", color="predicted_stress_level", barmode="group")
)

# 2. Time trend
st.subheader("📅 Stress Levels Over Time")
df_filtered['date'] = df_filtered['timestamp'].dt.date
trend = df_filtered.groupby(['date', 'predicted_stress_level']).size().reset_index(name='count')
fig_trend = px.line(trend, x='date', y='count', color='predicted_stress_level', markers=True)
st.plotly_chart(fig_trend)

# 3. Age vs Stress Level
st.subheader("📊 Age Distribution by Stress Level")
fig_age = px.box(df_filtered, x="predicted_stress_level", y="Age", color="predicted_stress_level")
st.plotly_chart(fig_age)

# 4. Remote work and work interference impact
st.subheader("🏠 Remote Work & Work Interference Impact")
col1, col2 = st.columns(2)

with col1:
    remote_fig = px.histogram(df_filtered, x="remote_work", color="predicted_stress_level", barmode="group")
    st.plotly_chart(remote_fig)

with col2:
    interfere_fig = px.histogram(df_filtered, x="work_interfere", color="predicted_stress_level", barmode="group")
    st.plotly_chart(interfere_fig)

# Optional: Display raw log data
with st.expander("📄 View Raw Log Data"):
    st.dataframe(df_filtered)
