# ✅ Phase 4: Monitoring Dashboard
#-------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

# Initialize session state
if 'monitoring_data' not in st.session_state:
    st.session_state.monitoring_data = []

st.set_page_config(page_title="Stress Monitoring", layout="wide")
st.title("📈 Monitoring Dashboard: Workplace Stress Predictor")

# Load from CSV backup if empty
if not st.session_state.monitoring_data:
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "monitoring_logs.csv")
        if os.path.exists(csv_path):
            df_backup = pd.read_csv(csv_path)
            if not df_backup.empty:
                st.session_state.monitoring_data = df_backup.to_dict('records')
                st.success("✅ Loaded historical data from CSV.")
    except Exception as e:
        st.warning(f"Couldn't load backup: {str(e)}")

# Stop if still no data
if not st.session_state.monitoring_data:
    st.warning("⚠️ No monitoring data available. Please make some predictions first!")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(st.session_state.monitoring_data)

# Handle missing required columns
required_columns = {'timestamp', 'predicted_stress_level'}
if not required_columns.issubset(df.columns):
    st.error(f"Missing required columns: {required_columns - set(df.columns)}")
    st.stop()

# Timestamp conversion
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

# Optional user filtering if 'username' exists
if 'username' in df.columns:
    unique_users = df['username'].dropna().unique().tolist()
    selected_user = st.sidebar.selectbox("🔐 Filter by user (optional):", ["All"] + unique_users)

    if selected_user != "All":
        df = df[df['username'] == selected_user]
        st.info(f"🔍 Showing data for user: **{selected_user}**")

# 1. Stress Level Distribution
st.subheader("🧠 Stress Level Distribution")
fig1 = px.histogram(df, x="predicted_stress_level",
                    color="predicted_stress_level",
                    title="Distribution of Stress Levels")
st.plotly_chart(fig1)

# 2. Time Trend Analysis
st.subheader("📅 Stress Trends Over Time")
trend_df = df.groupby(['date', 'predicted_stress_level']).size().reset_index(name='count')
fig2 = px.line(trend_df, x='date', y='count',
               color='predicted_stress_level',
               title="Stress Levels by Date")
st.plotly_chart(fig2)

# 3. Age Distribution
if 'Age' in df.columns:
    st.subheader("📊 Age Distribution by Stress Level")
    fig3 = px.box(df, x="predicted_stress_level", y="Age",
                  title="Age Variation Across Stress Levels")
    st.plotly_chart(fig3)

# 4. Raw Data
with st.expander("🔍 View Raw Logs"):
    st.dataframe(df)
