import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

# Initialize session state properly
if 'monitoring_data' not in st.session_state:
    st.session_state.monitoring_data = []

st.set_page_config(page_title="Stress Prediction Monitoring", layout="wide")
st.title("📈 Monitoring Dashboard: Workplace Stress Predictor")

# Try loading from CSV backup if session is empty
if not st.session_state.monitoring_data:
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "monitoring_logs.csv")
        if os.path.exists(csv_path):
            df_backup = pd.read_csv(csv_path)
            if not df_backup.empty:
                st.session_state.monitoring_data = df_backup.to_dict('records')
                st.success("Loaded historical data from backup")
    except Exception as e:
        st.warning(f"Couldn't load backup data: {str(e)}")

# Check if we have data to display
if not st.session_state.monitoring_data:
    st.warning("No monitoring data available yet. Please make some predictions first!")
    st.stop()

# Process the data
try:
    df = pd.DataFrame(st.session_state.monitoring_data)

    # Ensure required columns exist
    required_columns = {'timestamp', 'predicted_stress_level'}
    if not all(col in df.columns for col in required_columns):
        missing = required_columns - set(df.columns)
        st.error(f"Missing required data columns: {', '.join(missing)}")
        st.stop()

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # ===== Dashboard Visualizations =====
    # 1. Stress Level Distribution
    st.subheader("🧠 Stress Level Distribution")
    fig1 = px.histogram(df, x="predicted_stress_level",
                        color="predicted_stress_level",
                        title="Distribution of Stress Levels")
    st.plotly_chart(fig1)

    # 2. Time Trend Analysis
    st.subheader("📅 Stress Levels Over Time")
    trend_df = df.groupby(['date', 'predicted_stress_level']).size().reset_index(name='count')
    fig2 = px.line(trend_df, x='date', y='count',
                   color='predicted_stress_level',
                   title="Stress Trends Over Time")
    st.plotly_chart(fig2)

    # 3. Age Analysis
    if 'Age' in df.columns:
        st.subheader("📊 Age Distribution by Stress Level")
        fig3 = px.box(df, x="predicted_stress_level", y="Age",
                      title="Age Distribution Across Stress Levels")
        st.plotly_chart(fig3)

    # 4. Raw Data Explorer
    with st.expander("🔍 View Raw Data"):
        st.dataframe(df)

except Exception as e:
    st.error(f"Error processing data: {str(e)}")
    st.stop()