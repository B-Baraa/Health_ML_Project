# 🤯 Workplace Stress Predictor

**Workplace Stress Predictor** is an intelligent web app designed to estimate employee stress levels using a combination of mental health self-assessments and workplace factors.
It leverages machine learning and a customized scoring system to provide accurate, authenticated and actionable insights.

---

## 🚀 Live Demo

🔗 **App**: [Workplace Stress Predictor](https://healthmlproject-qqspnfh2ftyyweiebygvy8.streamlit.app/)  
📊 **Monitoring Dashboard**: [View Logs & Insights](https://healthmlproject-gvesg7jnuqmqcsrwssbtf3.streamlit.app/)

---

## 🧠 How It Works

The app uses:
- The user enters its username
- Fill out the form and Submit the answers
- **GAD-7 anxiety score** (self-reported)
- **Machine learning-based stress prediction** ratings
- **Workplace conditions questionnaire**

Each factor is weighted:
- 🧠 GAD-7 → 35%
- 🏢 Stress Prediction → 55%
- 📋 Questionnaire → 10%

Final stress levels are:
- **Low**
- **Moderate**
- **High**

All predictions are logged for transparency and future optimization.

---

## 📊 Features

- ✅ Simple GAD-7 slider input
- ✅ Questionnaire about workplace mental health practices
- ✅ Real-time machine learning-based stress prediction
- ✅ Logging system for monitoring usage
- ✅ Visual monitoring dashboard (via Streamlit)

---

## 🧰 Technologies Used

| Tool           | Purpose                          |
|----------------|----------------------------------|
| Python         | Programming                      |
| Streamlit      | Web application framework        |
| Scikit-learn   | Machine learning training        |
| Pandas         | Data manipulation                |
| Joblib         | Model persistence                |
| CSV Logging    | Lightweight monitoring           |
| numpy          | Numerical computing&data handling|
| seaborn        | Statistical data visualization	
---

## 📁 Project Structure

```bash
health_ml_project/
│
├── preprocessing/
│  └── clean_data.py           # Data cleaning scripts
│
├── visualization/                   
│   └── visual.py              # Data visualization scripts
│
├── training and testing/
│   └── traintest.py           # Data training scripts
│
├── deployment/
│   ├── deployment.py          # Streamlit app logic
│   ├── stress_prediction_model.pkl
│   ├── label_encoder.pkl
│   ├── requirements.txt
│   ├── monitoring.py          # Streamlit dashboard for logs
│   └── monitoring_logs.csv    # Auto-generated log file
│         
│
└── README.md

🔍 Monitoring Strategy
-All form submissions are stored in monitoring_logs.csv for local use
then st.session_state.monitoring_logs for streamlit cloud.
-The dashboard displays log trends and frequencies.
-Useful for tracking model use and detecting anomalies.

📈 Model Details
-Model: Random Forest Classifier
-Optimization: RandomizedSearchCV
-Labeling strategy: Weighted sum of anxiety + work interference + workplace conditions
-Accuracy: ~85% F1-weighted on test data

👨‍💻 Author
Built by Baraa Bouchiba
🔗(https://github.com/B-Baraa)

📌 Future Improvements
-Model retraining pipeline
-Advanced dashboards (e.g., user segmentation)
-Email alerts for high stress patterns

⚠️ Disclaimer
This tool is not a medical diagnostic tool. It is intended for educational, workplace awareness, and research support only.


