# ✅ Phase 1: Data Gathering & Exploration
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("survey.csv")

# ------------------
# 1. Initial Exploration
# ------------------
print("Initial Shape:", df.shape)
print("Column Names:", df.columns.tolist())
print(df.info())

# ------------------
# 2. Remove Duplicate Records
# ------------------
df.drop_duplicates(inplace=True)

# ------------------
# 3. Filter Valid Age Range (Working adults only)
# ------------------
df = df[(df['Age'] >= 18) & (df['Age'] <= 60)]

# ------------------
# 4. Drop Irrelevant or Non-existent Columns
# ------------------
cols_to_drop = ['comments', 'state', 'timestamp']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# ------------------
# 5. Handle Missing Values
# ------------------
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])  # fill with mode

# ------------------
# 6. Normalize Categorical Text Inputs
# ------------------
df['Gender'] = df['Gender'].str.lower().str.strip()
def encode_gender(g):
    g = str(g).strip().lower()
    female_keywords = ['female', 'f', 'woman', 'cis female', 'trans-female']
    male_keywords = ['male', 'm', 'cis male', 'male-ish', 'mal', 'something kinda male']

    if any(k in g for k in female_keywords):
        return 'F'
    elif any(k in g for k in male_keywords):
        return 'M'
    else:
        return 'Other'  # or np.nan

df['Gender_clean'] = df['Gender'].apply(encode_gender)

print(df['Gender_clean'].value_counts())
# binary encoding gender
df['Gender_binary'] = df['Gender_clean'].map({'M': 1, 'F': 0})
print(df['Gender_binary'].value_counts())

# ------------------
# 7. Binary Encoding for Yes/No Columns
binary_cols = ['self_employed', 'family_history', 'treatment', 'Gender']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)


# ------------------
# 8. Encode Other Categorical Features
# ------------------
label_enc_cols = ['work_interfere', 'no_employees', 'remote_work', 'tech_company',
                  'benefits', 'coworkers', 'supervisor', 'treatment', 'self_employed', 'family_history', 'mental_vs_physical', 'anonymity'
                  ]

for col in label_enc_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# ------------------
# 9. Add mock GAD-7 scores for now (simulate user input)
# ------------------
df['gad7_score'] = np.random.randint(0, 22, size=len(df))  # simulate scores between 0–21

# ------------------
# 10. Create Target Label (stress_level) based on combined logic
# ------------------
def label_stress_combined(row):
    gad_score = row['gad7_score']
    work = row['work_interfere']  # 0=Never, 1=Sometimes, 2=Often, 3=Always

    # GAD-7 interpretation
    if gad_score >= 15:
        gad_level = 'severe'
    elif gad_score >= 10:
        gad_level = 'moderate'
    elif gad_score >= 5:
        gad_level = 'mild'
    else:
        gad_level = 'none'

    # Combine into stress_level
    if gad_level == 'severe' or work in [2, 3]:
        return 'high'
    elif gad_level == 'moderate' or work == 1:
        return 'moderate'
    else:
        return 'low'

df['stress_level'] = df.apply(label_stress_combined, axis=1)
# ------------------
# 11. Save Cleaned Data
# ------------------
df.to_csv("data/cleaned_survey.csv", index=False)
print("\nCleaned data saved to 'data/cleaned_survey.csv'")

#Automated EDA Report
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Survey Data Exploration Report", explorative=True)
profile.to_file("data_exploration_report.html")
