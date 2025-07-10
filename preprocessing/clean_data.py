# ✅ Phase 1: Data Gathering & Exploration

# 🔹 1. Load the Data
import pandas as pd

# Load the dataset
df = pd.read_csv('survey.csv')

# Check basic structure
print(df.head())
print(df.info())
print(df.shape)

# 🔹 2. Initial Data Profiling
# Check for duplicates
print("Duplicate rows:", df.duplicated().sum())

# Count missing values
print("Missing values per column:\n", df.isnull().sum())

# Basic statistics for numerical features
print(df.describe())

# 🔹 3. Data Cleaning
# Normalize inconsistent text inputs (example: Gender column)
df['Gender'] = df['Gender'].str.lower().str.strip()

# Drop rows with missing values (if necessary)
df = df.dropna()

# Drop irrelevant or redundant columns
df = df.drop(['comments', 'state', 'Timestamp'], axis=1)

# Outlier detection using boxplot
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data=df, x='Age')
plt.title('Age Outlier Detection')
plt.show()

# Remove unrealistic ages
df = df[(df['Age'] >= 18) & (df['Age'] <= 60)]

# 🔹 4. Data Filtering
# Filter based on employment status if necessary
# Example: Remove students/unemployed if irrelevant
if 'self_employed' in df.columns:
    df = df[df['self_employed'].notnull()]

# 🔹 5. Feature Engineering
# Label Encoding for categorical features
from sklearn.preprocessing import LabelEncoder

df['work_interfere'] = LabelEncoder().fit_transform(df['work_interfere'].astype(str))

# Create a target variable: stress_level
def label_stress(row):
    if row['work_interfere'] in [2, 3]:  # Assuming 2 = Often, 3 = Always
        return 'high'
    elif row['work_interfere'] == 1:     # 1 = Sometimes
        return 'moderate'
    else:                                # 0 = Never or missing
        return 'low'

df['stress_level'] = df.apply(label_stress, axis=1)

# 🔹 6. Correlation & Initial Insights
# Visualize correlation matrix
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Explore group means by stress level
print(df.groupby('stress_level').mean(numeric_only=True))

# 🔹 7. Save Cleaned Data
import os

# Create 'data' directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save Cleaned Data
df.to_csv('data/cleaned_survey.csv', index=False)
print("✅ Cleaned data saved to 'data/cleaned_survey.csv'")
df.to_csv('data/cleaned_survey.csv', index=False)
print("Cleaned data saved to 'data/cleaned_survey.csv'")

#Automated EDA Report
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Survey Data Exploration Report", explorative=True)
profile.to_file("data_exploration_report.html")
