# ✅ Phase 2: Workplace Stress Analysis Visualization and model selection Suite
#---------------------------------------------
#visualization
#---------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
df = pd.read_csv("cleaned_survey.csv")
sns.set(style="whitegrid")
#--------------------------------------------------
# 1. STRESS LEVEL DISTRIBUTION
# Purpose: Show overall prevalence of stress levels in the workforce
plt.figure(figsize=(6, 6))
stress_counts = df['stress_level'].value_counts()
plt.pie(stress_counts, labels=stress_counts.index, autopct='%1.1f%%',
        colors=sns.color_palette("pastel"))
plt.title('Overall Stress Level Distribution in Workforce')
plt.show()

# 2. AGE ANALYSIS FOR HIGH-STRESS INDIVIDUALS
# Purpose: Identify if specific age groups are more vulnerable to high stress
plt.figure(figsize=(8, 5))
sns.histplot(df[df['stress_level'] == 'high']['Age'], bins=20,
             color='crimson', kde=True)
plt.title('Age Distribution of High-Stress Employees')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
#-----------------------------------------------
# 3.Subplots: Distribution of Features by Stress Level
import numpy as np
import re
def encode_no_employees(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if val == "5":
        return '<1000'
    else:
        # Everything else (ranges) is >1000
        return '>1000'

df['no_employees_cat'] = df['no_employees'].apply(encode_no_employees)

selected_features = ['supervisor', 'treatment', 'Gender_clean',
                     'family_history', 'no_employees_cat', 'benefits']

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

for i, feature in enumerate(selected_features):
    sns.countplot(data=df, x=feature, hue='stress_level',
                  palette='Set2', ax=axes[i])
    axes[i].set_title(f'{feature} Distribution of level-Stress Employees')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Count')
    axes[i].legend(title='Stress Level')
    axes[i].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()


#--------------------------------------------------
# 4. EMPLOYMENT TYPE VS STRESS (Tech vs Non-Tech)
# Purpose: Compare stress levels between tech and non-tech workers
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='stress_level', hue='tech_company',
              palette='coolwarm', order=['low', 'moderate', 'high'],
              hue_order=[0, 1])  # 0=Non-tech, 1=Tech

plt.title('Stress Levels by Employment Sector')
plt.xlabel('Stress Level')
plt.ylabel('Count')
plt.legend(title='Employment Type',
           labels=['Non-Tech Sector', 'Tech Company Employee'])

# Add percentage annotations
for i, level in enumerate(['low', 'moderate', 'high']):
    total = df[df['stress_level'] == level].shape[0]
    for j, tech in enumerate([0, 1]):
        count = df[(df['stress_level'] == level) & (df['tech_company'] == tech)].shape[0]
        plt.text(i + (j * 0.2 - 0.1), count + 5, f'{count / total * 100:.1f}%',
                 ha='center', color='black')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
#----------------------------------------------------------------
# 5. REMOTE WORK IMPACT ON STRESS
# Purpose: Examine if remote work correlates with stress levels
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='stress_level', hue='remote_work',
              palette='Greens', order=['low', 'moderate', 'high'])
plt.title('Stress Levels by Remote Work Arrangement')
plt.xlabel('Stress Level')
plt.ylabel('Count')
plt.legend(title='Remote Work?', labels=['On-Site', 'Remote'])

# Add remote work percentage annotations
for i, level in enumerate(['low', 'moderate', 'high']):
    remote = df[(df['stress_level'] == level) & (df['remote_work'] == 1)].shape[0]
    total = df[df['stress_level'] == level].shape[0]
    plt.text(i, total + 10, f'{remote / total * 100:.1f}% remote',
             ha='center', color='black')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
#--------------------------------------------------
# 6. HEALTH DISCLOSURE WILLINGNESS COMPARISON
# Purpose: Reveal stigma differences between mental/physical health
# Prepare data
mh_counts = df['mental_health_interview'].value_counts(normalize=True) * 100
ph_counts = df['phys_health_interview'].value_counts(normalize=True) * 100
interview_pct = pd.DataFrame({
    'Mental Health': mh_counts,
    'Physical Health': ph_counts
}).fillna(0).reindex(['Yes', 'Maybe', 'No'])

# Create dot plot comparison
plt.figure(figsize=(10, 4))
for i, response in enumerate(['Yes', 'Maybe', 'No']):
    # Mental health (salmon)
    plt.scatter(['Mental Health'], [i],
                s=interview_pct.loc[response, 'Mental Health'] * 5,
                color='salmon', alpha=0.7,
                label='Mental' if i == 0 else "")

    # Physical health (skyblue)
    plt.scatter(['Physical Health'], [i],
                s=interview_pct.loc[response, 'Physical Health'] * 5,
                color='skyblue', alpha=0.7,
                label='Physical' if i == 0 else "")

    # Connecting lines
    plt.plot([0, 1], [i, i], color='gray', linestyle='--', alpha=0.5)

plt.yticks([0, 1, 2], ['Yes', 'Maybe', 'No'])
plt.xticks([0, 1], ['Mental Health', 'Physical Health'])
plt.title('Interview Disclosure Willingness Comparison\n(Bubble Size = Percentage)')
plt.xlabel('Health Type')
plt.ylabel('Willingness Level')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

#--------------------------------------------------
# 7.Point plots of mean feature value per stress level
from sklearn.preprocessing import LabelEncoder

df = df.drop(columns=['Timestamp'], errors='ignore')

# Encode all object columns except 'stress_level'
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include='object'):
    if col != 'stress_level':
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

# Encode stress_level
le = LabelEncoder()
df_encoded['stress_level_encoded'] = le.fit_transform(df['stress_level'])  # high/moderate/low -> 0/1/2

# POINTPLOT
melted_df = df_encoded.drop(columns=['stress_level'], errors='ignore').melt(
    id_vars='stress_level_encoded', var_name='Feature', value_name='Value'
)

plt.figure(figsize=(16, 8))
sns.pointplot(data=melted_df, x='Feature', y='Value', hue='stress_level_encoded',
              palette='pastel', dodge=0.3, errorbar='ci', markers='o')

plt.title('📈 Mean Feature Value per Stress Level (Point Plot)', fontsize=14)
plt.xlabel('Feature')
plt.ylabel('Mean Encoded Value')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Stress Level\n(Encoded)', loc='upper right')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

#--------------------------------------------------
# 8. CORRELATION HEATMAP
# Purpose: Identify hidden relationships between all numeric variables
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm',
            vmin=-1, vmax=1, center=0)
plt.title('Feature Correlation Matrix\n(Revealing Hidden Relationships)')
plt.tight_layout()
plt.show()
#---------------------------------------------
#model selection
#---------------------------------------------
# ✅ Random forest:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Define features and target
selected_features = [
    'Age', 'self_employed', 'family_history', 'treatment',
    'remote_work', 'tech_company', 'mental_vs_physical', 'work_interfere', 'benefits'
    , 'anonymity', 'no_employees'
]
# Convert categorical features to numeric
from sklearn.preprocessing import LabelEncoder
for col in selected_features:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
X = df[selected_features]
y = df['stress_level']

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)

# Evaluation
print("Classification Report For Random Forest:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=rf_model.classes_,
             yticklabels=rf_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest')
plt.tight_layout()
plt.show()
#-----------------------------------------------------------------
#✅ XGBoost
from xgboost import XGBClassifier

# Encode target variable (stress_level)
le = LabelEncoder()
y_encoded = le.fit_transform(df['stress_level'])  # Converts to 0, 1, 2

# Optional: Keep track of label mapping
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label Mapping:", label_mapping)  # e.g., {'high': 0, 'low': 1, 'moderate': 2}

# Split data
X = df[selected_features]
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# Evaluation
print(" XGBoost Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

#----------------------------------------------------------------
#✅ Kmeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Encode stress_level target
le = LabelEncoder()
df['stress_level_encoded'] = le.fit_transform(df['stress_level'])

# Drop any rows with NaNs in numeric features
numeric_df = df.select_dtypes(include=['int64', 'float64']).drop(columns=['stress_level_encoded'])
numeric_df = numeric_df.dropna()

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

# Fit KMeans
print("\n🔹 KMeans Clustering (3 clusters):")
kmeans = KMeans(n_clusters=3, random_state=42)
df = df.loc[numeric_df.index]  # Align df to filtered numeric_df
df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

# Cross-tabulation with true labels
print(pd.crosstab(df['kmeans_cluster'], df['stress_level']))

# PCA for 2D Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                hue=df['kmeans_cluster'],
                palette='Set2', s=60)
plt.title("KMeans Clustering (PCA 2D Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# ✅ Decision Tree Classifier (Supervised)
from sklearn.tree import DecisionTreeClassifier
print("\n🔹 Decision Tree Classifier Results:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)

# Classification Report
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix: Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
