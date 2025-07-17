# ✅ Phase 3: Workplace Stress Analysis training and model testing Suite
# ----------------------------------
#  1. Import Libraries
# ----------------------------------
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# ----------------------------------
#  2. Load and Prepare Data
# ----------------------------------
df = pd.read_csv("cleaned_survey.csv")

# Encode target variable (stress level)
le = LabelEncoder()
df['stress_level_encoded'] = le.fit_transform(df['stress_level'])  # high/moderate/low → numeric

# Define selected features (numerical + encoded categoricals)
selected_features = [
    'Age', 'self_employed', 'family_history', 'treatment',
    'remote_work', 'tech_company', 'mental_vs_physical', 'work_interfere',
    'benefits', 'anonymity', 'no_employees'
]

X = df[selected_features]
y = df['stress_level_encoded']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------
#  3. Basic Random Forest Model
# ----------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ----------------------------------
#  4. GridSearchCV Optimization
# ----------------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)

best_grid_model = grid.best_estimator_
print("\n✅ GridSearchCV Best Params:", grid.best_params_)

# Feature Importance (GridSearchCV)
grid_importances = best_grid_model.feature_importances_
indices = np.argsort(grid_importances)[::-1]
sorted_features = [X.columns[i] for i in indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=grid_importances[indices], y=sorted_features)
plt.title("Feature Importance - GridSearchCV Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ----------------------------------
#  5. RandomizedSearchCV Optimization
# ----------------------------------
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    verbose=1,
    n_jobs=-1,
    scoring='f1_weighted',
    random_state=42
)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

print("\n✅ RandomizedSearchCV Best Params:")
print(random_search.best_params_)

# Feature Importance (RandomizedSearchCV)
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [X.columns[i] for i in indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=sorted_features)
plt.title("Feature Importance - RandomizedSearchCV Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ----------------------------------
#  6. Evaluate and Compare All Models
# ----------------------------------
models = {
    "Basic RandomForest": rf_model,
    "GridSearchCV RF": best_grid_model,
    "RandomizedSearchCV RF": best_model
}

f1_scores = {}

for name, model in models.items():
    print(f"\n Evaluation for: {name}")

    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    acc = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average='weighted')
    f1_train = f1_score(y_train, y_train_pred, average='weighted')
    f1_scores[name] = f1_test

    print(f"- Accuracy: {acc:.4f}")
    print(f"- Weighted F1 (Test): {f1_test:.4f}")
    print(f"- Weighted F1 (Train): {f1_train:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Overfitting Check
    print(" Overfitting Check:")
    if f1_train > f1_test + 0.1:
        print("⚠ Overfitting detected")
    elif f1_test > f1_train + 0.1:
        print("⚠ Underfitting detected")
    else:
        print("✅ Balanced model (no significant over/underfitting)")

# ----------------------------------
#  7. Best Model Summary
# ----------------------------------
best_model_name = max(f1_scores, key=f1_scores.get)
print(f"\n Best Performing Model: {best_model_name} with F1 Score = {f1_scores[best_model_name]:.4f}")
#------------------------------------------------
# 8. Save model and label encoder
#------------------------------------------------
# Save the model
joblib.dump(best_model, 'stress_prediction_model.pkl')

# Save LabelEncoder if needed
joblib.dump(le, 'label_encoder.pkl')
