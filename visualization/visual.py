# ✅ Phase 2: Workplace Stress Analysis Visualization Suite
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
df = pd.read_csv("cleaned_survey.csv")
sns.set(style="whitegrid")

# =============================================
# 1. STRESS LEVEL DISTRIBUTION (Overview)
# Purpose: Show overall prevalence of stress levels in the workforce
# =============================================
plt.figure(figsize=(6, 6))
stress_counts = df['stress_level'].value_counts()
plt.pie(stress_counts, labels=stress_counts.index, autopct='%1.1f%%',
        colors=sns.color_palette("pastel"))
plt.title('Overall Stress Level Distribution in Workforce')
plt.show()

# =============================================
# 2. AGE ANALYSIS FOR HIGH-STRESS INDIVIDUALS
# Purpose: Identify if specific age groups are more vulnerable to high stress
# =============================================
plt.figure(figsize=(8, 5))
sns.histplot(df[df['stress_level'] == 'high']['Age'], bins=20,
             color='crimson', kde=True)
plt.title('Age Distribution of High-Stress Employees')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# =============================================
# 3. EMPLOYMENT TYPE VS STRESS (Tech vs Non-Tech)
# Purpose: Compare stress levels between tech and non-tech workers
# =============================================
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

# =============================================
# 4. REMOTE WORK IMPACT ON STRESS
# Purpose: Examine if remote work correlates with stress levels
# =============================================
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

# =============================================
# 5. HEALTH DISCLOSURE WILLINGNESS COMPARISON
# Purpose: Reveal stigma differences between mental/physical health
# =============================================
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

# =============================================
# 6. CORRELATION HEATMAP
# Purpose: Identify hidden relationships between all numeric variables
# =============================================
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm',
            vmin=-1, vmax=1, center=0)
plt.title('Feature Correlation Matrix\n(Revealing Hidden Relationships)')
plt.tight_layout()
plt.show()