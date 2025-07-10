#Load and Explore the Data
import pandas as pd

# Load the dataset
df = pd.read_csv('survey.csv')

# Preview the data
print(df.head())

# Basic info
print(df.info())
#Data Cleaning
df.isnull().sum()
#drop missing values
df = df.dropna()
#Remove irrelevant or redundant columns
df = df.drop(['comments', 'state', 'Timestamp'], axis=1)
#Outlier Detection
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data=df, x='Age')
plt.show()

# Remove unrealistic ages
df = df[(df['Age'] >= 18) & (df['Age'] <= 60)]
#Feature Engineering
def label_stress(row):
    if row['work_interfere'] in ['Often', 'Always']:
        return 'high'
    elif row['work_interfere'] == 'Sometimes':
        return 'moderate'
    else:
        return 'low'

df['stress_level'] = df.apply(label_stress, axis=1)

