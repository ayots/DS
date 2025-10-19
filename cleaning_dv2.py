# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ 1. Load and Visualise Messy Dataset ------------------

df_raw = pd.read_csv('heart_disease_uci.csv')

# Heatmap of missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df_raw.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Raw Dataset')
plt.tight_layout()
plt.show()

# Bar chart of missing values
missing_counts = df_raw.isnull().sum()
plt.figure(figsize=(12, 6))
missing_counts.plot(kind='bar', color='salmon')
plt.title('Missing Values per Column')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------ 2. Data Cleaning ------------------

df = df_raw.drop_duplicates()
df = df[df['num'].notnull()].copy()
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df.drop(columns=['id', 'dataset', 'num'], inplace=True)
df.replace('?', np.nan, inplace=True)

# Fill 'fbs' and 'thal' with defaults
df['fbs'] = df['fbs'].fillna('FALSE')
df['thal'] = df['thal'].fillna('normal')

# Convert categorical columns to strings and fill missing
cat_cols = df.select_dtypes(include=['object', 'bool']).columns
for col in cat_cols:
    df[col] = df[col].astype(str).fillna(df[col].mode()[0])

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Convert numerical columns to float and fill missing
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# Scale numerical features (excluding target)
from sklearn.preprocessing import StandardScaler
features_to_scale = df.drop(columns=['target']).select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Export cleaned dataset
df.to_csv('heart_disease_cleaned.csv', index=False)

# ------------------ 3. Visualise Cleaned Dataset ------------------

plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values After Cleaning')
plt.tight_layout()
plt.show()

selected = ['age', 'chol', 'thalch', 'oldpeak', 'target']
df_pair = df[selected].copy()
df_pair['target'] = df_pair['target'].astype(str)

sns.pairplot(df_pair, hue='target', palette='Set2', diag_kind='kde')
plt.suptitle('Pairplot of Selected Features by Heart Disease Class', y=1.02)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.violinplot(x='target', y='chol', hue='target', data=df, palette='Set2', legend=False)
plt.title('Cholesterol Distribution by Heart Disease Class')
plt.xlabel('Heart Disease Class')
plt.ylabel('Cholesterol Level')
plt.tight_layout()
plt.show()

# ------------------ 4. Model Training and Evaluation ------------------

X = df.drop('target', axis=1)
y = df['target'].astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results)
results_df.to_csv('model_comparison.csv', index=False)
print("Model evaluation results:")
print(results_df)

# ------------------ 5. Bar Chart Comparison ------------------

plt.figure(figsize=(10, 6))
x = results_df['Model']
f1 = results_df['F1-score']
roc = results_df['ROC-AUC']
bar_width = 0.35
index = np.arange(len(x))

plt.bar(index, f1, bar_width, label='F1-score', color='skyblue')
plt.bar(index + bar_width, roc, bar_width, label='ROC-AUC', color='salmon')

plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Comparison: F1-score vs ROC-AUC')
plt.xticks(index + bar_width / 2, x, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('model_scores.png')
plt.show()
