# ------------------ 1. Import Libraries ------------------

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ 2. Load and Inspect Raw Dataset ------------------

df = pd.read_csv('heart_disease_cleaned.csv')

# Display basic info and summary statistics
print("Initial Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# ------------------ 3. Data Quality Assessment ------------------

# Check for missing values
missing = df.isnull().sum()
print("\nMissing Values per Column:")
print(missing)

# Visualise missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.tight_layout()
plt.show()

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# Identify outliers using IQR method
numeric_cols = df.select_dtypes(include=np.number).columns
outlier_counts = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    outlier_counts[col] = len(outliers)

print("\nOutlier Counts per Column:")
print(outlier_counts)

# Visualise outliers with boxplots
plt.figure(figsize=(12, 8))
df[numeric_cols].boxplot()
plt.title('Boxplot of Numeric Features')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------ 4. Data Cleaning ------------------

# Drop duplicates
df = df.drop_duplicates()

# Fill missing values with median
df = df.fillna(df.median(numeric_only=True))

# Save cleaned dataset
df.to_csv('cleaned_heart.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_heart.csv'.")

# ------------------ 5. Load Cleaned Dataset ------------------

df = pd.read_csv('cleaned_heart.csv')
X = df.drop('target', axis=1)
y = df['target'].astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ 6. Supervised Modelling ------------------

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve

models = {
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = []
feature_importances = {}

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

    if hasattr(model, 'feature_importances_'):
        feature_importances[name] = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_prob):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.title(f'ROC Curve: {name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Tabular results
results_df = pd.DataFrame(results)
print("\nSupervised Model Evaluation Results:")
print(results_df)

# Feature importance plots
for name, importance in feature_importances.items():
    plt.figure(figsize=(8, 5))
    importance.plot(kind='bar', color='steelblue')
    plt.title(f'Feature Importance for {name}')
    plt.ylabel('Importance Score')
    plt.xlabel('Feature')
    plt.tight_layout()
    plt.show()

# ------------------ 7. Unsupervised Clustering ------------------

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

X_pca = PCA(n_components=2).fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# PCA plots
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='Set1')
plt.title('K-Means Clustering (PCA)')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=dbscan_labels, palette='Set2')
plt.title('DBSCAN Clustering (PCA)')
plt.show()

# Silhouette scores
silhouette_scores = {
    'K-Means': silhouette_score(X, kmeans_labels),
    'DBSCAN': silhouette_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else 0
}
plt.figure(figsize=(6, 4))
sns.barplot(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()), palette='pastel')
plt.title('Silhouette Scores')
plt.ylabel('Score')
plt.tight_layout()
plt.show()

# Cluster distribution
cluster_counts = pd.Series(kmeans_labels).value_counts()
plt.figure(figsize=(5, 4))
cluster_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('K-Means Cluster Distribution')
plt.ylabel('')
plt.tight_layout()
plt.show()

# ARI scores
ari_kmeans = adjusted_rand_score(y, kmeans_labels)
ari_dbscan = adjusted_rand_score(y, dbscan_labels)
print(f"\nAdjusted Rand Index for K-Means: {ari_kmeans:.4f}")
print(f"Adjusted Rand Index for DBSCAN: {ari_dbscan:.4f}")

# ------------------ 8. Final Evaluation ------------------

best_supervised = results_df.sort_values(by='F1-score', ascending=False).iloc[0]
print(f"\nBest Supervised Model: {best_supervised['Model']} (F1-score = {best_supervised['F1-score']:.4f})")

best_unsupervised = 'K-Means' if ari_kmeans > ari_dbscan else 'DBSCAN'
best_ari = max(ari_kmeans, ari_dbscan)
print(f"Best Unsupervised Model: {best_unsupervised} (ARI = {best_ari:.4f})")

if best_supervised['F1-score'] > best_ari:
    print(f"\nFinal Winner: {best_supervised['Model']} — best predictive accuracy and clinical interpretability.")
else:
    print(f"\nFinal Winner: {best_unsupervised} — best clustering alignment with true labels.")
