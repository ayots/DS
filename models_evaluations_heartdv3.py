# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Import core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned heart disease dataset
df = pd.read_csv('heart_disease_cleaned.csv')

# Separate features and target variable
X = df.drop('target', axis=1)
y = df['target'].astype(int)

# Split the dataset into 80% training and 20% testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Import supervised learning models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Import evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Define supervised models
models = {
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Train and evaluate each supervised model
results = []
feature_importances = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Store evaluation metrics
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })

    # Store feature importance if available
    if hasattr(model, 'feature_importances_'):
        feature_importances[name] = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Display evaluation results in a table
results_df = pd.DataFrame(results)
print("Supervised Model Evaluation Results:")
print(results_df)

# Plot feature importance for each model
for name, importance in feature_importances.items():
    plt.figure(figsize=(8, 5))
    importance.plot(kind='bar', color='steelblue')
    plt.title(f'Feature Importance for {name}')
    plt.ylabel('Importance Score')
    plt.xlabel('Feature')
    plt.tight_layout()
    plt.savefig(f'{name}_feature_importance.png')
    plt.show()

# Import unsupervised clustering models
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

# Reduce feature space to 2D using PCA for visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply K-Means clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Visualise K-Means clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='Set1')
plt.title('K-Means Clustering Visualised with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.tight_layout()
plt.savefig('kmeans_pca.png')
plt.show()

# Visualise DBSCAN clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=dbscan_labels, palette='Set2')
plt.title('DBSCAN Clustering Visualised with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.tight_layout()
plt.savefig('dbscan_pca.png')
plt.show()

# Evaluate clustering performance using Adjusted Rand Index
ari_kmeans = adjusted_rand_score(y, kmeans_labels)
ari_dbscan = adjusted_rand_score(y, dbscan_labels)

print(f"\nAdjusted Rand Index for K-Means: {ari_kmeans:.4f}")
print(f"Adjusted Rand Index for DBSCAN: {ari_dbscan:.4f}")

# Identify best supervised model based on F1-score
best_supervised = results_df.sort_values(by='F1-score', ascending=False).iloc[0]
print(f"\nBest Supervised Model: {best_supervised['Model']} (F1-score = {best_supervised['F1-score']:.4f})")

# Identify best unsupervised model based on ARI
best_unsupervised = 'K-Means' if ari_kmeans > ari_dbscan else 'DBSCAN'
best_ari = max(ari_kmeans, ari_dbscan)
print(f"Best Unsupervised Model: {best_unsupervised} (ARI = {best_ari:.4f})")

# Final comparison and conclusion
if best_supervised['F1-score'] > best_ari:
    print(f"\nFinal Winner: {best_supervised['Model']} — best predictive accuracy and clinical interpretability.")
else:
    print(f"\nFinal Winner: {best_unsupervised} — best clustering alignment with true labels.")
