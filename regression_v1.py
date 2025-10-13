import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Generate synthetic dataset
np.random.seed(42)
n = 50

data = pd.DataFrame({
    'num_links': np.random.randint(0, 10, size=n),
    'has_urgent_words': np.random.randint(0, 2, size=n),
    'domain_age_days': np.random.randint(1, 1000, size=n),
    'has_attachment': np.random.randint(0, 2, size=n)
})

# Generate phishing_score as a weighted combination of features + noise
weights = np.array([0.1, 0.3, -0.0005, 0.2])
features = data[['num_links', 'has_urgent_words', 'domain_age_days', 'has_attachment']].values
noise = np.random.normal(0, 0.05, size=n)
phishing_score = np.clip(np.dot(features, weights) + noise, 0, 1)

data['phishing_score'] = phishing_score

# Step 2: Split into training and test sets
X = data[['num_links', 'has_urgent_words', 'domain_age_days', 'has_attachment']]
y = data['phishing_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict on test set
y_pred = model.predict(X_test)

# Step 5: Print model coefficients and predictions
print("Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

print("\nTest Predictions:")
for i, (true, pred) in enumerate(zip(y_test.values, y_pred)):
    print(f"Email {i+1}: True Score = {true:.3f}, Predicted Score = {pred:.3f}")
