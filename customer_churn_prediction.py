# Part 2: Customer Churn Prediction (Logistic Regression)
# Author: Cash Johnson
#
# Data source: customer_churn.csv (200 records). AI-generated per the
# instructor's Module 3 dataset instructions, patterned after the IBM Telco
# Customer Churn dataset on Kaggle
# (kaggle.com/datasets/blastchar/telco-customer-churn).
# See generate_datasets.py in this repo for exactly how it was built.
#
# Changes from starter code:
# - Replaced the 10-row hardcoded sample with a 200-row CSV loaded via pandas
# - Added test-set accuracy so we know the model actually predicts
# - Added coefficient explanations and a business interpretation section

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset from CSV
df = pd.read_csv('customer_churn.csv')
print(f"Loaded {len(df)} records from customer_churn.csv "
      f"({df['churn'].mean():.0%} churned)\n")

num_features = ['age', 'monthly_usage_hours', 'purchase_amount', 'customer_service_calls']
cat_features = ['region']
X = df[num_features + cat_features]
y = df['churn']

# Preprocessing: scale numeric features, one-hot encode region
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(sparse_output=False), cat_features)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Test set accuracy: {accuracy:.0%}\n")

# Score a new customer
new_customer = pd.DataFrame({
    'age': [35],
    'monthly_usage_hours': [30.0],
    'purchase_amount': [220.00],
    'customer_service_calls': [3],
    'region': ['West']
})
churn_probability = model.predict_proba(new_customer)[0][1]
threshold = 0.5
at_risk = churn_probability > threshold

print(f"New customer churn probability: {churn_probability:.2f} "
      f"({churn_probability:.0%} chance they leave)")
print(f"Classification at {threshold} threshold: "
      f"{'AT RISK (churn = 1)' if at_risk else 'not at risk (churn = 0)'}\n")

# Coefficients, explained
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
coefs = model.named_steps['classifier'].coef_[0]
print("Model coefficients (positive = pushes toward churn):")
for name, c in zip(feature_names, coefs):
    print(f"  {name}: {c:+.3f}")

print("""
What this means for the business:
- The probability is the model's estimate of how likely this specific
  customer is to leave. A 0.70 means a 70% chance of churn.
- The 0.5 threshold is the cutoff for action: above it we treat the
  customer as at risk, below it we leave them alone.
- Customer service calls carry a strong positive coefficient (more calls,
  more churn risk), while usage hours and purchase amount are negative
  (engaged, paying customers stick around).
- Practical use: score the whole customer base monthly, then point
  retention spend (discounts, a check-in call, a win-back offer) only at
  the customers above the threshold. That is far cheaper than blanket
  offers, since keeping a customer costs less than acquiring a new one.
""")
