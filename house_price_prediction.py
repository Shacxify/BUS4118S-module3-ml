# Part 1: House Price Prediction (Linear Regression)
# Author: Cash Johnson
#
# Data source: house_prices.csv (150 records). AI-generated per the
# instructor's Module 3 dataset instructions, calibrated to San Jose, CA
# metro pricing (Zillow Home Value Index reference points: roughly
# $500+/sq ft downtown, less in suburbs and the rural edges of the metro).
# See generate_datasets.py in this repo for exactly how it was built.
#
# Changes from starter code:
# - Replaced the 10-row hardcoded sample with a 150-row CSV loaded via pandas
# - Added test-set evaluation (R^2) so we know the model generalizes
# - Added plain-language explanations of every coefficient in the output

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset from CSV (module 2 skill: reading files instead of hardcoding data)
df = pd.read_csv('house_prices.csv')
print(f"Loaded {len(df)} records from house_prices.csv\n")

# Features and target
X = df[['square_footage', 'location']]
y = df['price']

# Preprocessing: one-hot encode the categorical 'location' column,
# pass square_footage through unchanged
preprocessor = ColumnTransformer(
    transformers=[('location', OneHotEncoder(sparse_output=False), ['location'])],
    remainder='passthrough')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split so we can check performance on houses the model never saw
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# How well does it generalize?
r2 = r2_score(y_test, model.predict(X_test))
print(f"Test set R^2: {r2:.3f} (share of price variation the model explains)\n")

# Predict a 2000 sq ft house in Downtown
new_house = pd.DataFrame({'square_footage': [2000], 'location': ['Downtown']})
predicted_price = model.predict(new_house)[0]
print(f"Predicted price for a 2000 sq ft house in Downtown: ${predicted_price:,.0f}\n")

# Coefficients, explained in plain language
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
coefs = model.named_steps['regressor'].coef_
print("Model coefficients:")
for name, c in zip(feature_names, coefs):
    print(f"  {name}: {c:,.2f}")

sqft_coef = coefs[list(feature_names).index('remainder__square_footage')]
print(f"""
What the coefficients mean:
- square_footage ({sqft_coef:,.0f}): every extra square foot adds about
  ${sqft_coef:,.0f} to the predicted price, holding location constant.
  Same idea as the "$200 per square foot" example from lecture, just at
  San Jose prices.
- The three location coefficients are premiums/discounts relative to the
  model's baseline. Downtown carries the largest positive effect and Rural
  the largest negative one, so the SAME house is worth more Downtown than
  in a Rural area purely because of where it sits.
""")
