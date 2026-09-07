"""
generate_datasets.py
Author: Cash Johnson
Builds the four CSV datasets used in the Module 3 assignment.

Data source note (also cited in each part's code):
All four datasets were AI-generated following the instructor's Module 3
dataset instructions (generate 100+ realistic records with an AI tool,
then load them from CSV). Values are calibrated to real-world reference
points so the data behaves realistically:
- house_prices.csv: San Jose, CA metro pricing (Zillow Home Value Index,
  ~$700-950 per sq ft downtown, lower in suburbs/rural edges of the metro)
- customer_churn.csv: patterned after the IBM Telco Customer Churn dataset
  on Kaggle (kaggle.com/datasets/blastchar/telco-customer-churn)
- customer_data.csv: patterned after the Mall Customer Segmentation dataset
  on Kaggle (kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- housing_demand.csv: monthly home-sales counts shaped like Santa Clara
  County MLS trends (seasonal spring peak, winter dip, mild growth)
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---------- Part 1: house prices (150 records) ----------
n = 150
locations = rng.choice(['Downtown', 'Suburb', 'Rural'], size=n, p=[0.35, 0.45, 0.20])
sqft = np.round(rng.normal(1900, 550, n).clip(650, 4200), 0)
base = {'Downtown': 250000, 'Suburb': 180000, 'Rural': 90000}
ppsf = {'Downtown': 520, 'Suburb': 430, 'Rural': 310}
price = np.array([base[l] for l in locations]) + sqft * np.array([ppsf[l] for l in locations])
price = np.round(price + rng.normal(0, 60000, n), -3).clip(150000)
pd.DataFrame({'square_footage': sqft.astype(int),
              'location': locations,
              'price': price.astype(int)}).to_csv('house_prices.csv', index=False)

# ---------- Part 2: customer churn (200 records) ----------
n = 200
age = rng.integers(18, 72, n)
usage = np.round(rng.gamma(3, 12, n).clip(1, 120), 1)
purchase = np.round(rng.gamma(4, 55, n).clip(20, 900), 2)
calls = rng.poisson(2.5, n).clip(0, 12)
region = rng.choice(['North', 'South', 'East', 'West'], size=n)
# churn is likelier with low usage, low spend, and many service calls
logit = 1.2 - 0.045 * usage - 0.006 * purchase + 0.55 * calls + rng.normal(0, 0.8, n)
churn = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
pd.DataFrame({'age': age, 'monthly_usage_hours': usage, 'purchase_amount': purchase,
              'customer_service_calls': calls, 'region': region,
              'churn': churn}).to_csv('customer_churn.csv', index=False)

# ---------- Part 3: segmentation (180 records, 3 natural groups) ----------
groups = [
    # (annual_spending mean, purchase_freq mean, age mean, count)
    (4200, 18, 34, 60),   # high spenders
    (1500, 22, 27, 60),   # frequent budget buyers
    (600, 4, 52, 60),     # low-engagement
]
rows = []
for spend_mu, freq_mu, age_mu, cnt in groups:
    rows.append(pd.DataFrame({
        'annual_spending': np.round(rng.normal(spend_mu, spend_mu * 0.22, cnt).clip(100), 2),
        'purchase_frequency': rng.normal(freq_mu, 4, cnt).clip(1).round(0).astype(int),
        'age': rng.normal(age_mu, 7, cnt).clip(18, 80).round(0).astype(int),
        'region': rng.choice(['North', 'South', 'East', 'West'], size=cnt),
    }))
seg = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)
seg.to_csv('customer_data.csv', index=False)

# ---------- Extra credit: monthly housing demand (48 months) ----------
months = np.arange(1, 49)
trend = 820 + 3.2 * months                       # mild long-run growth
season = 130 * np.sin((months - 3) * 2 * np.pi / 12)  # spring peak, winter dip
sales = np.round(trend + season + rng.normal(0, 45, 48)).astype(int)
pd.DataFrame({'month': months, 'sales': sales}).to_csv('housing_demand.csv', index=False)

print('Wrote house_prices.csv, customer_churn.csv, customer_data.csv, housing_demand.csv')
