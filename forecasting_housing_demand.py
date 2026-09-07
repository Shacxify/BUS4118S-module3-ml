# Extra Credit: Housing Demand Forecasting
# Author: Cash Johnson
#
# Data source: housing_demand.csv (48 monthly records). AI-generated per the
# instructor's Module 3 dataset instructions, shaped like Santa Clara County
# monthly home-sales trends (spring peak, winter dip, mild long-run growth).
# See generate_datasets.py in this repo for exactly how it was built.
#
# Changes from starter code:
# - 48 months of data loaded from CSV instead of a tiny sample
# - Added a month-of-year seasonality feature on top of the linear trend,
#   because housing demand is strongly seasonal and a straight line misses it
# - Forecast printed as a table and saved to a labeled chart (demand_forecast.png)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load and preprocess
df = pd.read_csv('housing_demand.csv')  # columns: month (1..N), sales
df = df.dropna().sort_values('month')
print(f"Loaded {len(df)} months of sales data from housing_demand.csv\n")

# Feature engineering: linear trend + sine/cosine terms for annual seasonality
def make_features(months):
    m = np.asarray(months, dtype=float)
    return np.column_stack([
        m,                                # long-run trend
        np.sin(2 * np.pi * m / 12),       # yearly cycle
        np.cos(2 * np.pi * m / 12),
    ])

X = make_features(df['month'])
y = df['sales']

model = LinearRegression()
model.fit(X, y)
print(f"In-sample R^2: {model.score(X, y):.3f}")
print(f"Trend coefficient: {model.coef_[0]:+.2f} sales per month "
      "(the market is slowly growing)\n")

# Forecast the next 6 months
last = int(df['month'].max())
future_months = np.arange(last + 1, last + 7)
predictions = model.predict(make_features(future_months))

print("6-month forecast:")
for m, p in zip(future_months, predictions):
    print(f"  Month {m}: {p:,.0f} homes")

# Visualize
plt.figure(figsize=(10, 5))
plt.plot(df['month'], y, label='Historical sales')
plt.plot(future_months, predictions, 'r--o', label='6-month forecast')
plt.xlabel('Month')
plt.ylabel('Homes sold')
plt.title('Housing Demand: History and 6-Month Forecast')
plt.legend()
plt.tight_layout()
plt.savefig('demand_forecast.png')
plt.close()
print("\nSaved chart to demand_forecast.png")

print("""
Assumptions:
- Past trend and seasonality continue; no shocks (rate spikes, new inventory).
- One region aggregated together; demand = closed sales.

Challenges:
- Plain linear regression on month alone flatlines through the seasonal
  swings; the sine/cosine terms were needed to track the spring/winter cycle.
- Only 4 years of history, so the trend estimate is sensitive to outliers.

Potential improvements:
- Add drivers: mortgage rates, active listings, median price.
- Try models built for time series (SARIMA, Prophet, gradient boosting).
- Use a holdout of the last 6 months to validate forecast error before trusting it.
""")
