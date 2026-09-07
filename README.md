# Module 3: Machine Learning Coding Exercise

Cash Johnson | TECH 171

Four scikit-learn exercises, each running on a realistic 100+ record dataset loaded from CSV. Datasets were AI-generated per the instructor's Module 3 dataset instructions and calibrated to real-world reference points (sources cited in each script's header comment). `generate_datasets.py` shows exactly how every CSV was built.

## Files

| Part | Script | Dataset | Output |
|---|---|---|---|
| 1. House Price Prediction (Linear Regression) | `house_price_prediction.py` | `house_prices.csv` (150 rows) | Predicted price for a 2000 sq ft Downtown house, coefficients explained |
| 2. Customer Churn (Logistic Regression) | `customer_churn_prediction.py` | `customer_churn.csv` (200 rows) | Churn probability, 0.5 threshold classification, business interpretation |
| 3. Customer Segmentation (K-Means) | `customer_segmentation.py` | `customer_data.csv` (180 rows) | `elbow_plot.png`, cluster profiles, marketing strategies, `customer_segments.csv` |
| Extra credit: Demand Forecasting | `forecasting_housing_demand.py` | `housing_demand.csv` (48 months) | 6-month forecast + `demand_forecast.png` |

## Run it

```
pip install pandas numpy scikit-learn matplotlib
python house_price_prediction.py
python customer_churn_prediction.py
python customer_segmentation.py
python forecasting_housing_demand.py
```

## Improvements over the starter code

- All hardcoded sample data replaced with CSVs read via pandas (100+ records each)
- Train/test evaluation added (R^2 for regression, accuracy for churn)
- Every coefficient explained in plain language in the output
- Elbow method extended to K=1..8 with printed inertia values to justify K=3
- Forecasting model adds seasonality features (sine/cosine) on top of the linear trend
