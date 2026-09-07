# Module 3: Machine Learning Coding Exercise

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-11557C)
![Course](https://img.shields.io/badge/SJSU-TECH%20171-0055A2)
![Status](https://img.shields.io/badge/all%20scripts-passing-brightgreen)

Cash Johnson | TECH 171

Four scikit-learn exercises, each running on a realistic 100+ record dataset loaded from CSV. Datasets were AI-generated per the instructor's Module 3 dataset instructions and calibrated to real-world reference points (see Data sources below, also cited in each script's header comment). `generate_datasets.py` shows exactly how every CSV was built.

## Data sources

All four CSVs were AI-generated following the instructor's Module 3 dataset instructions (generate 100+ realistic records with an AI tool, then load them from CSV). Each one is calibrated to a real-world reference point so the data behaves realistically:

- `house_prices.csv` (150 rows): San Jose, CA metro pricing, based on Zillow Home Value Index reference points (roughly $500+ per sq ft downtown, less in suburbs and the rural edges of the metro)
- `customer_churn.csv` (200 rows): patterned after the [IBM Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- `customer_data.csv` (180 rows): patterned after the [Mall Customer Segmentation dataset on Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- `housing_demand.csv` (120 months): monthly home-sales counts shaped like Santa Clara County MLS trends (spring peak, winter dip, mild long-run growth)

## Files

| Part | Script | Dataset | Output |
|---|---|---|---|
| 1. House Price Prediction (Linear Regression) | `house_price_prediction.py` | `house_prices.csv` (150 rows) | Predicted price for a 2000 sq ft Downtown house, coefficients explained |
| 2. Customer Churn (Logistic Regression) | `customer_churn_prediction.py` | `customer_churn.csv` (200 rows) | Churn probability, 0.5 threshold classification, business interpretation |
| 3. Customer Segmentation (K-Means) | `customer_segmentation.py` | `customer_data.csv` (180 rows) | `elbow_plot.png`, cluster profiles, marketing strategies, `customer_segments.csv` |
| Extra credit: Demand Forecasting | `forecasting_housing_demand.py` | `housing_demand.csv` (120 months) | 6-month forecast + `demand_forecast.png` |

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
