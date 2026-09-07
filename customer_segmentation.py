# Part 3: Customer Segmentation (K-Means Clustering)
# Author: Cash Johnson
#
# Data source: customer_data.csv (180 records). AI-generated per the
# instructor's Module 3 dataset instructions, patterned after the Mall
# Customer Segmentation dataset on Kaggle
# (kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).
# See generate_datasets.py in this repo for exactly how it was built.
#
# Changes from starter code:
# - Replaced the hardcoded sample with a 180-row CSV loaded via pandas
# - Extended the elbow search to K=1..8 and printed the inertia values
#   so the K choice is justified with numbers, not just a picture
# - Cluster strategies are generated from each cluster's actual averages

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset from CSV
df = pd.read_csv('customer_data.csv')
print(f"Loaded {len(df)} records from customer_data.csv\n")

# Scale the numeric features so no single feature (like spending, which is
# in the thousands) dominates the distance calculation
features = ['annual_spending', 'purchase_frequency', 'age']
X_scaled = StandardScaler().fit_transform(df[features])

# Elbow method: fit K-Means for K=1..8 and track inertia
K_range = range(1, 9)
inertia = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(K_range), inertia, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.savefig('elbow_plot.png')
plt.close()

print("Inertia by K:")
for k, i in zip(K_range, inertia):
    print(f"  K={k}: {i:,.1f}")

# Justifying K: inertia drops steeply through K=3, then flattens out.
# Past 3, each extra cluster buys very little, so K=3 is the elbow.
optimal_k = 3
print(f"\nChosen K = {optimal_k}: the inertia curve bends hardest at 3 "
      "(big drops from 1->2->3, small drops after). See elbow_plot.png.\n")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze each cluster's average profile
cluster_summary = df.groupby('cluster')[features].mean().round(2)
cluster_summary['customers'] = df['cluster'].value_counts().sort_index()
print("Cluster characteristics (averages):")
print(cluster_summary)

# Marketing strategy matched to what each cluster actually looks like
print("\nTargeted strategies:")
for cluster in range(optimal_k):
    spend = cluster_summary.loc[cluster, 'annual_spending']
    freq = cluster_summary.loc[cluster, 'purchase_frequency']
    print(f"\nCluster {cluster} (avg spend ${spend:,.0f}, {freq:.0f} purchases/yr):")
    if spend > 3000:
        print("  High spenders: VIP/loyalty tier, early access to new products,")
        print("  exclusive promotions. Protect these relationships first.")
    elif freq > 12:
        print("  Frequent budget buyers: bundle deals, bulk discounts, or a")
        print("  subscription plan to lock in their habit and raise basket size.")
    else:
        print("  Low-engagement customers: personalized re-engagement emails and")
        print("  a first-order-back incentive; cheap channels only, low expected LTV.")

# Save results
df.to_csv('customer_segments.csv', index=False)
print("\nSaved cluster assignments to customer_segments.csv "
      "and the elbow chart to elbow_plot.png")
