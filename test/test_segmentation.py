from app.services.ml.preprocessing import run_preprocessing
from app.services.ml.segmentation import run_segmentation

# Load and preprocess
with open("sample_data.csv", "rb") as f:
    file_bytes = f.read()

prep   = run_preprocessing(file_bytes, "text/csv")
result = run_segmentation(prep.df_rfm)

print(f"Optimal clusters found: {result.n_clusters}")
print(f"Silhouette score:       {result.silhouette_score}")
print(f"\n=== Cluster Profiles ===")
for p in result.cluster_profiles:
    print(f"\n  [{p['labels']}]")
    print(f"    Customers:      {p['size']} ({p['pct_of_customers']}%)")
    print(f"    Avg recency:    {p['avg_recency_days']} days ago")
    print(f"    Avg frequency:  {p['avg_frequency']} orders")
    print(f"    Avg spend:      ${p['avg_monetary']}")