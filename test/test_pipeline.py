from app.services.ml.preprocessing    import run_preprocessing
from app.services.ml.segmentation     import run_segmentation
from app.services.ml.association_rules import run_association_rules

with open("sample_data.csv", "rb") as f:
    file_bytes = f.read()

# Stage 1
print("Running preprocessing...")
prep = run_preprocessing(file_bytes, "text/csv")
print(f"  Customers:    {prep.summary['total_customers']}")
print(f"  Transactions: {prep.summary['total_transactions']}")
print(f"  Revenue:      ${prep.summary['total_revenue']:,.2f}")

# Stage 2
print("\nRunning segmentation...")
seg = run_segmentation(prep.df_rfm)
print(f"  Clusters found:    {seg.n_clusters}")
print(f"  Silhouette score:  {seg.silhouette_score}")
print(f"  Profile keys: {seg.cluster_profiles[0].keys()}")
for p in seg.cluster_profiles:
    print(f"  [{p['label']}] — {p['size']} customers ({p['pct_of_customers']}%)")

# Stage 3
print("\nRunning association rules...")
assoc = run_association_rules(prep.df_basket)
print(f"  Rules found: {assoc.total_found}")
if assoc.rules:
    top = assoc.rules[0]
    ant = " + ".join(top["antecedents"])
    con = " + ".join(top["consequents"])
    print(f"  Top rule: {ant} → {con} (lift: {top['lift']}x)")

# Stage 4 — build the data dict the prompt builder expects
print("\nBuilding analysis data...")
analysis_data = {
    "summary": prep.summary,
    "segments": seg.cluster_profiles,
    "association_rules": assoc.rules,
    "forecasts": {},   # forecasting comes next phase
}

print("\nPipeline complete. Data ready for LLM report.")
print(f"  Segments:          {len(analysis_data['segments'])}")
print(f"  Association rules: {len(analysis_data['association_rules'])}")