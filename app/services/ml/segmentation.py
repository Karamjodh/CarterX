import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

@dataclass
class SegmentationResult:
    df_rfm_labelled:  pd.DataFrame   # ← add this
    n_clusters:       int
    silhouette_score: float
    cluster_profiles: list

def run_segmentation(df_rfm : pd.DataFrame) -> SegmentationResult:
    """
    Finds natural customer segments using KMeans Clustering
    Takes the RFM table from preprocessing and returns
    labelled clusters with human-readable profiles
    """
    features = ["recency_scaled", "frequency_scaled", "monetary_scaled"]
    X = df_rfm[features].values
    best_k = 3
    best_score = -1
    max_k = min(8, len(X)// 10+1)
    for k in range(2, max_k+1):
        km = KMeans(n_clusters = k, random_state = 42, n_init = 10)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X,labels)
        if score > best_score:
            best_score = score
            best_k = k
    km_final = KMeans(n_clusters = best_k, random_state = 42, n_init = 10)
    df_rfm = df_rfm.copy()
    df_rfm["cluster"] = km_final.fit_predict(X)
    profiles = _build_profiles(df_rfm, best_k)
    return SegmentationResult(
    df_rfm_labelled  = df_rfm,       # ← add this
    n_clusters       = best_k,
    silhouette_score = round(float(best_score), 4),
    cluster_profiles = profiles,
)

def _build_profiles(df_rfm: pd.DataFrame, n_clusters: int) -> list:
    profiles   = []
    overall_r  = df_rfm["recency"].mean()
    overall_f  = df_rfm["frequency"].mean()
    overall_m  = df_rfm["monetary"].mean()
    max_r      = df_rfm["recency"].max()

    for c in range(n_clusters):
        subset = df_rfm[df_rfm["cluster"] == c]
        r_mean = subset["recency"].mean()
        f_mean = subset["frequency"].mean()
        m_mean = subset["monetary"].mean()
        pct    = len(subset) / len(df_rfm) * 100

        # Recency percentile — lower recency = more recent = better
        r_pct = r_mean / max_r  # 0 = bought today, 1 = bought longest ago

        # More granular labeling with better coverage
        if r_pct < 0.2 and f_mean >= overall_f and m_mean >= overall_m:
            label = "Champions"
        elif r_pct < 0.3 and f_mean >= overall_f:
            label = "Loyal Customers"
        elif r_pct < 0.3 and f_mean < overall_f * 0.5:
            label = "New Customers"
        elif r_pct < 0.4 and m_mean >= overall_m * 1.2:
            label = "Potential Loyalists"
        elif r_pct > 0.7 and f_mean >= overall_f:
            label = "At Risk"
        elif r_pct > 0.7 and f_mean < overall_f:
            label = "Hibernating"
        elif r_pct > 0.85:
            label = "Lost Customers"
        elif m_mean >= overall_m * 1.5:
            label = "High Value"
        elif f_mean < overall_f * 0.4:
            label = "Low Engagement"
        else:
            label = f"Segment {c + 1}"

        profiles.append({
            "cluster_id":        c,
            "label":             label,
            "size":              int(len(subset)),
            "pct_of_customers":  round(float(pct), 1),
            "avg_recency_days":  round(float(r_mean), 1),
            "avg_frequency":     round(float(f_mean), 1),
            "avg_monetary":      round(float(m_mean), 2),
        })

    return profiles