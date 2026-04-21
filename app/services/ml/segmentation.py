import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

@dataclass
class SegmentationResult:
    """
    Everything the segmentation stage produces.
    """
    df_rfm_labelled : pd.DataFrame # RFM table with clusters column added
    n_clusters : int # No. of clusters
    silhouette_score : float # Quality score
    cluster_profiles : list # summary per clusters

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
    df_rfm["clusters"] = km_final.fit_predict(X)
    profiles = _build_profiles(df_rfm, best_k)
    return SegmentationResult(
        df_rfm_labelled = df_rfm,
        n_clusters = best_k,
        silhouette_score = round(float(best_score), 4),
        cluster_profiles = profiles,
    )

def _build_profiles(df_rfm : pd.DataFrame, n_clusters : int) -> list :
    """
    Builds a human-readable summary for each cluster.
    Labesls are assigned based on relative RFM Values
    """
    profiles = []
    overall_r = df_rfm["recency"].mean()
    overall_f = df_rfm["frequency"].mean()
    overall_m = df_rfm["monetary"].mean()
    for c in range(n_clusters):
        subset = df_rfm[df_rfm["clusters"] == c]
        r_mean = subset["recency"].mean()
        f_mean = subset["frequency"].mean()
        m_mean = subset["monetary"].mean()
        pct = len(subset)/len(df_rfm) * 100
        if r_mean < overall_r and f_mean >= overall_f:
            label = "Champions"
        elif r_mean < overall_r and m_mean >= overall_m:
            label = "Loyal High-Value"
        elif r_mean > overall_r * 1.5:
            label = "At-Risk / Lapsed"
        elif f_mean < overall_f * 0.5:
            label = "Low Engagement"
        else:
            label = f"Segment {c + 1}"
        profiles.append({
            "cluster_id" : c,
            "labels" : label,
            "size" : int(len(subset)),
            "pct_of_customers" : round(float(pct),1),
            "avg_recency_days" : round(float(r_mean),1),
            "avg_frequency" : round(float(f_mean), 1),
            "avg_monetary" : round(float(m_mean),2)
        })
    return profiles